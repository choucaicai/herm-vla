"""Per-token frame routing followed by an exact sparse joint softmax."""

from __future__ import annotations

import math
from types import MethodType
import weakref
import warnings

import torch
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    ALL_ATTENTION_FUNCTIONS, apply_rotary_pos_emb, eager_attention_forward,
)


def _filter_legacy_dynamo_config_warning():
    # PyTorch 2.6 exports this unused legacy option even when its value is empty.
    warnings.filterwarnings(
        "ignore",
        message=r"^Skipping serialization of skipfiles_inline_module_allowlist value \{\}$",
        category=UserWarning,
        module=r"^torch\.utils\._config_module$",
    )


_filter_legacy_dynamo_config_warning()


_compiled_flex = None
_QUERY_BLOCK = 64
_KEY_BLOCK = 64
_KERNEL_OPTIONS = {
    "BLOCK_M": 32, "BLOCK_N": 64,
    "BLOCK_M1": 32, "BLOCK_N1": 64,
    "BLOCK_M2": 64, "BLOCK_N2": 32,
    "FORCE_USE_FLEX_ATTENTION": True,
}


def route_frames(query, means, lengths, topk, history_query_mask):
    """Score history for enabled valid queries; -1 denotes no historical read."""
    batch, heads, sequence, _ = query.shape
    if history_query_mask.shape != (batch, sequence):
        raise ValueError("history_query_mask must have shape [batch, sequence]")
    groups = query.shape[1] // means.shape[1]
    expanded_means = means.repeat_interleave(groups, dim=1)
    count = min(int(topk), means.shape[2])
    route = torch.full(
        (batch, heads, sequence, count), -1, dtype=torch.long, device=query.device,
    )
    # The caller enables every nonpadding token, including memory-read tokens.
    # Prompt lengths and available history can differ across batch slots.
    for sample in range(batch):
        indices = history_query_mask[sample].nonzero(as_tuple=True)[0]
        if indices.numel() == 0:
            continue
        selected_queries = query[sample].detach().index_select(1, indices).float()
        scores = torch.matmul(selected_queries, expanded_means[sample].transpose(-1, -2))
        scores *= query.shape[-1] ** -0.5
        scores.masked_fill_(lengths[sample][None, None, :] == 0, float("-inf"))
        selected = scores.topk(count, dim=-1).indices
        selected.masked_fill_(lengths[sample][selected] == 0, -1)
        route[sample, :, indices, :] = selected
    return route


def make_sparse_mask(route, lengths, valid_tokens, width):
    batch, heads, sequence, topk = route.shape
    frame_count = lengths.shape[1]
    history_tokens = frame_count * width
    meta = torch.tensor(
        [sequence, history_tokens, width, frame_count],
        device=route.device, dtype=torch.long,
    )
    padded_choices = F.pad(route, (0, max(0, 4 - topk)), value=-1)
    choices = tuple(padded_choices[..., rank].contiguous() for rank in range(4))

    def mask_mod(b, h, q, k):
        query_index = torch.minimum(q, meta[0] - 1)
        frame = torch.minimum(k // meta[2], meta[3] - 1)
        offset = k % meta[2]
        current_index = torch.minimum(
            torch.maximum(k - meta[1], torch.zeros_like(k)), meta[0] - 1,
        )
        chosen = (
            (choices[0][b, h, query_index] == frame)
            | (choices[1][b, h, query_index] == frame)
            | (choices[2][b, h, query_index] == frame)
            | (choices[3][b, h, query_index] == frame)
        )
        historical = (k < meta[1]) & chosen & (offset < lengths[b, frame])
        current = (
            (k >= meta[1]) & (k < meta[1] + meta[0])
            & (current_index <= q) & valid_tokens[b, current_index]
        )
        # Padding queries get a harmless diagonal to avoid all-masked rows.
        padding_self = (
            ~valid_tokens[b, query_index] & (k == meta[1] + q)
        )
        return (q < meta[0]) & (historical | current | padding_self)

    query_blocks = math.ceil(sequence / _QUERY_BLOCK)
    padded_route = F.pad(
        route, (0, 0, 0, query_blocks * _QUERY_BLOCK - sequence), value=-1,
    )
    frame_counts = torch.zeros(
        (batch, heads, query_blocks, frame_count),
        device=route.device, dtype=torch.int32,
    )
    block_routes = padded_route.reshape(batch, heads, query_blocks, _QUERY_BLOCK * topk)
    # Inactive and padded queries have -1 routes. They must neither introduce
    # a frame-0 block nor overwrite a real frame-0 selection in a mixed block.
    frame_counts.scatter_add_(
        -1, block_routes.clamp_min(0), (block_routes >= 0).to(torch.int32),
    )
    chosen_frames = frame_counts > 0
    historical_blocks = chosen_frames.repeat_interleave(width // _KEY_BLOCK, dim=-1)
    current_blocks = (
        torch.arange(math.ceil(sequence / _KEY_BLOCK), device=route.device)[None, :] * _KEY_BLOCK
        < (torch.arange(query_blocks, device=route.device)[:, None] + 1) * _QUERY_BLOCK
    )
    allowed = torch.cat([
        historical_blocks,
        current_blocks[None, None].expand(batch, heads, -1, -1),
    ], dim=-1)
    indices = allowed.to(torch.int32).argsort(dim=-1, descending=True, stable=True).to(torch.int32)
    counts = allowed.sum(dim=-1).to(torch.int32)
    block_mask = BlockMask.from_kv_blocks(
        counts, indices, BLOCK_SIZE=(_QUERY_BLOCK, _KEY_BLOCK),
        mask_mod=mask_mod, seq_lengths=(sequence, history_tokens + sequence),
    )
    return block_mask, mask_mod


def sparse_frame_attention(query, current_key, current_value, packed, valid_tokens,
                           *, history_query_mask, topk=4, backend="flex"):
    global _compiled_flex
    history_key, history_value, means, lengths, width = packed
    route = route_frames(query, means, lengths, topk, history_query_mask.bool() & valid_tokens)
    block_mask, mask_mod = make_sparse_mask(route, lengths, valid_tokens, width)
    key = torch.cat([history_key, current_key], dim=-2)
    value = torch.cat([history_value, current_value], dim=-2)
    if backend == "dense_reference":
        batch, heads, sequence, _ = query.shape
        b = torch.arange(batch, device=query.device)[:, None, None, None]
        h = torch.arange(heads, device=query.device)[None, :, None, None]
        q = torch.arange(sequence, device=query.device)[None, None, :, None]
        k = torch.arange(key.shape[-2], device=query.device)[None, None, None, :]
        mask = mask_mod(b, h, q, k)
        output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, enable_gqa=True,
        )
    elif backend == "flex":
        if query.device.type != "cuda":
            raise ValueError("The production MoBA backend requires CUDA")
        if _compiled_flex is None:
            _compiled_flex = torch.compile(flex_attention, fullgraph=True)
        output = _compiled_flex(
            query, key, value, block_mask=block_mask, enable_gqa=True,
            kernel_options=_KERNEL_OPTIONS,
        )
    else:
        raise ValueError(f"Unknown MoBA attention backend: {backend}")
    output = output * valid_tokens[:, None, :, None]
    return output, route


def _moba_forward(self, hidden_states, position_embeddings, attention_mask=None,
                  past_key_values=None, **kwargs):
    owner = self._moba_owner()
    context = None if owner is None else owner._moba_context
    if context is None:
        return self._moba_original_forward(
            hidden_states, position_embeddings=position_embeddings,
            attention_mask=attention_mask, past_key_values=past_key_values, **kwargs,
        )
    # The hybrid cache carries DeltaNet S while its full-attention entries stay empty.
    if past_key_values is not None and past_key_values.get_seq_length(self.layer_idx) != 0:
        raise ValueError("Full-attention HF KV cannot coexist with MoBA frame memory")
    shape = hidden_states.shape[:-1]
    hidden_shape = (*shape, -1, self.head_dim)
    query, gate = torch.chunk(
        self.q_proj(hidden_states).view(*shape, -1, self.head_dim * 2), 2, dim=-1,
    )
    gate = gate.reshape(*shape, -1)
    query = self.q_norm(query.view(hidden_shape)).transpose(1, 2)
    key = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
    context.capture(self.layer_idx, key, value)
    packed = context.pack_history(self.layer_idx, key)
    if packed is None:
        interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward,
        )
        output, _ = interface(
            self, query, key, value, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling, **kwargs,
        )
    else:
        output, route = sparse_frame_attention(
            query, key, value, packed, context.valid_tokens,
            history_query_mask=context.history_query_mask,
            topk=context.owner.topk, backend=owner.moba_backend,
        )
        context.route_indices[self.layer_idx] = route.detach()
        output = output.transpose(1, 2)
    output = output.reshape(*shape, -1).contiguous()
    return self.o_proj(output * torch.sigmoid(gate)), None


def install_moba_attention(policy):
    layers = policy.qwen_vl_interface.model.model.language_model.layers
    for layer_index in policy.moba_memory.layer_indices:
        attention = layers[layer_index].self_attn
        attention._moba_original_forward = attention.forward
        attention._moba_owner = weakref.ref(policy)
        attention.forward = MethodType(_moba_forward, attention)
