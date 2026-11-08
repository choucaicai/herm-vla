"""Fixed-zero episode memory with masked empty reads and per-slot write gates."""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class MemoryLayout:
    original_indices: torch.Tensor
    read_mask: torch.Tensor
    write_mask: torch.Tensor
    action_mask: torch.Tensor
    write_indices: torch.Tensor
    spans: list
    valid_mask: torch.Tensor
    previous_memory: torch.Tensor
    has_memory: torch.Tensor

    @property
    def history_query_mask(self):
        # Memory placeholders have pad-token IDs but are valid sequence tokens.
        return self.valid_mask

    def remap_frame_specs(self, specs):
        result = []
        for groups, (start, action_start, count) in zip(specs, self.spans):
            current = []
            for group in groups:
                indices = group["token_indices"]
                if any(index < start or index >= action_start for index in indices):
                    raise ValueError("Visual tokens must precede the action-query block")
                current.append(dict(group, token_indices=[index + count for index in indices]))
            result.append(current)
        if len(result) != len(self.spans):
            raise ValueError("Frame specs do not match the memory batch")
        return result


class IdentityFeedback(nn.Linear):
    def __init__(self, hidden_size, scale):
        self.initial_scale = float(scale)
        super().__init__(hidden_size, hidden_size, bias=False)

    def reset_parameters(self):
        nn.init.eye_(self.weight)
        with torch.no_grad():
            self.weight.mul_(self.initial_scale)


class QueryMemory(nn.Module):
    def __init__(self, hidden_size, count=64, initializer_range=0.02,
                 feedback_projection_init_scale=0.02, *, update_rule="gated",
                 gate_initial_write_ratio=0.2):
        super().__init__()
        self.hidden_size, self.count = int(hidden_size), int(count)
        if self.count != 64:
            raise ValueError("This experiment requires exactly 64 memory queries")
        if update_rule not in {"gated", "replace"}:
            raise ValueError("Memory update_rule must be gated or replace")
        if not 0 < float(gate_initial_write_ratio) < 1:
            raise ValueError("gate_initial_write_ratio must be strictly between zero and one")
        self.update_rule = update_rule
        self.write_queries = nn.Parameter(torch.empty(self.count, self.hidden_size))
        nn.init.normal_(self.write_queries, std=float(initializer_range))
        self.feedback_norm = nn.RMSNorm(self.hidden_size, eps=1e-6)
        self.feedback_projection = IdentityFeedback(self.hidden_size, feedback_projection_init_scale)
        self.write_gate = nn.Linear(2 * self.hidden_size, 1)
        nn.init.zeros_(self.write_gate.weight)
        ratio = float(gate_initial_write_ratio)
        nn.init.constant_(self.write_gate.bias, math.log(ratio / (1 - ratio)))
        self.last_gate = None
        self.last_stats = {}

    def read(self, state, reference):
        batch = reference.shape[0]
        # A runtime tensor, never a learned parameter or checkpoint buffer.
        initial = reference.new_zeros((batch, self.count, self.hidden_size))
        if state is None:
            return initial, torch.zeros(batch, dtype=torch.bool, device=reference.device)
        if isinstance(state, dict):
            if set(state) != {"tokens", "valid"}:
                raise ValueError("Merged query memory requires tokens and a valid-slot mask")
            tokens, valid = state["tokens"], state["valid"]
            if valid.shape != (batch,) or valid.dtype != torch.bool:
                raise ValueError("Invalid query-memory slot mask")
        else:
            tokens, valid = state, torch.ones(batch, device=reference.device, dtype=torch.bool)
        if not isinstance(tokens, torch.Tensor) or tokens.shape != initial.shape or not tokens.is_floating_point():
            raise ValueError(f"Query memory must have shape {tuple(initial.shape)}")
        valid = valid.to(reference.device)
        return torch.where(valid[:, None, None], tokens.to(reference), initial), valid

    def expand(self, inputs, state, *, action_token_id, pad_token_id, action_query_count=50):
        ids, valid = inputs["input_ids"], inputs["attention_mask"].bool()
        embeddings, positions = inputs["inputs_embeds"], inputs["position_ids"]
        if embeddings.shape[:2] != ids.shape or positions.shape != (3, *ids.shape):
            raise ValueError("Prepared embeddings must retain the native three-axis M-RoPE")
        memory, has_memory = self.read(state, embeddings)
        write = self.write_queries.to(embeddings)
        # Resolve spans once per decision, outside the attention-layer loop.
        host_ids, host_valid = ids.detach().cpu(), valid.detach().cpu()
        rows, embed_rows, mask_rows, type_rows, position_rows = [], [], [], [], []
        maps, read_masks, write_masks, action_masks, write_indices, spans = [], [], [], [], [], []
        count, length = self.count, ids.shape[1]
        token_types = inputs.get("mm_token_type_ids", torch.zeros_like(ids))
        for batch in range(ids.shape[0]):
            active = host_valid[batch].nonzero().flatten()
            if not active.numel():
                raise ValueError("A memory decision must contain a nonempty observation")
            start, stop = int(active[0]), int(active[-1]) + 1
            if not bool(host_valid[batch, start:stop].all()):
                raise ValueError("Only contiguous left/right prompt padding is supported")
            action = ((host_ids[batch] == action_token_id) & host_valid[batch]).nonzero().flatten()
            if action.numel() != action_query_count:
                raise ValueError("Native action queries must remain complete")
            action_start = int(action[0])
            if not torch.equal(action, torch.arange(action_start, action_start + action_query_count)):
                raise ValueError("Native action queries must remain adjacent")
            spans.append((start, action_start, count))
            filler = ids.new_full((count,), int(pad_token_id))
            rows.append(torch.cat([ids[batch, :start], filler, ids[batch, start:action_start], filler, ids[batch, action_start:]]))
            embed_rows.append(torch.cat([embeddings[batch, :start], memory[batch], embeddings[batch, start:action_start], write, embeddings[batch, action_start:]]))
            ones, zeros = valid.new_ones(count), token_types.new_zeros(count)
            read_valid = has_memory[batch].expand(count)
            mask_rows.append(torch.cat([valid[batch, :start], read_valid, valid[batch, start:action_start], ones, valid[batch, action_start:]]))
            type_rows.append(torch.cat([token_types[batch, :start], zeros, token_types[batch, start:action_start], zeros, token_types[batch, action_start:]]))
            native = positions[:, batch]
            read_positions = torch.arange(count, device=positions.device, dtype=positions.dtype)[None].expand(3, -1)
            write_positions = native[:, action_start:action_start + 1] + count + read_positions
            position_rows.append(torch.cat([native[:, :start], read_positions, native[:, start:action_start] + count, write_positions, native[:, action_start:] + 2 * count], dim=-1))
            original = torch.arange(length, device=ids.device)
            maps.append(original + (original >= start) * count + (original >= action_start) * count)
            read_mask = valid.new_zeros(length + 2 * count)
            write_mask, action_mask = read_mask.clone(), read_mask.clone()
            read_mask[start:start + count] = True
            write_start = action_start + count
            write_mask[write_start:write_start + count] = True
            action_mask[action_start + 2 * count:action_start + 2 * count + action_query_count] = True
            read_masks.append(read_mask)
            write_masks.append(write_mask)
            action_masks.append(action_mask)
            write_indices.append(torch.arange(write_start, write_start + count, device=ids.device))
        expanded = dict(inputs)
        expanded.update(input_ids=torch.stack(rows), inputs_embeds=torch.stack(embed_rows), attention_mask=torch.stack(mask_rows).to(inputs["attention_mask"].dtype), mm_token_type_ids=torch.stack(type_rows), position_ids=torch.stack(position_rows, dim=1))
        expanded.pop("cache_position", None)
        expanded.pop("rope_deltas", None)
        return expanded, MemoryLayout(
            torch.stack(maps), torch.stack(read_masks), torch.stack(write_masks),
            torch.stack(action_masks), torch.stack(write_indices), spans,
            expanded["attention_mask"].bool(),
            memory, has_memory,
        )

    def write(self, hidden, layout, *, force_gate=None):
        """Update runtime memory; force_gate is a diagnostic override for warm slots."""
        if force_gate is not None and not 0 <= float(force_gate) <= 1:
            raise ValueError("force_gate must be between zero and one")
        writes = hidden.gather(1, layout.write_indices[..., None].expand(-1, -1, hidden.shape[-1]))
        candidate = self.feedback_projection(self.feedback_norm(writes))
        # Normalize only gate inputs. Preserve the old state unchanged on the
        # retention branch, and perform sigmoid/interpolation in float32.
        with torch.autocast(hidden.device.type, enabled=False):
            previous = layout.previous_memory.float()
            features = torch.cat([
                F.rms_norm(previous, (self.hidden_size,), eps=1e-6),
                F.rms_norm(writes.float(), (self.hidden_size,), eps=1e-6),
            ], dim=-1)
            proposed = torch.sigmoid(F.linear(
                features, self.write_gate.weight.float(), self.write_gate.bias.float(),
            ))
            if force_gate is not None:
                proposed = proposed * 0 + float(force_gate)
            elif self.update_rule == "replace":
                proposed = proposed * 0 + 1
            gate = torch.where(layout.has_memory[:, None, None], proposed, torch.ones_like(proposed))
            updated = (1 - gate) * previous + gate * candidate.float()
        self._record_update(gate, previous, updated, layout.has_memory)
        return updated.to(candidate.dtype)

    def _record_update(self, gate, previous, updated, has_memory):
        with torch.no_grad():
            gate = gate.detach()
            warm = has_memory[:, None, None].expand_as(gate)
            warm_count = warm.sum()
            values = torch.stack([
                gate.mean(), gate.min(), gate.max(),
                (gate * warm).sum() / warm_count.clamp_min(1),
                (gate < 0.05).float().mean(), (gate > 0.95).float().mean(),
                (updated.detach() - previous.detach()).square().mean().sqrt(),
                updated.detach().square().mean().sqrt(),
                (~has_memory).sum().float(), has_memory.sum().float(),
            ]).cpu().tolist()
            names = [
                "memory_gate_mean", "memory_gate_min", "memory_gate_max",
                "memory_gate_warm_mean", "memory_gate_near_zero_fraction",
                "memory_gate_near_one_fraction", "memory_update_rms", "memory_state_rms",
                "memory_cold_slots", "memory_warm_slots",
            ]
            self.last_gate = gate.squeeze(-1)
            self.last_stats = dict(zip(names, values))
