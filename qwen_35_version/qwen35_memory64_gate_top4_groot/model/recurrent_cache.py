"""Restore only DeltaNet S; convolution and full-attention caches are local."""

from __future__ import annotations

import weakref

import torch
from transformers.cache_utils import DynamicCache


def install_functional_cache_updates(cache):
    # Stock cache copy_ overwrites tensors still needed by TBPTT backward.
    for layer in cache.layers:
        if not hasattr(layer, "update_recurrent_state"):
            continue
        layer_ref = weakref.proxy(layer)

        def update_conv(new_state, _layer=layer_ref, **kwargs):
            if not _layer.is_conv_states_initialized:
                _layer.lazy_initialization(conv_states=new_state)
            kernel = _layer.conv_kernel_size
            if new_state.shape[-1] >= kernel:
                result = new_state[..., -kernel:]
            else:
                result = torch.cat(
                    [_layer.conv_states[..., new_state.shape[-1]:], new_state],
                    dim=-1,
                )
            _layer.conv_states = result
            _layer.is_conv_states_initialized = True
            _layer.has_previous_state = True
            return result

        def update_recurrent(new_state, _layer=layer_ref, **kwargs):
            if not _layer.is_recurrent_states_initialized:
                _layer.lazy_initialization(recurrent_states=new_state)
            _layer.recurrent_states = new_state
            _layer.is_recurrent_states_initialized = True
            _layer.has_previous_state = True
            return new_state

        layer.update_conv_state = update_conv
        layer.update_recurrent_state = update_recurrent


def build_s_only_cache(config, decoder_layers, state=None):
    cache = DynamicCache(config=config)
    text_config = config.get_text_config()
    layer_types = text_config.layer_types
    if state is not None:
        if len(state) != len(layer_types):
            raise ValueError("S must contain one entry per decoder layer")
        for index, (kind, recurrent) in enumerate(zip(layer_types, state)):
            if kind != "linear_attention":
                if recurrent is not None:
                    raise ValueError("Full-attention layers cannot carry state")
                continue
            if recurrent is None:
                raise ValueError("A warm state must provide S for every linear layer")
            if not isinstance(recurrent, torch.Tensor) or recurrent.ndim != 4:
                raise TypeError("Each linear layer carries a tensor S, never a (C, S) pair")
            expected = (
                text_config.linear_num_value_heads,
                text_config.linear_key_head_dim,
                text_config.linear_value_head_dim,
            )
            if tuple(recurrent.shape[1:]) != expected:
                raise ValueError(f"Invalid S shape {tuple(recurrent.shape)}; expected B,{expected}")
            attention = decoder_layers[index].linear_attn
            conv = torch.zeros(
                recurrent.shape[0],
                attention.conv_dim,
                attention.conv_kernel_size,
                device=recurrent.device,
                dtype=attention.conv1d.weight.dtype,
            )
            layer = cache.layers[index]
            layer.lazy_initialization(conv_states=conv, recurrent_states=recurrent)
            layer.conv_states = conv
            layer.recurrent_states = recurrent.clone()
            layer.is_conv_states_initialized = True
            layer.is_recurrent_states_initialized = True
            layer.has_previous_state = True
    install_functional_cache_updates(cache)
    if cache.get_seq_length() != 0:
        raise RuntimeError("Full-attention KV must start empty on each decision")
    return cache


def extract_s_only_state(cache, layer_types):
    if cache is None or len(cache.layers) != len(layer_types):
        raise RuntimeError("Qwen3.5 did not return the expected decoder cache")
    state = []
    for kind, layer in zip(layer_types, cache.layers):
        if kind == "linear_attention":
            if not layer.is_recurrent_states_initialized:
                raise RuntimeError("A linear-attention layer did not return S")
            state.append(layer.recurrent_states)
        else:
            state.append(None)
    return state
