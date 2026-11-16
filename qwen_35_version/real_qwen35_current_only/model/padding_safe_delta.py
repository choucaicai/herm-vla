"""Make left padding an identity update of a carried DeltaNet state."""

from __future__ import annotations

from types import MethodType

import torch


def _keep_padding_mask(self, attention_mask, past_key_values):
    return attention_mask


def _padding_safe_forward(native_forward):
    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        if attention_mask is None:
            return native_forward(
                self, hidden_states, cache_params=cache_params, attention_mask=None, **kwargs
            )
        padding = ~attention_mask.to(dtype=torch.bool).unsqueeze(-1)

        def wrap(kernel):
            def masked_delta(query, key, value, *, g, beta, **kernel_kwargs):
                # Keep native positive beta. Zero Q/K/V and g make padding an
                # identity update without introducing a zero-beta kernel case.
                vectors = padding.unsqueeze(-1)
                return kernel(
                    query.masked_fill(vectors, 0),
                    key.masked_fill(vectors, 0),
                    value.masked_fill(vectors, 0),
                    g=g.masked_fill(padding, 0),
                    beta=beta,
                    **kernel_kwargs,
                )
            return masked_delta

        chunk = self.chunk_gated_delta_rule
        recurrent = self.recurrent_gated_delta_rule
        self.chunk_gated_delta_rule = wrap(chunk)
        self.recurrent_gated_delta_rule = wrap(recurrent)
        try:
            return native_forward(
                self, hidden_states, cache_params=cache_params,
                attention_mask=attention_mask, **kwargs,
            )
        finally:
            self.chunk_gated_delta_rule = chunk
            self.recurrent_gated_delta_rule = recurrent
    return forward


def install_padding_safe_deltanet(language_model):
    if getattr(language_model, "_s_only_padding_safe", False):
        return
    language_model._update_linear_attn_mask = MethodType(_keep_padding_mask, language_model)
    for layer in language_model.layers:
        attention = getattr(layer, "linear_attn", None)
        if attention is not None:
            attention.forward = MethodType(
                _padding_safe_forward(type(attention).forward), attention
            )
    language_model._s_only_padding_safe = True
