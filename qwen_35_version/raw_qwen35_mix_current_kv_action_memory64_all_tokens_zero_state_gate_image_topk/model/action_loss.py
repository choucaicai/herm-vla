"""Action supervision excluding padded episode-tail targets."""

from __future__ import annotations

import numpy as np
import torch


def masked_action_l1(prediction, target, mask=None):
    error = (prediction.float() - target.float()).abs()
    if mask is None:
        return error.mean()
    mask = torch.as_tensor(mask, device=error.device, dtype=error.dtype)
    if mask.ndim == 2:
        mask = mask.unsqueeze(-1)
    if mask.shape[:2] != error.shape[:2] or mask.shape[-1] not in (1, error.shape[-1]):
        raise ValueError(f"Invalid action mask {tuple(mask.shape)} for {tuple(error.shape)}")
    mask = mask.expand_as(error)
    return (error * mask).sum() / mask.sum().clamp_min(1)


def action_mask_from_examples(examples, horizon):
    masks = []
    for example in examples:
        value = example.get("action_valid_mask")
        if value is None:
            value = np.ones((horizon, 1), dtype=np.float32)
        value = np.asarray(value, dtype=np.float32)
        if value.ndim == 1:
            value = value[:, None]
        if value.shape[0] < horizon:
            raise ValueError("The action mask is shorter than the prediction horizon")
        if not bool(example.get("step_valid", True)) or bool(example.get("dummy", False)):
            value = np.zeros_like(value)
        masks.append(value[-horizon:])
    return np.stack(masks)
