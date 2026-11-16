"""Invert the Piper seven-dimensional continuous q01/q99 transform."""

import numpy as np


def unnormalize_actions(normalized_actions, action_norm_stats):
    values = np.asarray(normalized_actions, dtype=np.float32)
    low = np.asarray(action_norm_stats["q01"], dtype=np.float32)
    high = np.asarray(action_norm_stats["q99"], dtype=np.float32)
    mask = np.asarray(action_norm_stats.get("mask", [True] * 7), dtype=bool)
    if values.ndim < 2 or values.shape[-1] != 7 or low.shape != (7,) or high.shape != (7,):
        raise ValueError("Piper actions and normalization statistics must have seven dimensions")
    if mask.shape != (7,) or not mask.all():
        raise ValueError("All seven Piper positions, including the gripper, must be normalized")
    if not np.isfinite(values).all() or not np.isfinite(low).all() or not np.isfinite(high).all():
        raise ValueError("Non-finite Piper actions/statistics")
    if (high < low).any():
        raise ValueError("Invalid normalization interval")
    return 0.5 * (np.clip(values, -1, 1) + 1) * (high - low) + low
