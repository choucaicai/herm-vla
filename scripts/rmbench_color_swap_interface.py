"""Wrap any RMBench policy with an explicitly selected legacy-color adapter."""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import logging
from types import ModuleType
from typing import Any

import numpy as np

CAMERAS = ("head_camera", "left_camera", "right_camera")


@dataclass
class ColorSwapModel:
    policy: ModuleType
    model: Any
    swap_rb: bool

    def __getattr__(self, name):
        return getattr(self.model, name)


def get_model(usr_args):
    base_name = usr_args.get("base_policy_name")
    if not base_name or base_name == __name__:
        raise ValueError("A distinct base_policy_name is required for the color adapter")
    enabled = usr_args.get("swap_rb", False)
    if not isinstance(enabled, bool):
        raise ValueError("swap_rb must be a boolean")
    policy = importlib.import_module(base_name)
    args = dict(usr_args)
    args.pop("base_policy_name", None)
    args.pop("swap_rb", None)
    args["policy_name"] = base_name
    logging.getLogger(__name__).info("Camera R/B swap=%s; base policy=%s", enabled, base_name)
    print(f"[camera-color] swap_rb={enabled}; base_policy={base_name}; cameras={','.join(CAMERAS)}", flush=True)
    return ColorSwapModel(policy, policy.get_model(args), enabled)


def swap_camera_rb(observation):
    """Copy camera leaves so the simulator's own RGB observations remain intact."""
    updated = dict(observation)
    updated["observation"] = dict(observation["observation"])
    for camera in CAMERAS:
        leaf = dict(observation["observation"][camera])
        rgb = np.asarray(leaf["rgb"])
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"{camera}: expected HxWx3 RGB, got {rgb.shape}")
        leaf["rgb"] = rgb[..., ::-1].copy()
        updated["observation"][camera] = leaf
    return updated


def eval(TASK_ENV, model, observation):
    # This runs before the underlying adapter resizes/stores the initial,
    # historical and current images and before the model updates KV/S.
    if model.swap_rb:
        observation = swap_camera_rb(observation)
    return model.policy.eval(TASK_ENV, model.model, observation)


def reset_model(model):
    return model.policy.reset_model(model.model)
