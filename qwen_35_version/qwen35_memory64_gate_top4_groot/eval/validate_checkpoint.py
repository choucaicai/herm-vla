"""Validate the recurrent recipe and checkpoint layout before launching GPUs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf
import torch

from eventvla.model.tools import read_mode_config
from qwen_35_version.qwen35_memory64_gate_top4_groot.model.eventvla_qwen35_mix import (
    action_checkpoint_config, validate_config,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    config, stats = read_mode_config(checkpoint)
    inference_config = action_checkpoint_config(OmegaConf.create(config))
    validate_config(inference_config)
    data = config["datasets"]["vla_data"]
    image = data["temporal"]["image"]
    absolute = list(image.get("absolute_indices", []))
    deltas = list(image["delta_indices"])
    if any(offset > 0 for offset in deltas):
        raise ValueError(
            "Checkpoint was trained with future frames [25, 0]; "
            "online historical-frame evaluation requires a corrected checkpoint"
        )
    if "new_embodiment" not in stats:
        raise ValueError("Missing new_embodiment normalization statistics")
    state = torch.load(checkpoint, map_location="meta", weights_only=True, mmap=True)
    for key in ("module", "state_dict"):
        if key in state and isinstance(state[key], dict):
            state = state[key]
            break
    if not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("Expected a tensor state dictionary")
    if not any(key.startswith("qwen_vl_interface.") for key in state):
        raise ValueError("Missing Qwen backbone parameters")
    if not any(key.startswith("action_model.") for key in state):
        raise ValueError("Missing action head parameters")
    required = {
        "query_memory.write_queries": (64, 2560),
        "query_memory.write_gate.weight": (1, 5120),
        "query_memory.write_gate.bias": (1,),
        "query_memory.feedback_projection.weight": (2560, 2560),
    }
    action_dim = int(inference_config.framework.action_model.action_dim)
    required.update({
        "action_model.action_encoder.layer1.weight": (768, action_dim),
        "action_model.action_decoder.layer2.weight": (action_dim, 1024),
        "action_model.future_tokens.weight": (32, 768),
    })
    for index in range(16):
        prefix = f"action_model.model.transformer_blocks.{index}.attn1"
        required[f"{prefix}.to_q.weight"] = (768, 768)
        required[f"{prefix}.to_k.weight"] = (768, 768 if index % 2 else 2560)
    if "action_model.model.fc2.weight" in state:
        raise ValueError("An old MLP checkpoint cannot be served by the GR00T expert")
    if "query_memory.initial_memory" in state:
        raise ValueError("This version cannot contain a learned initial memory")
    for name, shape in required.items():
        if name not in state or tuple(state[name].shape) != shape:
            raise ValueError(f"Missing or incompatible zero-state gate parameter: {name}")
    print(json.dumps({
        "checkpoint": str(checkpoint),
        "framework": config["framework"]["name"],
        "inference_framework": inference_config.framework.name,
        "history_query_scope": "all_tokens",
        "recurrent_state_mode": "memory_queries",
        "memory_initial_state": "zeros",
        "memory_mask_empty_read": True,
        "memory_update_rule": inference_config.framework.memory_queries.update_rule,
        "memory_gate_granularity": "slot",
        "absolute_indices": absolute,
        "delta_indices": deltas,
        "include_initial": bool(image["include_initial"]),
        "images_per_sample": int(data["num_camera_views"]) * (len(absolute) + len(deltas)),
        "sampling_interval": int(data["sampling_interval"]),
        "tensor_count": len(state),
        "parameter_elements": sum(value.numel() for value in state.values()),
        "strict_load": "enforced by server",
        "passed": True,
    }), flush=True)


if __name__ == "__main__":
    main()
