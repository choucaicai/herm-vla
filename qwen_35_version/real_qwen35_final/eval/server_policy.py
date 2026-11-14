"""Serve zero-state gated memory and detached visual KV over WebSocket."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from accelerate import PartialState
import torch

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from qwen_35_version.real_qwen35_final.eval.policy import RecurrentServingPolicy
from qwen_35_version.real_qwen35_final.model.frame_memory import retrieval_settings


def load_policy(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    model = RecurrentServingPolicy.from_checkpoint(checkpoint)
    data = model.config.datasets.vla_data
    image = data.temporal.image
    absolute = list(image.get("absolute_indices", []))
    deltas = list(image.delta_indices)
    image_count = int(data.num_camera_views) * (len(absolute) + len(deltas))
    block_unit, topk = retrieval_settings(model.config.framework.moba_memory)
    metadata = {
        "framework": "RealQwen35Final",
        "model_class": "RecurrentServingPolicy",
        "input_mode": f"temporal_{image_count}_images",
        "memory": "memory_queries",
        "memory_query_count": 64,
        "memory_initial_state": "zeros",
        "memory_mask_empty_read": True,
        "memory_update_rule": model.query_memory.update_rule,
        "memory_gate_granularity": "slot",
        "memory_force_first_write": True,
        "native_s_policy": "reset_every_decision",
        "conv_state_policy": "zero_at_every_forward",
        "full_attention_kv_policy": f"detached_current_visual_images_all_tokens_top{topk}",
        "moba_block_unit": block_unit,
        "moba_topk_history_images": topk,
        "moba_history_query_scope": "all_tokens",
        "moba_cache_delta_indices": (
            None if model.moba_cache_delta_indices is None
            else list(model.moba_cache_delta_indices)
        ),
        "checkpoint": str(checkpoint),
        "num_views": int(data.num_camera_views),
        "delta_indices": deltas,
        "absolute_indices": absolute,
        "sampling_interval": int(data.sampling_interval),
        "action_horizon": int(model.chunk_len),
        "action_dim": int(model.config.framework.action_model.action_dim),
        "action_names": [f"joint_{i}.pos" for i in range(1, 7)] + ["gripper.pos"],
        "action_mode": "abs",
        "action_units": "same as the recorded Piper dataset; no unit conversion",
        "gripper_mode": "continuous",
        "camera_order": ["cam_high", "cam_follower_wrist"],
        "fps": 25,
        "actions_field": "denormalized absolute positions [1, 50, 7]",
        "action_token": model.action_token,
        "action_token_id": int(model.action_token_id),
        "strict_checkpoint_load": True,
    }
    logging.info("Recurrent checkpoint loaded strictly: %s; metadata=%s", checkpoint, metadata)
    return model, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--port", type=int, default=7402)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--idle_timeout", type=int, default=1800)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    PartialState()
    model, metadata = load_policy(args.ckpt_path)
    if args.use_bf16:
        model = model.to(torch.bfloat16)
    model = model.to("cuda").eval()
    WebsocketPolicyServer(
        model,
        host="127.0.0.1",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    main()
