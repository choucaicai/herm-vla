"""Serve episode-scoped zero-state gated memory and detached visual KV."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch

from eventvla.model.framework.share_tools import read_mode_config
from qwen_35_version.real_qwen35_final.model.eventvla_qwen35_mix import (
    RealQwen35Final, action_checkpoint_config, validate_temporal_batch,
)
from qwen_35_version.real_qwen35_final.training.state import (
    detach_slot_state, state_metrics,
)


class RecurrentServingPolicy(RealQwen35Final):
    def __init__(self, config):
        super().__init__(config)
        self.reset_memory()

    @classmethod
    def from_checkpoint(cls, checkpoint):
        checkpoint = Path(checkpoint).resolve()
        config, norm_stats = read_mode_config(checkpoint)
        config = action_checkpoint_config(OmegaConf.create(config))
        if os.environ.get("QWEN35_BASE_MODEL"):
            config.framework.qwenvl.base_vlm = os.environ["QWEN35_BASE_MODEL"]
        deltas = list(config.datasets.vla_data.temporal.image.delta_indices)
        if any(offset > 0 for offset in deltas):
            raise ValueError(
                "This checkpoint was trained with future frames [25, 0]; "
                "changing its config cannot produce historical-frame training"
            )
        config.trainer.enable_gradient_checkpointing = False
        model = cls(config)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        for key in ("module", "state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
        model.load_state_dict(state, strict=True)
        model.norm_stats = norm_stats
        model.unnormalize_actions(np.zeros((1, 7)), model.norm_stats["new_embodiment"]["action"])
        model.eval()
        return model

    def reset_memory(self, batch_size=1):
        if int(batch_size) != 1:
            raise ValueError("Use one serving policy per environment")
        super().reset_memory(batch_size=1)
        self._query_memory_state = None
        self._active_episode = None
        self._last_decision = -1
        self._last_timestep = None
        self._last_fingerprint = None
        self._last_response = None

    @staticmethod
    def _fingerprint(examples):
        digest = hashlib.sha256()
        for example in examples:
            fields = {key: value for key, value in example.items() if key != "image"}
            digest.update(json.dumps(fields, sort_keys=True, default=str).encode())
            for image in example["image"]:
                pixels = np.asarray(image)
                digest.update(str((pixels.shape, pixels.dtype)).encode())
                digest.update(pixels.tobytes())
        return digest.hexdigest()

    @torch.inference_mode()
    def predict_action(self, examples=None, episode_id=None, decision_id=None, **kwargs):
        if examples is None or len(examples) != 1:
            raise ValueError("Serving requires exactly one environment observation")
        validate_temporal_batch(
            examples, data_config=self.config.datasets.vla_data,
        )
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("Provide a nonempty episode_id")
        if not isinstance(decision_id, int) or isinstance(decision_id, bool) or decision_id < 0:
            raise ValueError("Provide a nonnegative integer decision_id")
        if self._active_episode != episode_id:
            if decision_id != 0:
                raise RuntimeError("A new episode must start at decision_id=0")
            self.reset_memory()
            self._active_episode = episode_id
        fingerprint = self._fingerprint(examples)
        if decision_id == self._last_decision:
            if fingerprint != self._last_fingerprint:
                raise ValueError("A repeated decision must contain identical observations")
            return copy.deepcopy(self._last_response)
        if decision_id != self._last_decision + 1:
            raise ValueError("Decision IDs must be consecutive within an episode")
        timestep = int(examples[0]["timestep"])
        if self._last_timestep is not None and timestep <= self._last_timestep:
            raise ValueError("Observation timesteps must increase")
        samples = [dict(examples[0], episode_id=episode_id, slot_id=0)]
        try:
            response, next_state = super().predict_action(
                examples=samples, recurrent_state=self._query_memory_state,
                return_recurrent_state=True, **kwargs,
            )
        except Exception:
            self.discard_moba_memory()
            raise
        actions = response["normalized_actions"]
        if actions.shape != (1, 50, 7) or not np.isfinite(actions).all():
            self.discard_moba_memory()
            raise RuntimeError("Invalid action prediction")
        response["actions"] = self.unnormalize_actions(
            actions, self.norm_stats["new_embodiment"]["action"],
        )
        if next_state.shape != (1, 64, 2560) or not bool(torch.isfinite(next_state).all()):
            self.discard_moba_memory()
            raise RuntimeError("Invalid recurrent memory-token state")
        self.commit_moba_memory()
        response["recurrent_memory"] = {
            "episode_id": episode_id,
            "decision_id": decision_id,
            "state_mode": "memory_queries",
            "warm_start": self._query_memory_state is not None,
            "conv_state_policy": "zero_at_every_forward",
            "initial_full_attention_tokens": 0,
            "initial_state": "zeros",
            "mask_empty_read": True,
            "update_rule": self.query_memory.update_rule,
            "gate_granularity": "slot",
            **self.query_memory.last_stats,
            "gate_by_slot": self.query_memory.last_gate.cpu().tolist(),
            **state_metrics(next_state),
        }
        response["moba_memory"] = {
            "history_query_scope": "all_tokens",
            "cache_delta_indices": (
                None if self.moba_cache_delta_indices is None
                else list(self.moba_cache_delta_indices)
            ),
            "topk_history_images": self.moba_memory.topk,
            "block_unit": self.moba_memory.block_unit,
            **self.moba_memory.last_stats,
            "slots": self.moba_memory.state_summary(),
        }
        self._query_memory_state = detach_slot_state(next_state)
        self._last_decision = decision_id
        self._last_timestep = timestep
        self._last_fingerprint = fingerprint
        self._last_response = copy.deepcopy(response)
        return response
