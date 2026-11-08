"""RMBench adapter for exact frame history and zero-state gated memory."""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
import uuid

from examples.RMBench.eval_files.model2robotwin_mem_interface import (
    ModelClient as EventVLAClient,
    eval,
    reset_model,
)
from ..data.frame_tokens import normalize_cache_delta_indices


class ModelClient(EventVLAClient):
    def __init__(self, policy_ckpt_path, **kwargs):
        self.checkpoint = str(Path(policy_ckpt_path).resolve())
        self._episode_id = uuid.uuid4().hex
        self._decision_id = 0
        absolute = kwargs.pop("temporal_absolute_indices", None)
        deltas = kwargs.pop("temporal_anchor_deltas", None)
        kwargs.setdefault("first_chunk_random_replan", False)
        # The shared client validates its original four-frame profile at init.
        # Restore the checkpoint's own profile before any observation.
        super().__init__(
            policy_ckpt_path=policy_ckpt_path,
            temporal_absolute_indices=(0,),
            temporal_anchor_deltas=(-30, -15, 0),
            **kwargs,
        )
        framework = self.model_config["framework"]
        image_config = self.vla_data_config["temporal"]["image"]
        trained_absolute = tuple(image_config.get("absolute_indices", []))
        trained_deltas = tuple(image_config["delta_indices"])
        if (trained_absolute, trained_deltas) not in {
            ((0,), (-15, 0)), ((), (-25, 0)), ((), (-30, -15, 0)),
        }:
            raise ValueError(
                "Online evaluation requires a checkpoint trained with historical frames; "
                f"checkpoint profile is absolute={trained_absolute}, delta={trained_deltas}"
            )
        if absolute is not None and tuple(absolute) != trained_absolute:
            raise ValueError("The absolute-frame override must match the checkpoint")
        if deltas is not None and tuple(deltas) != trained_deltas:
            raise ValueError("The relative-frame override must match the checkpoint")
        if (
            framework["name"] != "RawQwen35MixCurrentKVActionMemory64AllTokensZeroStateGateImageTopK"
            or framework.get("recurrent_state_mode") != "memory_queries"
            or framework.get("action_token") != "\u25c6"
            or framework.get("memory_queries", {}).get("initial_state") != "zeros"
            or framework.get("memory_queries", {}).get("mask_empty_read") is not True
            or framework.get("memory_queries", {}).get("force_first_write") is not True
            or framework.get("memory_queries", {}).get("gate_granularity") != "slot"
            or framework.get("memory_queries", {}).get("update_rule") not in {"gated", "replace"}
            or bool(image_config["include_initial"]) != bool(trained_absolute)
            or self.memory_ablation_mode != "raw_anchors_only"
            or self.action_chunk_size != 50
            or self.sampling_interval != 50
        ):
            raise ValueError("Expected a zero-state memory checkpoint with historical images and stride 50")
        self.temporal_absolute_indices = trained_absolute
        self.temporal_anchor_deltas = trained_deltas
        self.moba_cache_delta_indices = normalize_cache_delta_indices(
            framework["moba_memory"].get("cache_delta_indices", [0]), trained_deltas
        )
        self.temporal_history_capacity = max(
            int(self.horizon or 0), abs(min((0, *trained_deltas))) + 1,
            max((0, *trained_absolute)) + 1,
        )
        self.image_history = deque(maxlen=self.temporal_history_capacity)
        self.replan_horizon = 50

    def _connect_client(self):
        client = super()._connect_client()
        metadata = client.get_server_metadata()
        topk = int(self.model_config["framework"]["moba_memory"]["topk_history_images"])
        expected = {
            "framework": "RawQwen35MixCurrentKVActionMemory64AllTokensZeroStateGateImageTopK",
            "model_class": "RecurrentServingPolicy",
            "input_mode": (
                f"temporal_{3 * (len(self.temporal_absolute_indices) + len(self.temporal_anchor_deltas))}_images"
            ),
            "memory": "memory_queries",
            "memory_query_count": 64,
            "memory_initial_state": "zeros",
            "memory_mask_empty_read": True,
            "memory_update_rule": self.model_config["framework"]["memory_queries"]["update_rule"],
            "memory_gate_granularity": "slot",
            "memory_force_first_write": True,
            "native_s_policy": "reset_every_decision",
            "conv_state_policy": "zero_at_every_forward",
            "full_attention_kv_policy": f"detached_current_visual_images_all_tokens_top{topk}",
            "moba_block_unit": "image",
            "moba_topk_history_images": topk,
            "moba_cache_delta_indices": (
                None if self.moba_cache_delta_indices is None
                else list(self.moba_cache_delta_indices)
            ),
            "num_views": 3,
            "delta_indices": list(self.temporal_anchor_deltas),
            "absolute_indices": list(self.temporal_absolute_indices),
            "sampling_interval": 50,
            "action_horizon": 50,
            "action_dim": 14,
            "action_token": "\u25c6",
            "action_token_id": 158778,
            "strict_checkpoint_load": True,
        }
        if (
            any(metadata.get(key) != value for key, value in expected.items())
            or Path(metadata.get("checkpoint", "")).resolve() != Path(self.checkpoint)
        ):
            client.close()
            self.client = None
            raise ValueError(f"Unexpected recurrent model service metadata: {metadata}")
        return client

    def reset(self, task_description):
        self._episode_id = uuid.uuid4().hex
        self._decision_id = 0
        super().reset(task_description)

    def _call_server_with_reconnect(self, method_name, *args):
        if method_name != "predict_action":
            response = super()._call_server_with_reconnect(method_name, *args)
            self._raise_for_server_error(response)
            return response
        query = dict(args[0])
        query["episode_id"] = self._episode_id
        query["decision_id"] = self._decision_id
        # The parent retries the same query, so a lost response cannot advance S twice.
        response = super()._call_server_with_reconnect(method_name, query)
        self._raise_for_server_error(response)
        memory = response.get("data", {}).get("recurrent_memory", {})
        if (
            memory.get("episode_id") != self._episode_id
            or memory.get("decision_id") != self._decision_id
            or memory.get("state_mode") != "memory_queries"
            or memory.get("warm_start") != (self._decision_id > 0)
            or memory.get("conv_state_policy") != "zero_at_every_forward"
            or memory.get("initial_full_attention_tokens") != 0
        ):
            raise RuntimeError(f"Unexpected recurrent state acknowledgement: {memory}")
        moba = response.get("data", {}).get("moba_memory", {})
        expected_topk = int(self.model_config["framework"]["moba_memory"]["topk_history_images"])
        if (moba.get("block_unit") != "image"
                or moba.get("topk_history_images") != expected_topk
                or not 0 <= float(moba.get("moba_topk_effective", -1)) <= expected_topk):
            raise RuntimeError(f"Unexpected image KV acknowledgement: {moba}")
        logging.info(
            "Recurrent decision: episode=%s decision=%d timestep=%d warm_start=%s mode=memory_queries",
            self._episode_id,
            self._decision_id,
            query["examples"][0]["timestep"],
            memory["warm_start"],
        )
        self._decision_id += 1
        return response

    def _record_temporal_images(self, images, step):
        if len(images) != 3:
            raise ValueError("Each environment frame must contain three camera views")
        if self.image_history and int(step) != int(self.image_history[-1]["step"]) + 1:
            raise ValueError("Collect every environment frame to preserve exact relative history")
        super()._record_temporal_images(images, step)

    def _build_temporal_anchor_images(self, step):
        images, metas = super()._build_temporal_anchor_images(step)
        for meta in metas:
            if meta["delta"] is not None:
                meta["frame_timestep"] = max(0, int(step) + int(meta["delta"]))
            else:
                meta["frame_timestep"] = int(self.initial_step or 0) + int(meta["absolute_index"])
        return images, metas

    def needs_observation(self, step):
        return True


def get_model(usr_args):
    checkpoint = usr_args.get("policy_ckpt_path")
    if checkpoint is None:
        raise ValueError("policy_ckpt_path must be provided")
    return ModelClient(
        policy_ckpt_path=checkpoint,
        host=usr_args.get("host", "127.0.0.1"),
        port=usr_args.get("port", 7402),
        unnorm_key=usr_args.get("unnorm_key"),
        action_mode=usr_args.get("action_mode", "abs"),
        first_chunk_random_replan=usr_args.get("first_chunk_random_replan", False),
        first_chunk_random_replan_seed=usr_args.get("first_chunk_random_replan_seed", 42),
        first_chunk_fixed_replan_step=usr_args.get("first_chunk_fixed_replan_step"),
        sampling_interval=usr_args.get("sampling_interval"),
        replan_after_keyframe_commit=False,
        image_size=usr_args.get("image_size"),
        temporal_absolute_indices=usr_args.get("temporal_absolute_indices"),
        temporal_anchor_deltas=usr_args.get("temporal_anchor_deltas"),
        temporal_view_names=usr_args.get("temporal_view_names"),
    )
