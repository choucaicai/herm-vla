"""EventVLA with 64 recurrent memory queries and detached historical visual KV."""

from __future__ import annotations

import copy

import numpy as np
import torch

from deployment.model_server.tools.image_tools import to_pil_preserve
from eventvla.model.framework.EventVLA import EventVLA
from eventvla.model.modules.action_model.MLP_ActionHeader import L1RegressionActionHead
from eventvla.model.tools import FRAMEWORK_REGISTRY
from eventvla.training.trainer_utils.trainer_tools import resize_images

from .action_loss import action_mask_from_examples, masked_action_l1
from .normalization import unnormalize_actions
from .padding_safe_delta import install_padding_safe_deltanet
from .recurrent_cache import build_s_only_cache
from .query_memory import QueryMemory
from .frame_memory import MemoryContext, retrieval_settings
from .moba_attention import install_moba_attention
from .slot_memory import SlotFrameKVMemory
from .window_vision import prepare_window_vision_inputs
from ..data.frame_tokens import (
    build_frame_specs, normalize_cache_delta_indices, select_cache_frame_specs,
)
from ..data.temporal_profile import temporal_indices

ACTION_TOKEN = "\u25c6"
FRAMEWORK_NAME = "RealQwen35Final"


def action_checkpoint_config(config):
    """Accept only checkpoints trained with the query-memory architecture."""
    config = copy.deepcopy(config)
    if config.framework.name != FRAMEWORK_NAME:
        raise ValueError("Query-memory serving requires its own trained checkpoint")
    return config


def validate_config(config):
    if config.framework.name != FRAMEWORK_NAME:
        raise ValueError(f"This package requires framework.name={FRAMEWORK_NAME}")
    if config.framework.memory_ablation_mode != "raw_anchors_only":
        raise ValueError("This version requires raw_anchors_only image injection")
    if config.framework.get("action_token", "") != ACTION_TOKEN:
        raise ValueError("The raw Qwen3.5 diamond action marker must be retained")
    action = config.framework.action_model
    if (action.action_dim, action.action_horizon, action.future_action_window_size,
            action.past_action_window_size) != (7, 50, 49, 0):
        raise ValueError("The EventVLA action layout must be 50 x 7 for Piper")
    if action.action_model_type != "MLP":
        raise ValueError("The original EventVLA MLP action head is required")
    data = config.datasets.vla_data
    if data.get("action_mode") != "abs" or data.get("lerobot_version") != "v3.0":
        raise ValueError("Piper requires absolute actions and LeRobot v3.0")
    if data.get("include_state", False):
        raise ValueError("The final architecture consumes images and language; include_state must be false")
    absolute, delta = temporal_indices(config)
    if bool(data.temporal.image.include_initial) != bool(absolute):
        raise ValueError("include_initial must match the configured absolute frames")
    if bool(data.temporal.image.include_current) != (0 in delta):
        raise ValueError("include_current must match the configured relative frames")
    if int(data.num_camera_views) != 2 or int(data.sampling_interval) != 50:
        raise ValueError("This recipe requires two views and a 50-frame decision stride")
    if not data.get("processor_in_dataloader_workers", False) or data.num_workers < 1:
        raise ValueError("Qwen chat templates must be applied inside DataLoader workers")
    if not data.get("pack_tbptt_windows", False) or int(config.trainer.tbptt_steps) != 2:
        raise ValueError("This recipe requires packed two-step TBPTT windows")
    if int(config.trainer.gradient_accumulation_steps) != 1:
        raise ValueError("Each two-step window is one optimizer update")
    if bool(config.trainer.enable_gradient_checkpointing):
        raise ValueError("Gradient checkpointing must be disabled for stateful TBPTT")
    memory = config.framework.memory_buffer
    if bool(memory.enable) or bool(data.keyframe_image_memory.enabled):
        raise ValueError("External keyframe memory must be disabled")
    if str(memory.use_keyframe_predict_head) != "disabled" or float(memory.keyframe_loss_weight) != 0:
        raise ValueError("This action-only recipe disables keyframe auxiliary supervision")
    if config.framework.get("recurrent_state_mode") != "memory_queries":
        raise ValueError("Cross-decision state must be memory_queries")
    memory_queries = config.framework.get("memory_queries", {})
    if int(memory_queries.get("count", 0)) != 64:
        raise ValueError("This experiment requires exactly 64 memory queries")
    if memory_queries.get("native_s_policy") != "reset_every_decision":
        raise ValueError("Native DeltaNet S must reset on every decision")
    if memory_queries.get("initial_state") != "zeros":
        raise ValueError("This version requires a fixed zero initial memory")
    if memory_queries.get("mask_empty_read") is not True:
        raise ValueError("Empty read-memory slots must be masked")
    if memory_queries.get("force_first_write") is not True:
        raise ValueError("The first memory write must be complete")
    if memory_queries.get("update_rule") not in {"gated", "replace"}:
        raise ValueError("Memory update_rule must be gated or replace")
    if memory_queries.get("gate_granularity") != "slot":
        raise ValueError("This version requires one gate per memory slot")
    if not 0 < float(memory_queries.get("gate_initial_write_ratio", 0)) < 1:
        raise ValueError("gate_initial_write_ratio must be between zero and one")
    moba = config.framework.get("moba_memory", {})
    if not bool(moba.get("enabled", False)):
        raise ValueError("The mix recipe requires MoBA frame memory")
    if moba.get("history_query_scope") != "all_tokens":
        raise ValueError("Historical KV reads require all_tokens")
    if moba.get("cache_grad") != "detached":
        raise ValueError("Historical KV must stay detached")
    retrieval_settings(moba)
    if moba.get("attention_backend", "flex") not in {"flex", "dense_reference"}:
        raise ValueError("Unknown MoBA attention backend")
    capacity = moba.get("max_history_frames")
    if capacity is not None and int(capacity) < 1:
        raise ValueError("max_history_frames must be positive or null")
    normalize_cache_delta_indices(moba.get("cache_delta_indices", [0]), delta)


def validate_temporal_batch(examples, expected_views=2, *, data_config=None):
    if data_config is None:
        absolute, delta = [0], [-15, 0]
    else:
        expected_views = int(data_config.num_camera_views)
        absolute = list(data_config.temporal.image.get("absolute_indices", []))
        delta = list(data_config.temporal.image.delta_indices)
    profile = [
        ("first" if frame == 0 else "absolute", None, frame)
        for frame in absolute
    ] + [
        ("current" if offset == 0 else "history", offset, None)
        for offset in delta
    ]
    expected_count = expected_views * len(profile)
    for index, example in enumerate(examples):
        images = example.get("image")
        metas = example.get("image_metas")
        count = len(images) if images is not None else len(metas or [])
        if count != expected_count:
            raise ValueError(
                f"Sample {index} must contain {expected_count} temporal images, got {count}"
            )
        if metas is None or len(metas) != expected_count:
            raise ValueError(f"Temporal metadata must describe {expected_count} images")
        for time_index, expected in enumerate(profile):
            start = time_index * expected_views
            for meta in metas[start:start + expected_views]:
                actual = (
                    meta.get("time_role"), meta.get("delta"), meta.get("absolute_index")
                )
                if actual != expected:
                    raise ValueError(
                        f"Expected {expected}; got {actual} at time {time_index}"
                    )
            views = [meta.get("view") for meta in metas[start:start + expected_views]]
            if views != ["cam_high", "cam_follower_wrist"]:
                raise ValueError(f"Expected high/wrist camera order, got {views}")


@FRAMEWORK_REGISTRY.register(FRAMEWORK_NAME)
class RealQwen35Final(EventVLA):
    """Carry 64 embeddings; reset native S; let every valid token read past KV."""

    # The shared base class binarizes dimension 6; Piper's gripper is continuous.
    unnormalize_actions = staticmethod(unnormalize_actions)

    def __init__(self, config=None, **kwargs):
        validate_config(config)
        # The shared parent validates its original four-frame profile at init.
        # Use a private config for that check, then retain this package's profile.
        parent_config = copy.deepcopy(config)
        parent_config.datasets.vla_data.temporal.image.absolute_indices = [0]
        parent_config.datasets.vla_data.temporal.image.delta_indices = [-30, -15, 0]
        super().__init__(config=parent_config, **kwargs)
        self.config = config
        config.framework.action_model.action_hidden_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )
        if self.qwen_vl_interface.model_type != "qwen3_5":
            raise ValueError("Expected the Qwen3.5 backbone")
        if not isinstance(self.action_model, L1RegressionActionHead):
            raise ValueError("Expected the original EventVLA MLP action head")
        head = self.action_model.model
        if (head.fc1.in_features, head.fc1.out_features, head.fc2.out_features) != (2560, 5120, 7):
            raise ValueError("The original action head widths must be 2560, 5120, 7")
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        token_ids = tokenizer(self.action_token, add_special_tokens=False)["input_ids"]
        repeated = tokenizer(self.action_token * self.chunk_len, add_special_tokens=False)["input_ids"]
        if token_ids != [158778] or repeated != token_ids * 50:
            raise ValueError("Fifty diamond queries must encode as fifty adjacent native tokens")
        self.action_query_info = {
            "marker": self.action_token,
            "token_ids": token_ids,
            "query_token_id": self.action_token_id,
            "tokens_per_marker": 1,
            "query_count": self.chunk_len,
            "vocabulary_size": len(tokenizer),
            "vocabulary_modified": False,
        }
        self.qwen_vl_interface.model.config.use_cache = True
        self._layer_types = tuple(self.qwen_vl_interface.model.config.text_config.layer_types)
        install_padding_safe_deltanet(self.qwen_vl_interface.model.model.language_model)
        if self.qwen_vl_interface.model.is_gradient_checkpointing:
            raise RuntimeError("The backbone has gradient checkpointing enabled")
        settings = config.framework.memory_queries
        self.query_memory = QueryMemory(
            head.fc1.in_features, count=int(settings.count),
            initializer_range=float(settings.initializer_range),
            feedback_projection_init_scale=float(settings.feedback_projection_init_scale),
            update_rule=str(settings.update_rule),
            gate_initial_write_ratio=float(settings.gate_initial_write_ratio),
        )
        self._memory_pad_token_id = int(tokenizer.pad_token_id)
        self._last_query_memory_layout = None
        self._last_memory64_report = {}
        self._initialize_moba(config.framework.moba_memory)

    def _initialize_moba(self, settings):
        block_unit, topk = retrieval_settings(settings)
        self.moba_enabled = bool(settings.get("enabled", True))
        self.moba_backend = str(settings.get("attention_backend", "flex"))
        self.moba_cache_delta_indices = normalize_cache_delta_indices(
            settings.get("cache_delta_indices", [0])
        )
        self.moba_memory = SlotFrameKVMemory(
            [index for index, kind in enumerate(self._layer_types) if kind == "full_attention"],
            topk=topk,
            max_history_frames=settings.get("max_history_frames"),
            block_unit=block_unit,
        )
        self.moba_memory.reset()
        self._moba_context = None
        self._moba_examples = None
        self._moba_frame_specs = None
        install_moba_attention(self)

    def reset_memory(self, batch_size=1):
        super().reset_memory(batch_size=batch_size)
        if hasattr(self, "query_memory"):
            self.query_memory.last_gate = None
            self.query_memory.last_stats = {}
        if hasattr(self, "moba_memory"):
            self.moba_memory.reset(batch_size)
            self._moba_context = None

    def commit_moba_memory(self):
        self.moba_memory.commit()

    def discard_moba_memory(self):
        self.moba_memory.discard()

    def reset_memory_by_mask(self, reset_mask, episode_ids=None):
        super().reset_memory_by_mask(reset_mask, episode_ids=episode_ids)
        if reset_mask is not None and hasattr(self, "moba_memory"):
            self.moba_memory.reset_by_mask(reset_mask, episode_ids)

    def _prepare_moba_context(self, qwen_inputs, *, frame_specs, history_query_mask):
        examples = self._moba_examples
        if examples is None:
            raise RuntimeError("Query memory requires episode and slot metadata")
        self.moba_memory.prepare_slots(examples)
        specs = select_cache_frame_specs(frame_specs, self.moba_cache_delta_indices)
        positions = qwen_inputs["position_ids"].clone()
        valid = qwen_inputs["attention_mask"].bool()
        next_cursors = []
        for index, slot in enumerate(self.moba_memory.slots):
            positions[:, index, valid[index]] += slot.rope_cursor
            next_cursors.append(int(positions[:, index, valid[index]].max().item()) + 1)
        self._moba_context = MemoryContext(
            self.moba_memory, examples, specs, valid, next_cursors,
            history_query_mask=history_query_mask & valid,
        )
        return dict(qwen_inputs, position_ids=positions)

    def _build_recurrent_cache(self, state=None):
        if state is not None:
            raise ValueError("Native S cannot be passed between query-memory decisions")
        model = self.qwen_vl_interface.model
        return build_s_only_cache(model.config, model.model.language_model.layers, None)

    def _forward_qwen_with_state(self, qwen_inputs, state=None):
        original = qwen_inputs
        if qwen_inputs.get("inputs_embeds") is None:
            prepared, _ = prepare_window_vision_inputs(
                self.qwen_vl_interface.model.model, [qwen_inputs],
            )
            qwen_inputs = prepared[0]
        specs = self._moba_frame_specs
        if specs is None:
            specs = build_frame_specs(
                self._moba_examples, original["input_ids"], original["image_grid_thw"],
                image_token_id=self.qwen_vl_interface.model.config.image_token_id,
                deduplicate=False,
            )
        expanded, layout = self.query_memory.expand(
            qwen_inputs, state, action_token_id=self.action_token_id,
            pad_token_id=self._memory_pad_token_id, action_query_count=self.chunk_len,
        )
        self._last_query_memory_layout = layout
        expanded = self._prepare_moba_context(
            expanded, frame_specs=layout.remap_frame_specs(specs),
            history_query_mask=layout.history_query_mask,
        )
        cache = self._build_recurrent_cache()
        cold_layers = [cache.layers[index] for index, kind in enumerate(self._layer_types)
                       if kind == "linear_attention"]
        if any(layer.is_recurrent_states_initialized or layer.is_conv_states_initialized
               for layer in cold_layers):
            raise RuntimeError("Every native DeltaNet state must start empty")
        self._last_memory64_report = {
            "memory_tokens": 64, "cross_decision_s_layers": 0,
            "native_s_reset_layers": len(cold_layers),
            "memory_read_tokens": 64, "memory_write_tokens": 64,
            "memory_initial_state": "zeros",
            "memory_update_rule": self.query_memory.update_rule,
            "memory_gate_granularity": "slot",
            "memory_has_history": layout.has_memory.tolist(),
            "memory_valid_read_tokens": (layout.read_mask & layout.valid_mask).sum(-1).tolist(),
            "action_queries": self.chunk_len,
            "history_query_scope": "all_tokens",
            "history_query_tokens_per_sample": layout.history_query_mask.sum(-1).tolist(),
            "history_query_includes_memory_read": True,
            "history_query_includes_image_text": True,
            "input_tokens": int(original["input_ids"].shape[1]),
            "expanded_tokens": int(expanded["input_ids"].shape[1]),
        }
        model_inputs = dict(expanded)
        for key in ("labels", "past_key_values", "input_ids"):
            model_inputs.pop(key, None)
        device = expanded["input_ids"].device
        try:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outputs = self.qwen_vl_interface.model.model(
                    **model_inputs, past_key_values=cache, use_cache=True,
                    output_attentions=False, output_hidden_states=False, return_dict=True,
                )
                next_state = self.query_memory.write(outputs.last_hidden_state, layout)
            self._last_memory64_report.update(self.query_memory.last_stats)
            self._last_memory64_report["memory_gate_by_slot"] = self.query_memory.last_gate.cpu().tolist()
            hidden = outputs.last_hidden_state.gather(
                1, layout.original_indices[..., None].expand(-1, -1, outputs.last_hidden_state.shape[-1]),
            )
            # Keep feedback parameters in distributed autograd even at episode ends.
            hidden = hidden + (next_state.sum() * 0).to(hidden.dtype)
            if cache.get_seq_length() != 0:
                raise RuntimeError("Full-attention KV must remain exclusively in MoBA")
            self.moba_memory.stage(self._moba_context)
            return hidden, next_state
        except Exception:
            self.discard_moba_memory()
            raise
        finally:
            self._moba_context = None

    def _prepare_qwen_inputs(self, examples):
        if isinstance(examples, dict) and examples.get("_eventvla_worker_preprocessed", False):
            raw_examples = examples["examples"]
            device = self.qwen_vl_interface.model.device
            inputs = {
                key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                for key, value in examples["qwen_inputs"].items()
            }
        else:
            raw_examples = examples
            validate_temporal_batch(
                raw_examples, data_config=self.config.datasets.vla_data
            )
            images = [
                self._ensure_image_list(to_pil_preserve(example["image"]))
                for example in raw_examples
            ]
            size = self.config.datasets.vla_data.get("image_size")
            if size:
                images = resize_images(images, target_size=size)
            inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=images,
                instructions=self._build_vla_prompts([example["lang"] for example in raw_examples]),
                image_metas=self._infer_batch_image_metas(raw_examples, images),
                use_image_role_text=self.use_image_role_text,
            )
        return inputs, raw_examples

    def _predict_queries(self, qwen_inputs, state):
        hidden, next_state = self._forward_qwen_with_state(qwen_inputs, state)
        queries = self._gather_action_token_embeddings(
            hidden, qwen_inputs["input_ids"], action_token_id=self.action_token_id
        )
        dtype = self._module_compute_dtype(self.action_model, queries)
        prediction = self.action_model.predict_action(queries.to(dtype=dtype))
        return prediction, next_state

    def forward(
        self, examples=None, recurrent_state=None, return_recurrent_state=False,
        prepare_window_vision=False, **kwargs,
    ):
        if prepare_window_vision:
            inputs_by_time = [
                None if batch is None else self._prepare_qwen_inputs(batch)[0]
                for batch in examples
            ]
            prepared, report = prepare_window_vision_inputs(
                self.qwen_vl_interface.model.model, inputs_by_time
            )
            self._window_vision_report = report
            return [
                None if batch is None else dict(batch, qwen_inputs=inputs)
                for batch, inputs in zip(examples, prepared)
            ]
        qwen_inputs, raw_examples = self._prepare_qwen_inputs(examples)
        self._moba_examples = raw_examples
        self._moba_frame_specs = examples.get("moba_frame_specs") if isinstance(examples, dict) else None
        try:
            prediction, next_state = self._predict_queries(qwen_inputs, recurrent_state)
        finally:
            self._moba_examples = None
            self._moba_frame_specs = None
        target = torch.as_tensor(
            np.stack([example["action"] for example in raw_examples]),
            device=prediction.device, dtype=prediction.dtype,
        )[:, -self.chunk_len:, :]
        mask = action_mask_from_examples(raw_examples, self.chunk_len)
        action_loss = masked_action_l1(prediction, target, mask)
        valid_elements = float(mask.sum()) * (
            prediction.shape[-1] if mask.shape[-1] == 1 else 1
        )
        output = {
            "action_loss": action_loss,
            "total_loss": action_loss + self._zero_keyframe_head_loss(prediction),
            "action_valid_fraction": prediction.new_tensor(float(mask.mean())).detach(),
            "valid_action_elements": prediction.new_tensor(valid_elements, dtype=torch.float32),
        }
        for name, value in self.moba_memory.last_stats.items():
            output[name] = prediction.new_tensor(value, dtype=torch.float32)
        for name, value in self.query_memory.last_stats.items():
            output[name] = prediction.new_tensor(value, dtype=torch.float32)
        return (output, next_state) if return_recurrent_state else output

    def forward_recurrent_step(self, examples, recurrent_state=None):
        return self.forward(examples, recurrent_state, return_recurrent_state=True)

    @torch.inference_mode()
    def predict_action(self, examples=None, recurrent_state=None, return_recurrent_state=False, **kwargs):
        qwen_inputs, raw_examples = self._prepare_qwen_inputs(examples)
        self._moba_examples = raw_examples
        self._moba_frame_specs = examples.get("moba_frame_specs") if isinstance(examples, dict) else None
        try:
            prediction, next_state = self._predict_queries(qwen_inputs, recurrent_state)
        finally:
            self._moba_examples = None
            self._moba_frame_specs = None
        output = {"normalized_actions": prediction.float().cpu().numpy()}
        return (output, next_state) if return_recurrent_state else output
