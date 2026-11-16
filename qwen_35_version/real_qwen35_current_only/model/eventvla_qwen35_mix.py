"""Current-observation Qwen3.5: no F/M/H and no cross-decision native state."""

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
from .window_vision import prepare_window_vision_inputs
from ..data.temporal_profile import temporal_indices

ACTION_TOKEN = "\u25c6"
FRAMEWORK_NAME = "RealQwen35CurrentOnly"


def action_checkpoint_config(config):
    """Accept only checkpoints trained with this current-only architecture."""
    config = copy.deepcopy(config)
    if config.framework.name != FRAMEWORK_NAME:
        raise ValueError("Current-only serving requires its own trained checkpoint")
    return config


def validate_config(config):
    if config.framework.name != FRAMEWORK_NAME:
        raise ValueError(f"This package requires framework.name={FRAMEWORK_NAME}")
    if config.framework.memory_ablation_mode != "current_only":
        raise ValueError("This baseline requires current_only image injection")
    if config.framework.get("action_token", "") != ACTION_TOKEN:
        raise ValueError("The raw Qwen3.5 diamond action marker must be retained")
    action = config.framework.action_model
    if (action.action_dim, action.action_horizon, action.future_action_window_size,
            action.past_action_window_size) != (7, 50, 49, 0):
        raise ValueError("Piper actions must have shape 50 x 7")
    if action.action_model_type != "MLP":
        raise ValueError("The original EventVLA MLP action head is required")
    data = config.datasets.vla_data
    if data.get("action_mode") != "abs" or data.get("lerobot_version") != "v3.0":
        raise ValueError("Piper requires absolute positions and LeRobot v3.0")
    if data.get("include_state", False):
        raise ValueError("This image/language baseline does not consume proprioceptive state")
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
    if config.framework.get("recurrent_state_mode") != "none":
        raise ValueError("Current-only baseline cannot carry recurrent state")
    if absolute or tuple(delta) != (0,):
        raise ValueError("Current-only inputs require absolute=[] and delta=[0]")
    memory_queries = config.framework.get("memory_queries", {})
    if memory_queries.get("enabled") is not False or int(memory_queries.get("count", -1)) != 0:
        raise ValueError("Both memory read and memory write queries must be absent")
    if memory_queries.get("native_s_policy") != "reset_every_decision":
        raise ValueError("Native S/conv must reset every decision")
    moba = config.framework.get("moba_memory", {})
    if moba.get("enabled") is not False or list(moba.get("cache_delta_indices", [0])):
        raise ValueError("Historical KV reads and writes must both be disabled")


def validate_temporal_batch(examples, expected_views=2, *, data_config=None):
    if data_config is None:
        absolute, delta = [], [0]
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
            if any(int(meta["frame_timestep"]) != int(example["timestep"])
                   for meta in metas[start:start + expected_views]):
                raise ValueError("Only the current decision frame may be supplied")


@FRAMEWORK_REGISTRY.register(FRAMEWORK_NAME)
class RealQwen35CurrentOnly(EventVLA):
    """Run each current observation with a fresh, local backbone cache."""

    unnormalize_actions = staticmethod(unnormalize_actions)

    def __init__(self, config=None, **kwargs):
        validate_config(config)
        super().__init__(config=config, **kwargs)
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
            raise ValueError("Piper action head widths must be 2560, 5120, 7")
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
        self._last_current_only_report = {}

    def _forward_qwen_with_state(self, qwen_inputs, state=None):
        if state is not None:
            raise ValueError("Current-only policy rejects cross-decision state")
        original = qwen_inputs
        if qwen_inputs.get("inputs_embeds") is None:
            prepared, _ = prepare_window_vision_inputs(
                self.qwen_vl_interface.model.model, [qwen_inputs],
            )
            qwen_inputs = prepared[0]
        model = self.qwen_vl_interface.model
        cache = build_s_only_cache(model.config, model.model.language_model.layers, None)
        cold = [cache.layers[i] for i, kind in enumerate(self._layer_types)
                if kind == "linear_attention"]
        if any(layer.is_recurrent_states_initialized or layer.is_conv_states_initialized for layer in cold):
            raise RuntimeError("Native S/conv must start empty on every decision")
        if cache.get_seq_length() != 0:
            raise RuntimeError("Full-attention cache must start empty")
        self._last_current_only_report = {
            "memory_read_tokens": 0, "memory_write_tokens": 0,
            "cross_decision_s_layers": 0, "native_s_reset_layers": len(cold),
            "initial_full_attention_tokens": 0, "history_kv_tokens": 0,
            "input_tokens": int(original["input_ids"].shape[1]),
            "expanded_tokens": int(qwen_inputs["input_ids"].shape[1]),
            "action_query_counts": (original["input_ids"] == self.action_token_id).sum(-1).tolist(),
            "rope_policy": "local_per_decision",
        }
        model_inputs = dict(qwen_inputs)
        for key in ("labels", "past_key_values", "input_ids"):
            model_inputs.pop(key, None)
        device = qwen_inputs["input_ids"].device
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = model.model(
                **model_inputs, past_key_values=cache, use_cache=True,
                output_attentions=False, output_hidden_states=False, return_dict=True,
            )
        # All native S/conv and full-attention KV are local to this forward.
        return outputs.last_hidden_state, None

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
        prediction, next_state = self._predict_queries(qwen_inputs, recurrent_state)
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
        return (output, next_state) if return_recurrent_state else output

    def forward_recurrent_step(self, examples, recurrent_state=None):
        return self.forward(examples, recurrent_state, return_recurrent_state=True)

    @torch.inference_mode()
    def predict_action(self, examples=None, recurrent_state=None, return_recurrent_state=False, **kwargs):
        qwen_inputs, raw_examples = self._prepare_qwen_inputs(examples)
        prediction, next_state = self._predict_queries(qwen_inputs, recurrent_state)
        output = {"normalized_actions": prediction.float().cpu().numpy()}
        return (output, next_state) if return_recurrent_state else output
