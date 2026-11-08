"""Raw Qwen3.5 EventVLA with S-only cross-decision memory."""

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
from .padding_safe_delta import install_padding_safe_deltanet
from .recurrent_cache import build_s_only_cache, extract_s_only_state
from .window_vision import prepare_window_vision_inputs

ACTION_TOKEN = "\u25c6"
FRAMEWORK_NAME = "RawQwen35RecurrentAbs0"


def validate_config(config):
    if config.framework.name != FRAMEWORK_NAME:
        raise ValueError(f"This package requires framework.name={FRAMEWORK_NAME}")
    if config.framework.memory_ablation_mode != "raw_anchors_only":
        raise ValueError("This version requires raw_anchors_only image injection")
    if config.framework.get("action_token", "") != ACTION_TOKEN:
        raise ValueError("The raw Qwen3.5 diamond action marker must be retained")
    action = config.framework.action_model
    if (action.action_dim, action.action_horizon, action.future_action_window_size,
            action.past_action_window_size) != (14, 50, 49, 0):
        raise ValueError("The EventVLA action layout must remain 50 x 14")
    if action.action_model_type != "MLP":
        raise ValueError("The original EventVLA MLP action head is required")
    data = config.datasets.vla_data
    if list(data.temporal.image.absolute_indices) != [0]:
        raise ValueError("The fixed anchor must be absolute episode frame 0")
    if list(data.temporal.image.delta_indices) != [-15, 0]:
        raise ValueError("The relative observations must be t-15 and t")
    if int(data.num_camera_views) != 3 or int(data.sampling_interval) != 50:
        raise ValueError("This recipe requires three views and a 50-frame decision stride")
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
    if config.framework.get("recurrent_state_mode") != "s_only":
        raise ValueError("Cross-decision state must be s_only")


def validate_temporal_batch(examples, expected_views=3):
    for index, example in enumerate(examples):
        images = example.get("image")
        metas = example.get("image_metas")
        count = len(images) if images is not None else len(metas or [])
        if count != expected_views * 3:
            raise ValueError(f"Sample {index} must contain nine temporal images, got {count}")
        if metas is None or len(metas) != expected_views * 3:
            raise ValueError("Temporal image metadata must describe nine images")
        profile = (("first", None, 0), ("history", -15, None), ("current", 0, None))
        for time_index, expected in enumerate(profile):
            start = time_index * expected_views
            for meta in metas[start:start + expected_views]:
                actual = (
                    meta.get("time_role"), meta.get("delta"), meta.get("absolute_index")
                )
                if actual != expected:
                    raise ValueError(
                        f"Expected frame 0, t-15 and t metadata; got {actual} at time {time_index}"
                    )


@FRAMEWORK_REGISTRY.register(FRAMEWORK_NAME)
class RawQwen35RecurrentAbs0(EventVLA):
    """Preserve raw EventVLA weights and prompts, carrying only DeltaNet S."""

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
        if (head.fc1.in_features, head.fc1.out_features, head.fc2.out_features) != (2560, 5120, 14):
            raise ValueError("The original action head widths must remain 2560, 5120, 14")
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

    def _build_recurrent_cache(self, state):
        model = self.qwen_vl_interface.model
        return build_s_only_cache(model.config, model.model.language_model.layers, state)

    def _forward_qwen_with_state(self, qwen_inputs, state=None):
        # The package's DeltaNet mask makes padding leave S unchanged, so
        # different prompt lengths can remain in one recurrent batch.
        cache = self._build_recurrent_cache(state)
        model_inputs = dict(qwen_inputs)
        model_inputs.pop("labels", None)
        model_inputs.pop("past_key_values", None)
        if model_inputs.get("inputs_embeds") is not None:
            if model_inputs.get("position_ids") is None:
                raise ValueError("Prepared image embeddings must retain native M-RoPE positions")
            model_inputs.pop("input_ids")
        if state is not None:
            batch_size = qwen_inputs["input_ids"].shape[0]
            if any(item is not None and item.shape[0] != batch_size for item in state):
                raise ValueError("The S batch dimension must match the current observations")
        device = qwen_inputs["input_ids"].device
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = self.qwen_vl_interface.model.model(
                **model_inputs,
                past_key_values=cache,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        next_state = extract_s_only_state(outputs.past_key_values, self._layer_types)
        return outputs.last_hidden_state, next_state

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
            validate_temporal_batch(raw_examples)
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
        qwen_inputs, _ = self._prepare_qwen_inputs(examples)
        prediction, next_state = self._predict_queries(qwen_inputs, recurrent_state)
        output = {"normalized_actions": prediction.float().cpu().numpy()}
        return (output, next_state) if return_recurrent_state else output
