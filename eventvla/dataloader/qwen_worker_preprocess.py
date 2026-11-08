"""Worker-side Qwen vision/text preprocessing for rank-local VLA batches."""

from __future__ import annotations

import json
import os
from typing import Any

from transformers import AutoProcessor


_CONFIG_ENV = "EVENTVLA_QWEN_WORKER_PREPROCESS_CONFIG"
_PROCESSOR = None
_PROCESSOR_MODEL_ID = None


def configure_qwen_worker_preprocessing(
    *,
    enabled: bool,
    model_id: str,
    action_token: str,
    chunk_len: int,
    use_image_role_text: bool,
    cot_prompt: str | None = None,
) -> None:
    """Publish a small JSON config inherited by DataLoader worker processes."""
    payload = {
        "enabled": bool(enabled),
        "model_id": str(model_id),
        "action_token": str(action_token),
        "chunk_len": int(chunk_len),
        "use_image_role_text": bool(use_image_role_text),
        "cot_prompt": cot_prompt,
    }
    os.environ[_CONFIG_ENV] = json.dumps(payload, separators=(",", ":"))
    global _PROCESSOR, _PROCESSOR_MODEL_ID
    _PROCESSOR = None
    _PROCESSOR_MODEL_ID = None


def _config() -> dict[str, Any]:
    raw = os.environ.get(_CONFIG_ENV, "")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid {_CONFIG_ENV}") from exc


def enabled() -> bool:
    return bool(_config().get("enabled", False))


def _get_processor():
    global _PROCESSOR, _PROCESSOR_MODEL_ID
    cfg = _config()
    model_id = cfg.get("model_id")
    if not cfg.get("enabled", False) or not model_id:
        return None
    if _PROCESSOR is None or _PROCESSOR_MODEL_ID != model_id:
        _PROCESSOR = AutoProcessor.from_pretrained(model_id)
        _PROCESSOR.tokenizer.padding_side = "left"
        _PROCESSOR_MODEL_ID = model_id
    return _PROCESSOR


def _build_image_content(images, metas, use_image_role_text: bool):
    if not use_image_role_text or metas is None or len(metas) != len(images):
        return [{"type": "image", "image": image} for image in images]

    content = []
    last_group = None
    for image, meta in zip(images, metas):
        role = str(meta.get("role", "")).lower() if isinstance(meta, dict) else ""
        group = "memory" if role == "memory_keyframe" else "anchor"
        if group != last_group:
            label = "Past keyframe images:" if group == "memory" else "Temporal observation images:"
            content.append({"type": "text", "text": label})
            last_group = group
        content.append({"type": "image", "image": image})
    return content


def _prompt(sample: dict, cfg: dict[str, Any]) -> str:
    instruction = sample.get("lang", sample.get("language", ""))
    suffix = (
        f" Please predict the next {int(cfg['chunk_len'])} robot actions and estimate "
        f"which future observation positions inside this chunk are key events: "
        f"<action>{cfg['action_token'] * int(cfg['chunk_len'])}<action>."
    )
    prompt = str(instruction) + suffix
    cot_prompt = cfg.get("cot_prompt")
    if cot_prompt:
        prompt = str(cot_prompt).replace("{instruction}", prompt)
    return prompt


def preprocess_batch(batch: list[dict]) -> dict:
    """Process a complete sampled batch inside the DataLoader worker."""
    cfg = _config()
    processor = _get_processor()
    if processor is None:
        return batch

    messages = []
    for sample in batch:
        content = _build_image_content(
            sample.get("image", []),
            sample.get("image_metas"),
            bool(cfg.get("use_image_role_text", False)),
        )
        content.append({"type": "text", "text": _prompt(sample, cfg)})
        messages.append([{"role": "user", "content": content}])

    batch_inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        processor_kwargs={
            "padding": True,
            "return_tensors": "pt",
        },
    )
    qwen_inputs = {key: value for key, value in batch_inputs.items()}

    # Keep action/state/language metadata, but avoid sending the 96 PIL images
    # through IPC a second time after their tensor representation is ready.
    examples = []
    image_counts = []
    for sample in batch:
        example = dict(sample)
        image_counts.append(len(sample.get("image", [])))
        example.pop("image", None)
        examples.append(example)
    return {
        "_eventvla_worker_preprocessed": True,
        "_eventvla_worker_image_counts": image_counts,
        "examples": examples,
        "qwen_inputs": qwen_inputs,
    }


def collate_qwen_worker_batch(batch: list[dict]):
    return preprocess_batch(batch) if enabled() else batch
