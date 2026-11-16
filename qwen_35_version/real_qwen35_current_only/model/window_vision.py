"""Batch independent image encoding before the recurrent language unroll."""

from __future__ import annotations

import torch


def prepare_window_vision_inputs(backbone, inputs_by_time):
    valid = [inputs for inputs in inputs_by_time if inputs is not None]
    report = {
        "valid_decisions": sum(inputs["input_ids"].shape[0] for inputs in valid),
        "images": sum(inputs["image_grid_thw"].shape[0] for inputs in valid),
        "vision_encoder_calls": int(bool(valid)),
    }
    if not valid:
        return list(inputs_by_time), report
    if any(inputs.get("pixel_values_videos") is not None for inputs in valid):
        raise ValueError("Window vision batching supports image observations only")

    pixels = torch.cat([inputs["pixel_values"] for inputs in valid], dim=0)
    grids = torch.cat([inputs["image_grid_thw"] for inputs in valid], dim=0)
    # Qwen's packed vision attention keeps each image separate using grid_thw.
    features = backbone.get_image_features(
        pixels, grids, return_dict=True
    ).pooler_output
    if len(features) != grids.shape[0]:
        raise RuntimeError("Qwen must return one feature tensor per image")

    prepared = []
    offset = 0
    for inputs in inputs_by_time:
        if inputs is None:
            prepared.append(None)
            continue
        count = inputs["image_grid_thw"].shape[0]
        embeddings = backbone.get_input_embeddings()(inputs["input_ids"])
        image_features = torch.cat(features[offset:offset + count], dim=0).to(
            device=embeddings.device, dtype=embeddings.dtype
        )
        image_mask, _ = backbone.get_placeholder_mask(
            inputs["input_ids"], embeddings, image_features=image_features
        )
        embeddings = embeddings.masked_scatter(image_mask, image_features)
        # The model accepts either IDs or embeddings. Preserve native M-RoPE
        # before the language forward consumes embeddings instead of IDs.
        positions, _ = backbone.get_rope_index(
            inputs["input_ids"],
            mm_token_type_ids=inputs["mm_token_type_ids"],
            image_grid_thw=inputs["image_grid_thw"],
            attention_mask=inputs.get("attention_mask"),
        )
        current = dict(inputs)
        current.pop("pixel_values")
        current["inputs_embeds"] = embeddings
        current["position_ids"] = positions
        prepared.append(current)
        offset += count
    report["embeddings_require_grad"] = all(
        inputs["inputs_embeds"].requires_grad
        for inputs in prepared if inputs is not None
    )
    return prepared, report
