"""Remove prompt padding before recurrent updates without mixing image patches."""

from __future__ import annotations

import torch


def unpadded_input_groups(inputs, images_per_sample=9):
    mask = inputs.get("attention_mask")
    if mask is None or bool(mask.all()):
        return [(list(range(inputs["input_ids"].shape[0])), inputs)]
    lengths = mask.sum(dim=1).tolist()
    groups = {}
    for index, length in enumerate(lengths):
        length = int(length)
        if length <= 0 or not bool(mask[index, -length:].all()):
            raise ValueError("The recurrent processor must use contiguous left padding")
        groups.setdefault(length, []).append(index)
    batch_size, padded_length = mask.shape
    image_grid = inputs.get("image_grid_thw")
    pixel_values = inputs.get("pixel_values")
    patch_offsets = None
    if image_grid is not None:
        if image_grid.shape[0] != batch_size * images_per_sample:
            raise ValueError("Image grids must contain nine images per sample")
        if pixel_values is not None:
            patches = image_grid.prod(dim=-1).tolist()
            patch_offsets = [0]
            for count in patches:
                patch_offsets.append(patch_offsets[-1] + int(count))
            if pixel_values.shape[0] != patch_offsets[-1]:
                raise ValueError("Image grids and packed vision patches disagree")
        elif inputs.get("inputs_embeds") is None:
            raise ValueError("Image grids require pixels or prepared image embeddings")
    result = []
    for length, indices in groups.items():
        selected = torch.tensor(indices, device=mask.device, dtype=torch.long)
        group = {}
        for key, value in inputs.items():
            if key in {"input_ids", "attention_mask", "mm_token_type_ids", "token_type_ids"}:
                group[key] = value.index_select(0, selected)[..., padded_length - length:]
            elif key == "inputs_embeds":
                group[key] = value.index_select(0, selected)[:, padded_length - length:, :]
            elif key == "position_ids":
                dimension = 1 if value.ndim == 3 else 0
                group[key] = value.index_select(dimension, selected)[..., padded_length - length:]
            elif key not in {"pixel_values", "image_grid_thw"}:
                group[key] = value
        if image_grid is not None:
            image_indices = [
                image for index in indices
                for image in range(index * images_per_sample, (index + 1) * images_per_sample)
            ]
            group["image_grid_thw"] = image_grid[image_indices]
            if pixel_values is not None:
                group["pixel_values"] = torch.cat([
                    pixel_values[patch_offsets[index * images_per_sample]:
                                 patch_offsets[(index + 1) * images_per_sample]]
                    for index in indices
                ], dim=0)
        result.append((indices, group))
    return result
