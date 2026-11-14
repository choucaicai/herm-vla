"""Map temporal image roles to the visual tokens emitted by the processor."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import torch


def temporal_image_count(data_config):
    image = data_config["temporal"]["image"]
    groups = len(image.get("absolute_indices", [])) + len(image["delta_indices"])
    return int(data_config["num_camera_views"]) * groups


def normalize_cache_delta_indices(value, available_deltas=None):
    """None writes every input delta; an empty list disables visual KV writes."""
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("cache_delta_indices must be a list of integer input deltas or null")
    if any(isinstance(delta, bool) or not isinstance(delta, Integral) for delta in value):
        raise ValueError("cache_delta_indices must contain integer input deltas")
    deltas = tuple(int(delta) for delta in value)
    if len(set(deltas)) != len(deltas):
        raise ValueError("cache_delta_indices cannot contain duplicate deltas")
    if available_deltas is not None and not set(deltas).issubset(available_deltas):
        raise ValueError("cache_delta_indices must be a subset of temporal.image.delta_indices")
    return deltas


def select_cache_frame_specs(frame_specs, cache_delta_indices):
    """Filter temporal roles before deduplicating clipped physical frames."""
    deltas = normalize_cache_delta_indices(cache_delta_indices)
    allowed = None if deltas is None else set(deltas)
    result = []
    for groups in frame_specs:
        unique = {}
        for group in groups:
            if allowed is not None and group["delta"] not in allowed:
                continue
            frame = group["frame_timestep"]
            # Prefer the role closest to current among the selected input groups.
            if frame not in unique or abs(group["delta"]) < abs(unique[frame]["delta"]):
                unique[frame] = group
        result.append([unique[frame] for frame in sorted(unique)])
    return result


def build_frame_specs(examples, input_ids, image_grid_thw, image_token_id=248056,
                      *, deduplicate=True):
    grid = image_grid_thw.detach().cpu()
    ids = input_ids.detach().cpu()
    result = []
    image_cursor = 0
    for sample, row in zip(examples, ids):
        metas = sample.get("image_metas")
        if not metas or len(metas) % 2:
            raise ValueError("Piper MoBA requires complete groups of two temporal images")
        timestep = int(sample["timestep"])
        positions = (row == image_token_id).nonzero().flatten().tolist()
        token_cursor = 0
        groups = {}
        for image_index, meta in enumerate(metas):
            if image_cursor >= len(grid):
                raise ValueError("Missing image_grid_thw entries for temporal images")
            count = int(grid[image_cursor].prod().item()) // 4
            image_cursor += 1
            tokens = positions[token_cursor:token_cursor + count]
            token_cursor += count
            if not tokens or len(tokens) != count or tokens != list(range(tokens[0], tokens[0] + count)):
                raise ValueError("Image token spans do not match image_grid_thw")
            if meta.get("absolute_index") is not None:
                continue
            delta = meta.get("delta")
            if meta.get("time_role") not in {"history", "current"} or delta is None:
                raise ValueError("Unexpected relative temporal image role")
            if int(delta) > 0 and "frame_timestep" not in meta and "trajectory_length" not in sample:
                raise ValueError("Positive offsets require episode length or resolved frame metadata")
            frame = max(0, timestep + int(delta))
            if "trajectory_length" in sample:
                frame = min(int(sample["trajectory_length"]) - 1, frame)
            frame = int(meta.get("frame_timestep", frame))
            group_key = (int(meta.get("time_index", image_index // 2)), frame, int(delta))
            group = groups.setdefault(group_key, {
                "frame_timestep": frame, "delta": int(delta), "token_indices": [],
                "views": [], "image_spans": [],
            })
            view = str(meta.get("view", str(image_index % 2)))
            start = len(group["token_indices"])
            # Relative offsets survive insertion of the recurrent read/write tokens.
            group["image_spans"].append((view, start, start + len(tokens)))
            group["token_indices"].extend(tokens)
            group["views"].append(view)
        if token_cursor != len(positions):
            raise ValueError("Unmapped visual tokens remain in the sample")
        for group in groups.values():
            if len(group["views"]) != 2 or len(set(group["views"])) != 2:
                raise ValueError("Each historical time must contain exactly two cameras")
        result.append(list(groups.values()))
    if len(result) != len(examples) or image_cursor != len(grid):
        raise ValueError("Batch/image metadata alignment failed")
    return select_cache_frame_specs(result, None) if deduplicate else result
