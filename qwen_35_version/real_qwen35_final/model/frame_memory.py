"""Detached, transactional visual KV queues owned by episode batch slots."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import math

import torch
from torch.nn import functional as F


def retrieval_settings(settings):
    """Resolve one explicit routing unit without ambiguous Top-k configuration."""
    unit = str(settings.get("block_unit", "frame_time"))
    if unit not in {"frame_time", "image"}:
        raise ValueError("block_unit must be frame_time or image")
    selected = "topk_history_images" if unit == "image" else "topk_history_frames"
    other = "topk_history_frames" if unit == "image" else "topk_history_images"
    if other in settings:
        raise ValueError(f"block_unit={unit} requires {selected}, not {other}")
    topk = int(settings.get(selected, 4))
    if not 1 <= topk <= 8:
        raise ValueError(f"{selected} must be between one and eight")
    return unit, topk


@dataclass
class FrameEntry:
    frame_timestep: int
    layers: dict = field(default_factory=dict)
    image_spans: tuple = ()
    image_means: dict = field(default_factory=dict)


@dataclass
class EpisodeSlot:
    episode_id: object = None
    frames: OrderedDict = field(default_factory=OrderedDict)
    rope_cursor: int = 0
    last_timestep: int | None = None


class MemoryContext:
    def __init__(self, owner, examples, frame_specs, valid_tokens, next_cursors,
                 *, history_query_mask=None):
        self.owner = owner
        self.examples = examples
        self.valid_tokens = valid_tokens.bool()
        self.history_query_mask = history_query_mask
        self.next_cursors = next_cursors
        self.history = [list(slot.frames.values()) for slot in owner.slots[:len(examples)]]
        self.history_block_ids = [
            [(entry.frame_timestep, view) for entry in frames
             for view, _, _ in entry.image_spans]
            if owner.block_unit == "image"
            else [(entry.frame_timestep, None) for entry in frames]
            for frames in self.history
        ]
        self.new_frames = []
        self.layer_indices = owner.layer_indices
        self.layers_seen = set()
        self.route_indices = {}
        for slot, specs in zip(owner.slots, frame_specs):
            staged = []
            for spec in specs:
                frame = spec["frame_timestep"]
                if frame in slot.frames:
                    continue
                spans = tuple(tuple(span) for span in spec.get("image_spans", ()))
                if owner.block_unit == "image":
                    if len(spans) != 2 or len({span[0] for span in spans}) != 2:
                        raise ValueError("Piper image routing requires two distinct camera spans")
                    cursor = 0
                    for _, start, stop in spans:
                        if start != cursor or stop <= start:
                            raise ValueError("Camera spans must partition the cached visual tokens")
                        cursor = stop
                    if cursor != len(spec["token_indices"]):
                        raise ValueError("Camera spans do not cover the cached visual tokens")
                staged.append((
                    FrameEntry(frame, image_spans=spans),
                    torch.tensor(spec["token_indices"], device=valid_tokens.device, dtype=torch.long),
                ))
            self.new_frames.append(staged)

    def capture(self, layer_index, keys, values):
        if layer_index in self.layers_seen:
            raise RuntimeError("A full-attention layer wrote twice in one forward")
        self.layers_seen.add(layer_index)
        for sample_index, frames in enumerate(self.new_frames):
            for entry, indices in frames:
                # index_select already allocates storage independent of this call's KV.
                key = keys[sample_index].index_select(1, indices).detach()
                value = values[sample_index].index_select(1, indices).detach()
                entry.layers[layer_index] = (key, value, key.float().mean(dim=1))
                if self.owner.block_unit == "image":
                    entry.image_means[layer_index] = torch.stack([
                        key[:, start:stop].float().mean(dim=1)
                        for _, start, stop in entry.image_spans
                    ], dim=1)

    def _history_blocks(self, layer_index):
        """Slice each stored camera independently; no duplicate persistent KV."""
        unit = getattr(getattr(self, "owner", None), "block_unit", "frame_time")
        result = []
        for frames in self.history:
            blocks = []
            for entry in frames:
                key, value, mean = entry.layers[layer_index]
                if unit == "frame_time":
                    blocks.append((key, value, mean))
                else:
                    for view_index, (_, start, stop) in enumerate(entry.image_spans):
                        blocks.append((key[:, start:stop], value[:, start:stop],
                                       entry.image_means[layer_index][:, view_index]))
            result.append(blocks)
        return result

    def pack_history(self, layer_index, reference):
        batch, heads, _, dim = reference.shape
        history = self._history_blocks(layer_index)
        frame_count = max((len(blocks) for blocks in history), default=0)
        if frame_count == 0:
            return None
        width = max(
            key.shape[1] for blocks in history for key, _, _ in blocks
        )
        width = math.ceil(width / 64) * 64
        rows, frame_lengths = [], []
        for frames in history:
            frame_keys, frame_values, frame_means, lengths = [], [], [], []
            for key, value, mean in frames:
                if key.device != reference.device or key.dtype != reference.dtype:
                    raise RuntimeError("Cached KV device/dtype changed without an episode reset")
                count = key.shape[1]
                frame_keys.append(F.pad(key, (0, 0, 0, width - count)) if count < width else key)
                frame_values.append(F.pad(value, (0, 0, 0, width - count)) if count < width else value)
                frame_means.append(mean)
                lengths.append(count)
            if frames:
                key_row = torch.cat(frame_keys, dim=1)
                value_row = torch.cat(frame_values, dim=1)
                mean_row = torch.stack(frame_means, dim=1)
                extra = frame_count - len(frames)
                if extra:
                    key_row = F.pad(key_row, (0, 0, 0, extra * width))
                    value_row = F.pad(value_row, (0, 0, 0, extra * width))
                    mean_row = F.pad(mean_row, (0, 0, 0, extra))
            else:
                key_row = reference.new_zeros((heads, frame_count * width, dim))
                value_row = torch.zeros_like(key_row)
                mean_row = torch.zeros(
                    (heads, frame_count, dim), device=reference.device, dtype=torch.float32
                )
            rows.append((key_row, value_row, mean_row))
            frame_lengths.append(lengths + [0] * (frame_count - len(frames)))
        keys = torch.stack([row[0] for row in rows])
        values = torch.stack([row[1] for row in rows])
        means = torch.stack([row[2] for row in rows])
        lengths = torch.tensor(frame_lengths, device=reference.device, dtype=torch.long)
        return keys, values, means, lengths, width

    def stats(self):
        counts = [len(frames) for frames in self.history]
        image_counts = [sum(len(entry.image_spans) for entry in frames) for frames in self.history]
        block_counts = image_counts if self.owner.block_unit == "image" else counts
        block_tokens = [
            stop - start for frames in self.history for entry in frames
            for _, start, stop in entry.image_spans
        ] if self.owner.block_unit == "image" else [
            len(entry.layers[self.layer_indices[0]][0][0])
            for frames in self.history for entry in frames
        ]
        staged = [len(frames) for frames in self.new_frames]
        result = {
            "moba_history_frames": sum(counts) / max(len(counts), 1),
            "moba_history_frames_max": max(counts, default=0),
            "moba_new_frames": sum(staged) / max(len(staged), 1),
            "moba_history_images": sum(image_counts) / max(len(image_counts), 1),
            "moba_history_blocks": sum(block_counts) / max(len(block_counts), 1),
            "moba_history_block_tokens_max": max(block_tokens, default=0),
            "moba_topk_effective": sum(min(self.owner.topk, n) for n in block_counts) / max(len(counts), 1),
            "moba_history_attention_layers": len(self.route_indices),
        }
        if self.history_query_mask is not None:
            result["moba_history_queries_per_sample"] = (
                self.history_query_mask.sum().item() / max(len(counts), 1)
            )
        return result


class FrameKVMemory:
    def __init__(self, layer_indices, topk=4, max_history_frames=None, *, block_unit="frame_time"):
        if not 1 <= topk <= 8:
            raise ValueError("Top-k must be between one and eight")
        if block_unit not in {"frame_time", "image"}:
            raise ValueError("block_unit must be frame_time or image")
        self.block_unit = block_unit
        self.layer_indices = tuple(layer_indices)
        self.topk = int(topk)
        self.max_history_frames = max_history_frames
        self.slots = []
        self.pending = None
        self.last_stats = {}

    def reset(self, batch_size=1):
        self.slots = [EpisodeSlot() for _ in range(int(batch_size))]
        self.pending = None
        self.last_stats = {}

    def reset_by_mask(self, reset_mask, episode_ids=None):
        mask = reset_mask.detach().cpu().bool().tolist()
        while len(self.slots) < len(mask):
            self.slots.append(EpisodeSlot())
        for index, reset in enumerate(mask):
            if reset:
                self.slots[index] = EpisodeSlot(
                    episode_id=None if episode_ids is None else episode_ids[index]
                )

    def prepare_slots(self, examples):
        if self.pending is not None:
            raise RuntimeError("Commit or discard pending visual KV before another forward")
        while len(self.slots) < len(examples):
            self.slots.append(EpisodeSlot())
        for index, sample in enumerate(examples):
            episode = sample.get("episode_id")
            if episode is None:
                raise ValueError("Stateful visual memory requires episode_id")
            slot = self.slots[index]
            if sample.get("is_new_episode", False) or slot.episode_id != episode:
                self.slots[index] = slot = EpisodeSlot(episode_id=episode)
            timestep = int(sample["timestep"])
            if slot.last_timestep is not None and timestep <= slot.last_timestep:
                raise ValueError("Episode timesteps must increase; repeated requests need decision IDs")

    def stage(self, context):
        if context.layers_seen != set(self.layer_indices):
            raise RuntimeError("The visual cache did not capture every full-attention layer")
        self.pending = context
        self.last_stats = context.stats()

    def commit(self):
        context = self.pending
        if context is None:
            return
        for index, (sample, frames, cursor) in enumerate(zip(
            context.examples, context.new_frames, context.next_cursors
        )):
            slot = self.slots[index]
            for entry, _ in frames:
                slot.frames.setdefault(entry.frame_timestep, entry)
            if self.max_history_frames is not None:
                while len(slot.frames) > int(self.max_history_frames):
                    slot.frames.popitem(last=False)
            slot.rope_cursor = int(cursor)
            slot.last_timestep = int(sample["timestep"])
            if bool(sample.get("done", False)):
                self.slots[index] = EpisodeSlot(episode_id=slot.episode_id)
        self.pending = None

    def discard(self):
        self.pending = None

    def state_summary(self):
        return [{
            "episode_id": slot.episode_id,
            "frame_timesteps": list(slot.frames),
            "block_unit": self.block_unit,
            "cached_images": sum(len(entry.image_spans) for entry in slot.frames.values()),
            "rope_cursor": slot.rope_cursor,
            "last_timestep": slot.last_timestep,
        } for slot in self.slots]
