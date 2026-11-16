"""Stride-aware episode streams with per-decision slot refill."""

from __future__ import annotations

import copy
import hashlib
import os
from bisect import bisect_right
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True, slots=True)
class RememWindow:
    """One policy-decision unit in a strided episode view."""

    dataset_index: int
    trajectory_id: int
    start: int
    valid_length: int = 1
    phase: int = 0
    logical_index: int = 0
    episode_id: str = ""
    view_id: str = ""


@dataclass(frozen=True, slots=True)
class RememScheduledWindow:
    """A scheduled TBPTT sequence.

    ``window_index`` is retained for compatibility with the first ReMem
    implementation.  The new lazy sampler records the episode, physical start
    and ``sequence_length=K`` directly; ``window_indices`` is accepted only
    for backward compatibility.  ``valid_length`` counts real decisions, not
    padded rows.
    """

    window_index: int
    slot_id: int
    is_first: bool
    is_last: bool
    valid_length: int | None = None
    episode_id: str | None = None
    dummy: bool = False
    window_indices: tuple[int, ...] | None = None
    view_id: str | None = None
    phase: int | None = None
    logical_start: int | None = None
    physical_start: int | None = None
    dataset_index: int | None = None
    trajectory_id: int | None = None
    sequence_length: int | None = None
    source_epoch: int | None = None
    stream_visit_id: int | None = None


# More descriptive name for new callers; keep the old exported name above.
RememScheduledSequence = RememScheduledWindow


@dataclass(frozen=True, slots=True)
class RememSlotWindow:
    """K decisions in one slot, including independent episode boundaries."""

    slot_id: int
    step_descriptors: tuple[RememScheduledWindow, ...]
    completed_epochs: int
    stream_window_index: int


@dataclass(frozen=True, slots=True)
class _RememEpisode:
    dataset_index: int
    trajectory_id: int
    trajectory_length: int
    index_length: int
    global_offset: int
    episode_id: str


@dataclass(frozen=True, slots=True)
class _RememViewSchedule:
    episode: _RememEpisode
    phase: int
    view_id: str
    decision_count: int
    sequence_count: int


class _LazyWindowSequence(Sequence[RememWindow]):
    """Compatibility view without allocating one Python object per frame."""

    def __init__(self, dataset: "RememEpisodeWindowDataset"):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int | slice) -> RememWindow | list[RememWindow]:
        if isinstance(index, slice):
            return [self.dataset._window_at(i) for i in range(*index.indices(len(self)))]
        return self.dataset._window_at(index)


class _LazyViewMapping(Mapping[str, tuple[int, ...]]):
    """Expose legacy view membership without materializing every member."""

    def __init__(self, dataset: "RememEpisodeWindowDataset"):
        self.dataset = dataset

    def __len__(self) -> int:
        return sum(len(phases) for phases in self.dataset._episode_phases.values())

    def __iter__(self):
        for episode in self.dataset._episodes:
            for phase in self.dataset._episode_phases[episode.episode_id]:
                yield self.dataset._view_key(episode.episode_id, phase)

    def __getitem__(self, view_id: str) -> tuple[int, ...]:
        return self.dataset.windows_for_view(view_id)


class RememEpisodeWindowDataset(Dataset):
    """Expose one-step strided views over one or more source datasets.

    ``window_size`` is the number of consecutive raw steps in an ordinary
    (non-scheduled) item and should be 1 for stateful TBPTT.  ``window_stride``
    is the distance between policy decisions and is independent of K.  With
    the requested setup these are ``window_size=1, window_stride=50``.

    For each source trajectory, all valid phases ``0..stride-1`` are indexed:
    phase ``r`` contains ``r, r+stride, r+2*stride, ...``.  The sampler picks
    one phase for a trajectory visit and keeps it fixed until that view ends.
    """

    def __init__(
        self,
        dataset: Any | Sequence[Any],
        window_size: int = 1,
        window_stride: int | None = None,
        *,
        decision_stride: int | None = None,
        drop_last: bool = False,
        reset_each_window: bool = True,
        batch_tbptt_video: bool = True,
    ):
        if isinstance(dataset, (list, tuple)):
            self.datasets = list(dataset)
        elif not hasattr(dataset, "trajectory_ids") and hasattr(dataset, "datasets"):
            self.datasets = list(dataset.datasets)
        else:
            self.datasets = [dataset]
        if not self.datasets:
            raise ValueError("RememEpisodeWindowDataset requires at least one dataset")

        self.window_size = int(window_size)
        selected_stride = decision_stride if decision_stride is not None else window_stride
        if selected_stride is None:
            selected_stride = self.window_size
        self.window_stride = int(selected_stride)
        self.decision_stride = self.window_stride
        if self.window_size <= 0 or self.window_stride <= 0:
            raise ValueError("window_size and window_stride must be positive")
        self.drop_last = bool(drop_last)
        self.reset_each_window = bool(reset_each_window)
        self.batch_tbptt_video = bool(batch_tbptt_video)
        self.epoch = 0
        self.tag = getattr(self.datasets[0], "tag", None)
        self.merged_metadata = {
            getattr(source, "tag", str(index)): getattr(source, "metadata", None)
            for index, source in enumerate(self.datasets)
            if getattr(source, "metadata", None) is not None
        }

        self._episodes: list[_RememEpisode] = []
        self._episode_by_id: dict[str, _RememEpisode] = {}
        self._episode_ends: list[int] = []
        self._episode_phases: dict[str, tuple[int, ...]] = {}
        self._trajectory_lengths: dict[tuple[int, int], int] = {}
        self._dummy_template = None
        self._build_windows()
        # Older callers index ``dataset.windows``.  Keep that interface lazy:
        # the full composite soup contains roughly 33M raw steps, so eagerly
        # allocating one dataclass plus one list integer per step would consume
        # tens of gigabytes on every distributed rank.
        self.windows: Sequence[RememWindow] = _LazyWindowSequence(self)
        self._lazy_view_windows = _LazyViewMapping(self)

    @staticmethod
    def _episode_key(dataset_index: int, trajectory_id: int) -> str:
        # Include the source index: two datasets in a soup may both contain
        # trajectory 0.
        return f"{int(dataset_index)}:{int(trajectory_id)}"

    @staticmethod
    def _view_key(episode_id: str, phase: int) -> str:
        return f"{episode_id}/phase{int(phase)}"

    def _build_windows(self) -> None:
        episodes: list[_RememEpisode] = []
        total = 0
        for dataset_index, source in enumerate(self.datasets):
            trajectory_ids = np.asarray(source.trajectory_ids).reshape(-1)
            trajectory_lengths = np.asarray(source.trajectory_lengths).reshape(-1)
            if len(trajectory_ids) != len(trajectory_lengths):
                raise ValueError("trajectory_ids and trajectory_lengths must align")
            for trajectory_id, length_value in zip(trajectory_ids, trajectory_lengths):
                length = int(length_value)
                if length <= 0:
                    continue
                trajectory_id = int(trajectory_id)
                episode_id = self._episode_key(dataset_index, trajectory_id)
                self._trajectory_lengths[(dataset_index, trajectory_id)] = length
                index_length = (
                    max(0, length - self.window_size + 1)
                    if self.drop_last
                    else length
                )
                if index_length <= 0:
                    continue
                episode = _RememEpisode(
                    dataset_index=dataset_index,
                    trajectory_id=trajectory_id,
                    trajectory_length=length,
                    index_length=index_length,
                    global_offset=total,
                    episode_id=episode_id,
                )
                episodes.append(episode)
                self._episode_by_id[episode_id] = episode
                self._episode_phases[episode_id] = tuple(
                    range(min(self.window_stride, index_length))
                )
                total += index_length
                self._episode_ends.append(total)
        if not episodes or total <= 0:
            raise ValueError("No non-empty trajectories available for ReMem windows")
        self._episodes = episodes

    def _window_at(self, index: int) -> RememWindow:
        length = len(self)
        index = int(index)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError(index)
        episode_position = bisect_right(self._episode_ends, index)
        episode = self._episodes[episode_position]
        start = index - episode.global_offset
        phase = start % self.window_stride
        logical_index = start // self.window_stride
        return RememWindow(
            dataset_index=episode.dataset_index,
            trajectory_id=episode.trajectory_id,
            start=start,
            valid_length=min(self.window_size, episode.trajectory_length - start),
            phase=phase,
            logical_index=logical_index,
            episode_id=episode.episode_id,
            view_id=self._view_key(episode.episode_id, phase),
        )

    @property
    def episode_ids(self) -> tuple[str, ...]:
        return tuple(episode.episode_id for episode in self._episodes)

    @property
    def episode_phases(self) -> dict[str, tuple[int, ...]]:
        return dict(self._episode_phases)

    @property
    def view_windows(self) -> Mapping[str, tuple[int, ...]]:
        return self._lazy_view_windows

    def windows_for_view(self, view_id: str) -> tuple[int, ...]:
        try:
            episode_id, phase_text = view_id.rsplit("/phase", 1)
            phase = int(phase_text)
            episode = self._episode_by_id[episode_id]
            if phase not in self._episode_phases[episode_id]:
                raise KeyError(view_id)
        except (KeyError, ValueError) as exc:
            raise KeyError(f"Unknown ReMem episode view {view_id!r}") from exc
        return tuple(
            episode.global_offset + start
            for start in range(phase, episode.index_length, self.window_stride)
        )

    def __len__(self) -> int:
        return self._episode_ends[-1]

    def __str__(self) -> str:
        return (
            f"RememEpisodeWindowDataset({len(self)} lazy one-step windows, "
            f"episodes={len(self._episode_phases)}, phases/episode<="
            f"{self.window_stride}, window_size={self.window_size}, "
            f"stride={self.window_stride})"
        )

    @property
    def modality_configs(self):
        return getattr(self.datasets[0], "modality_configs", None)

    @property
    def metadata(self):
        return getattr(self.datasets[0], "metadata", None)

    @property
    def dataset_name(self) -> str:
        names = [getattr(dataset, "dataset_name", str(dataset)) for dataset in self.datasets]
        return "+".join(names)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        for source in self.datasets:
            setter = getattr(source, "set_epoch", None)
            if setter is not None:
                setter(epoch)

    def _transform_step(
        self, source: Any, trajectory_id: int, base_index: int, raw: dict | None = None
    ) -> dict:
        if raw is None:
            raw = source.get_step_data(trajectory_id, int(base_index))
        transform = getattr(source, "transforms", None)
        transformed = transform(raw) if transform is not None else raw
        # StarVLA's transform still contains separate modality keys. Its normal
        # packer produces current PIL images, language and normalized actions.
        step = source._pack_sample(transformed)
        step["robot_tag"] = source.tag
        return step

    def _action_valid_mask(
        self,
        dataset_index: int,
        trajectory_id: int,
        base_index: int,
        step: dict[str, Any],
    ) -> np.ndarray | None:
        """Return a temporal action-valid mask for padded future targets."""

        if "action" not in step or not hasattr(step["action"], "shape"):
            return None
        action = np.asarray(step["action"])
        if action.ndim < 2:
            return None
        horizon = int(action.shape[0])
        source = self.datasets[int(dataset_index)]
        modality_configs = getattr(source, "modality_configs", None) or {}
        action_cfg = modality_configs.get("action")
        deltas = getattr(action_cfg, "delta_indices", None)
        if deltas is None:
            return np.ones((horizon, 1), dtype=np.float32)
        deltas = np.asarray(deltas).reshape(-1)
        length = self._trajectory_lengths[(int(dataset_index), int(trajectory_id))]
        if len(deltas) != horizon:
            raise ValueError(f"Action offsets ({len(deltas)}) do not match horizon ({horizon})")
        indices = int(base_index) + deltas.astype(np.int64)
        return ((indices >= 0) & (indices < length)).astype(np.float32)[:, None]

    def _load_unit(
        self, window: RememWindow, raw: dict | None = None
    ) -> tuple[dict[str, Any], np.ndarray | None]:
        step = self._transform_step(
            self.datasets[window.dataset_index], window.trajectory_id, window.start, raw
        )
        length = self._trajectory_lengths[(window.dataset_index, window.trajectory_id)]
        step["trajectory_length"] = length
        for meta in step.get("image_metas", []):
            absolute = meta.get("absolute_index")
            delta = meta.get("delta")
            if absolute is None and delta is None:
                raise ValueError("Temporal image metadata must include a frame index")
            requested = int(absolute) if absolute is not None else window.start + int(delta)
            # Use the same episode-boundary clipping as the video reader.
            meta["frame_timestep"] = min(length - 1, max(0, requested))
        return step, self._action_valid_mask(
            window.dataset_index, window.trajectory_id, window.start, step
        )

    def _get_raw_units(self, windows: Sequence[RememWindow]) -> list[dict]:
        """Fetch each trajectory's union of camera frames before transforming."""
        groups = {}
        for position, window in enumerate(windows):
            groups.setdefault((window.dataset_index, window.trajectory_id), []).append(position)
        raw_steps = [None] * len(windows)
        for (dataset_index, trajectory_id), positions in groups.items():
            source = self.datasets[dataset_index]
            starts = [windows[position].start for position in positions]
            batch_reader = getattr(source, "get_step_data_batch", None)
            raw = (
                batch_reader(trajectory_id, starts) if batch_reader is not None
                else [source.get_step_data(trajectory_id, start) for start in starts]
            )
            if len(raw) != len(positions):
                raise ValueError("The batch reader must return one raw sample per decision")
            for position, data in zip(positions, raw):
                raw_steps[position] = data
        return raw_steps

    def _load_units(self, windows: Sequence[RememWindow]):
        if not self.batch_tbptt_video:
            return [self._load_unit(window) for window in windows]
        raw_steps = self._get_raw_units(windows)
        return [self._load_unit(window, raw) for window, raw in zip(windows, raw_steps)]

    def _load_stream_slot(self, packet: RememSlotWindow) -> dict[str, Any]:
        descriptors = packet.step_descriptors
        raw_steps = [None] * len(descriptors)
        epochs = []
        current_epoch = self.epoch
        groups = {}
        for position, descriptor in enumerate(descriptors):
            if descriptor.slot_id != packet.slot_id or descriptor.sequence_length != 1:
                raise ValueError("A stream packet must contain one-decision descriptors for its slot")
            if descriptor.source_epoch is not None:
                current_epoch = descriptor.source_epoch
            epochs.append(current_epoch)
            if (self.batch_tbptt_video and not descriptor.dummy
                    and descriptor.valid_length != 0
                    and descriptor.dataset_index is not None
                    and descriptor.trajectory_id is not None
                    and descriptor.physical_start is not None):
                groups.setdefault(current_epoch, []).append(position)
        for epoch, positions in groups.items():
            if epoch != self.epoch:
                self.set_epoch(epoch)
            windows = [RememWindow(
                dataset_index=int(descriptors[position].dataset_index),
                trajectory_id=int(descriptors[position].trajectory_id),
                start=int(descriptors[position].physical_start),
            ) for position in positions]
            raw = self._get_raw_units(windows)
            for position, data in zip(positions, raw):
                raw_steps[position] = data
        units = []
        for position, descriptor in enumerate(descriptors):
            if epochs[position] != self.epoch:
                self.set_epoch(epochs[position])
            units.append(RememEpisodeWindowDataset.__getitem__(
                self, descriptor, _raw_data=raw_steps[position]
            ))
        if not units:
            raise ValueError("A stream slot window must contain at least one decision")
        metadata = []
        for unit in units:
            valid = not unit["dummy"] and unit["valid_length"] == 1
            metadata.append({
                "episode_id": unit["episode_id"],
                "view_id": unit["view_id"],
                "trajectory_id": unit["trajectory_id"],
                "phase": unit["phase"],
                "timestep": unit["physical_indices"][0],
                "logical_step": unit["logical_indices"][0],
                "anchor_index": unit["logical_indices"][0],
                "slot_id": packet.slot_id,
                "chunk_index": unit["logical_indices"][0],
                "is_new_episode": valid and unit["reset"],
                "reset": valid and unit["reset"],
                "is_last": valid and unit["is_last"],
                "done": valid and unit["is_last"],
                "step_valid": valid,
                "dummy": unit["dummy"],
                "remem_action_valid_mask": unit["action_valid_masks"][0],
                "action_valid_mask": unit["action_valid_masks"][0],
                "source_epoch": unit["source_epoch"],
                "stream_visit_id": unit["stream_visit_id"],
            })
        result = dict(units[0])
        result.update(
            steps=[unit["steps"][0] for unit in units],
            physical_indices=[unit["physical_indices"][0] for unit in units],
            logical_indices=[unit["logical_indices"][0] for unit in units],
            action_valid_masks=[unit["action_valid_masks"][0] for unit in units],
            valid_length=sum(item["step_valid"] for item in metadata),
            dummy=not any(item["step_valid"] for item in metadata),
            is_last=units[-1]["is_last"],
            step_metadata=metadata,
            continuous_refill=True,
            stream_epoch=packet.completed_epochs,
            stream_window_index=packet.stream_window_index,
        )
        return result

    def __getitem__(
        self, index: int | RememScheduledWindow | RememSlotWindow, *, _raw_data: dict | None = None
    ) -> dict[str, Any]:
        if isinstance(index, RememSlotWindow):
            return self._load_stream_slot(index)
        schedule = index if isinstance(index, RememScheduledWindow) else None
        if schedule is None:
            window = self.windows[int(index)]
            steps: list[dict[str, Any]] = []
            action_masks: list[np.ndarray | None] = []
            windows = []
            for offset in range(window.valid_length):
                unit = RememWindow(
                    dataset_index=window.dataset_index,
                    trajectory_id=window.trajectory_id,
                    start=window.start + offset,
                    phase=window.phase,
                    logical_index=window.logical_index + offset,
                    episode_id=window.episode_id,
                    view_id=window.view_id,
                )
                windows.append(unit)
            for step, mask in self._load_units(windows):
                steps.append(step)
                action_masks.append(mask)
            if not steps:
                raise RuntimeError(f"Empty ReMem window: {window}")
            sequence_length = max(1, window.valid_length)
            while len(steps) < sequence_length:
                steps.append(copy.deepcopy(steps[-1]))
                action_masks.append(None if action_masks[-1] is None else action_masks[-1].copy())
            physical_indices = [window.start + offset for offset in range(sequence_length)]
            logical_indices = [window.logical_index + offset for offset in range(sequence_length)]
            valid_length = int(window.valid_length)
            episode_id = window.episode_id
            view_id = window.view_id
            phase = window.phase
            reset = self.reset_each_window
            episode = self._episode_by_id[window.episode_id]
            is_last = window.start + self.window_stride >= episode.index_length
            slot_id = -1
            dummy = False
            trajectory_id = window.trajectory_id
        else:
            sequence_length = int(
                schedule.sequence_length
                if schedule.sequence_length is not None
                else len(schedule.window_indices or (schedule.window_index,))
            )
            if sequence_length <= 0:
                raise ValueError("Scheduled ReMem sequence must contain at least one unit")
            valid_length = int(
                schedule.valid_length if schedule.valid_length is not None else sequence_length
            )
            valid_length = max(0, min(valid_length, sequence_length))

            if schedule.dataset_index is not None and schedule.trajectory_id is not None:
                dataset_index = int(schedule.dataset_index)
                trajectory_id = int(schedule.trajectory_id)
                real_episode_id = self._episode_key(dataset_index, trajectory_id)
                episode_id = schedule.episode_id or real_episode_id
                episode = self._episode_by_id[real_episode_id]
                physical_start = int(
                    schedule.physical_start if schedule.physical_start is not None else 0
                )
                logical_start = int(
                    schedule.logical_start if schedule.logical_start is not None else 0
                )
            else:
                # Compatibility with descriptors produced by the original
                # materialized-window implementation.
                indices = list(schedule.window_indices or (schedule.window_index,))
                first_window = self.windows[int(indices[0]) if valid_length else 0]
                dataset_index = first_window.dataset_index
                trajectory_id = first_window.trajectory_id
                episode_id = schedule.episode_id or first_window.episode_id
                episode = self._episode_by_id[first_window.episode_id]
                physical_start = int(
                    schedule.physical_start
                    if schedule.physical_start is not None
                    else first_window.start
                )
                logical_start = int(
                    schedule.logical_start
                    if schedule.logical_start is not None
                    else first_window.logical_index
                )

            phase = int(
                schedule.phase
                if schedule.phase is not None
                else physical_start % self.window_stride
            )
            view_id = schedule.view_id or self._view_key(episode_id, phase)
            # Dummy rows still need a correctly shaped example for the normal
            # collator.  They load the first real step, then are fully masked.
            template_start = (
                physical_start
                if valid_length > 0 and not schedule.dummy
                else 0
            )
            template_episode = (
                episode
                if valid_length > 0 and not schedule.dummy
                else self._episodes[0]
            )
            template_window = RememWindow(
                dataset_index=template_episode.dataset_index,
                trajectory_id=template_episode.trajectory_id,
                start=template_start,
                phase=phase,
                logical_index=logical_start,
                episode_id=template_episode.episode_id,
                view_id=self._view_key(template_episode.episode_id, phase),
            )
            if valid_length > 0 and not schedule.dummy:
                if _raw_data is not None:
                    if sequence_length != 1:
                        raise ValueError("A preloaded raw sample must describe one decision")
                    loaded_units = [self._load_unit(template_window, _raw_data)]
                else:
                    windows = []
                    for position in range(valid_length):
                        unit_start = physical_start + position * self.window_stride
                        if unit_start >= episode.index_length:
                            raise ValueError(
                                f"Scheduled ReMem step {unit_start} exceeds episode "
                                f"index length {episode.index_length}"
                            )
                        windows.append(RememWindow(
                            dataset_index=dataset_index, trajectory_id=trajectory_id,
                            start=unit_start, phase=phase,
                            logical_index=logical_start + position,
                            episode_id=episode_id, view_id=view_id,
                        ))
                    loaded_units = self._load_units(windows)
                template_step, template_mask = loaded_units[0]
            else:
                if self._dummy_template is None:
                    self._dummy_template = self._load_unit(template_window)
                template_step, template_mask = self._dummy_template
            steps = []
            action_masks = []
            physical_indices = []
            logical_indices = []
            last_real_start = physical_start + max(0, valid_length - 1) * self.window_stride
            for position in range(sequence_length):
                if position < valid_length and not schedule.dummy:
                    unit_start = physical_start + position * self.window_stride
                    if unit_start >= episode.index_length:
                        raise ValueError(
                            f"Scheduled ReMem step {unit_start} exceeds episode "
                            f"index length {episode.index_length}"
                        )
                    step, mask = loaded_units[position]
                    physical_indices.append(unit_start)
                else:
                    # Padding is never mutated or supervised; reuse its image/array storage.
                    step = dict(template_step)
                    mask = template_mask
                    physical_indices.append(last_real_start if valid_length else physical_start)
                logical_indices.append(logical_start + position)
                steps.append(step)
                action_masks.append(mask)
            reset = bool(schedule.is_first)
            is_last = bool(schedule.is_last)
            slot_id = int(schedule.slot_id)
            dummy = bool(schedule.dummy)

        return {
            "steps": steps,
            "episode_id": episode_id,
            "episode_length": int(episode.index_length),
            "view_id": view_id,
            "phase": int(phase),
            "trajectory_id": int(trajectory_id),
            "start_index": int(physical_indices[0]),
            "logical_start_index": int(logical_indices[0]),
            "physical_indices": physical_indices,
            "logical_indices": logical_indices,
            "valid_length": int(valid_length),
            "reset": bool(reset),
            "is_last": bool(is_last),
            "slot_id": int(slot_id),
            "chunk_index": int(logical_indices[0]),
            "dummy": bool(dummy),
            "action_valid_masks": action_masks,
            "source_epoch": schedule.source_epoch if schedule is not None else self.epoch,
            "stream_visit_id": schedule.stream_visit_id if schedule is not None else None,
        }


class RememSlotBatchSampler:
    """RoboCasa slot schedule, explicitly sharded for StarVLA's unwrapped loader.

    Sampling starts a fresh schedule on resume. No exact checkpoint cursor is
    maintained. Task balancing is the existing RMBench policy, applied to
    decision counts rather than to raw frames.
    """

    def __init__(self, dataset, batch_size, *, tbptt_steps=4, rank=0,
                 world_size=1, seed=42, phase_mode="balanced_random",
                 balance_task_step_counts=True, shuffle_trajectories=True,
                 continuous_refill=False):
        self.dataset = dataset
        self.local_batch_size = self.batch_size = int(batch_size)
        self.tbptt_steps = int(tbptt_steps)
        self.rank, self.world_size = int(rank), int(world_size)
        if self.local_batch_size < 1 or self.tbptt_steps < 1:
            raise ValueError("batch_size and tbptt_steps must be positive")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("Invalid distributed rank")
        if phase_mode not in {"balanced_random", "fixed_zero"}:
            raise ValueError("phase_mode must be balanced_random or fixed_zero")
        self.global_batch_size = self.local_batch_size * self.world_size
        self.seed = self.phase_seed = int(seed)
        self.phase_mode = phase_mode
        self.balance_task_step_counts = bool(balance_task_step_counts)
        self.shuffle_trajectories = bool(shuffle_trajectories)
        self.continuous_refill = bool(continuous_refill)
        self.split_batches = True
        self.epoch = 0
        self._tick_count_cache = {}

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self.dataset.set_epoch(self.epoch)
        self._tick_count_cache.clear()

    @staticmethod
    def _stable_episode_seed(episode_id):
        digest = hashlib.sha256(episode_id.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "little", signed=False) & 0x7FFFFFFF

    def _phase_for_episode(self, episode_id: str, epoch: int) -> int:
        phases = list(self.dataset._episode_phases[episode_id])
        if not phases:
            raise RuntimeError(f"Episode {episode_id!r} has no valid phase")
        if self.phase_mode == "fixed_zero":
            return 0 if 0 in phases else phases[0]
        # A deterministic permutation per episode.  ``epoch`` selects one
        # phase from that permutation, so visits are balanced over time.
        rng = np.random.default_rng(
            self.phase_seed + self._stable_episode_seed(episode_id)
        )
        permutation = list(phases)
        rng.shuffle(permutation)
        return int(permutation[int(epoch) % len(permutation)])

    def _episode_order(self, epoch: int) -> list[str]:
        rng = np.random.default_rng(self.seed + int(epoch))
        episode_ids = list(self.dataset.episode_ids)
        rng.shuffle(episode_ids)
        return episode_ids

    def _view_schedule(self, episode_id: str, epoch: int) -> _RememViewSchedule:
        phase = self._phase_for_episode(episode_id, epoch)
        episode = self.dataset._episode_by_id[episode_id]
        view_id = self.dataset._view_key(episode_id, phase)
        decision_count = 1 + (episode.index_length - 1 - phase) // self.dataset.window_stride
        return _RememViewSchedule(
            episode=episode,
            phase=phase,
            view_id=view_id,
            decision_count=decision_count,
            sequence_count=(decision_count + self.tbptt_steps - 1) // self.tbptt_steps,
        )

    def _sequence_descriptor(
        self, view: _RememViewSchedule, position: int, slot_id: int
    ) -> RememScheduledWindow:
        logical_start = int(position) * self.tbptt_steps
        valid = min(self.tbptt_steps, view.decision_count - logical_start)
        physical_start = view.phase + logical_start * self.dataset.window_stride
        return RememScheduledWindow(
            # A real lazy global index is retained for compatibility, though
            # dataset loading uses the explicit episode/physical fields.
            window_index=view.episode.global_offset + physical_start,
            slot_id=int(slot_id),
            is_first=position == 0,
            is_last=position + 1 >= view.sequence_count,
            valid_length=valid,
            episode_id=view.episode.episode_id,
            view_id=view.view_id,
            phase=view.phase,
            logical_start=logical_start,
            physical_start=physical_start,
            dataset_index=view.episode.dataset_index,
            trajectory_id=view.episode.trajectory_id,
            sequence_length=self.tbptt_steps,
        )

    def _scheduled_views(self, epoch):
        views = [self._view_schedule(ep, epoch) for ep in self.dataset.episode_ids]
        if self.balance_task_step_counts:
            groups = {}
            for view in views:
                groups.setdefault(view.episode.dataset_index, []).append(view)
            target = max(sum(v.decision_count for v in group) for group in groups.values())
            views = []
            for group in groups.values():
                count = 0
                index = 0
                while count < target:
                    view = group[index % len(group)]
                    views.append(view)
                    count += view.decision_count
                    index += 1
        if self.shuffle_trajectories:
            np.random.default_rng(self.seed + int(epoch)).shuffle(views)
        return views

    def _count_ticks(self, epoch: int) -> int:
        epoch = int(epoch)
        cached = self._tick_count_cache.get(epoch)
        if cached is not None:
            return cached
        queue = deque(view.sequence_count for view in self._scheduled_views(epoch))
        remaining = [0 for _ in range(self.global_batch_size)]
        ticks = 0
        while queue or any(value > 0 for value in remaining):
            active = False
            for slot in range(self.global_batch_size):
                if remaining[slot] <= 0 and queue:
                    remaining[slot] = queue.popleft()
                if remaining[slot] > 0:
                    remaining[slot] -= 1
                    active = True
            if active:
                ticks += 1
        self._tick_count_cache[epoch] = ticks
        return ticks

    def _iter_batches(self, epoch: int, *, start_tick: int = 0):
        self.dataset.set_epoch(epoch)
        queue = deque(self._scheduled_views(epoch))
        active: list[tuple[_RememViewSchedule, int] | None] = [
            None for _ in range(self.global_batch_size)
        ]
        tick = 0
        while queue or any(item is not None for item in active):
            rows: list[RememScheduledWindow] = []
            for global_slot, current in enumerate(active):
                if current is None and queue:
                    view = queue.popleft()
                    active[global_slot] = (view, 0)
                    current = active[global_slot]
                if current is None:
                    template = self.dataset._episodes[0]
                    rows.append(
                        RememScheduledWindow(
                            window_index=0,
                            slot_id=global_slot,
                            is_first=True,
                            is_last=True,
                            valid_length=0,
                            episode_id=f"__dummy__:{epoch}:{tick}:{global_slot}",
                            dummy=True,
                            window_indices=tuple([0] * self.tbptt_steps),
                            view_id=f"__dummy_view__:{epoch}:{tick}:{global_slot}",
                            phase=0,
                            logical_start=0,
                            physical_start=0,
                            dataset_index=template.dataset_index,
                            trajectory_id=template.trajectory_id,
                            sequence_length=self.tbptt_steps,
                        )
                    )
                    continue
                view, position = current
                descriptor = self._sequence_descriptor(view, position, global_slot)
                active[global_slot] = (
                    (view, position + 1)
                    if position + 1 < view.sequence_count
                    else None
                )
                rows.append(descriptor)
            if tick >= start_tick and not all(row.dummy for row in rows):
                if self.split_batches:
                    yield rows
                else:
                    for rank in range(self.world_size):
                        start = rank * self.local_batch_size
                        end = start + self.local_batch_size
                        yield rows[start:end]
            tick += 1

    def __iter__(self):
        if self.continuous_refill:
            yield from self._iter_refilled_stream()
            return
        start = self.rank * self.local_batch_size
        for rows in self._iter_batches(self.epoch):
            yield rows[start:start + self.local_batch_size]

    def __len__(self):
        if self.continuous_refill:
            decisions = sum(view.decision_count for view in self._scheduled_views(self.epoch))
            capacity = self.global_batch_size * self.tbptt_steps
            return (decisions + capacity - 1) // capacity
        return self._count_ticks(self.epoch)

    def _iter_refilled_stream(self):
        """Infinite deterministic stream; epoch counters never flush active slots."""
        queue = deque()
        active = [None] * self.global_batch_size
        remaining = {}
        enqueue_epoch = completed_epochs = self.epoch
        visit_id = 0
        window_index = 0
        start = self.rank * self.local_batch_size
        while True:
            by_slot = [[] for _ in active]
            for _ in range(self.tbptt_steps):
                for slot, current in enumerate(active):
                    if current is None:
                        if not queue:
                            views = self._scheduled_views(enqueue_epoch)
                            if not views:
                                raise RuntimeError("Cannot refill an empty episode schedule")
                            remaining[enqueue_epoch] = sum(view.decision_count for view in views)
                            queue.extend((view, enqueue_epoch) for view in views)
                            enqueue_epoch += 1
                        view, source_epoch = queue.popleft()
                        current = (view, 0, source_epoch, visit_id)
                        visit_id += 1
                    view, position, source_epoch, current_visit_id = current
                    physical = view.phase + position * self.dataset.window_stride
                    ended = position + 1 == view.decision_count
                    by_slot[slot].append(RememScheduledWindow(
                        window_index=view.episode.global_offset + physical,
                        slot_id=slot,
                        is_first=position == 0,
                        is_last=ended,
                        valid_length=1,
                        episode_id=view.episode.episode_id,
                        view_id=view.view_id,
                        phase=view.phase,
                        logical_start=position,
                        physical_start=physical,
                        dataset_index=view.episode.dataset_index,
                        trajectory_id=view.episode.trajectory_id,
                        sequence_length=1,
                        source_epoch=source_epoch,
                        stream_visit_id=current_visit_id,
                    ))
                    active[slot] = None if ended else (
                        view, position + 1, source_epoch, current_visit_id
                    )
                    remaining[source_epoch] -= 1
                while remaining.get(completed_epochs) == 0:
                    del remaining[completed_epochs]
                    completed_epochs += 1
            yield [
                RememSlotWindow(
                    slot_id=slot,
                    step_descriptors=tuple(by_slot[slot]),
                    completed_epochs=completed_epochs,
                    stream_window_index=window_index,
                )
                for slot in range(start, start + self.local_batch_size)
            ]
            window_index += 1
