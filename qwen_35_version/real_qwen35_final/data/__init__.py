"""Dedicated episode-window dataloader for S-only Qwen3.5 recurrent history."""
from __future__ import annotations

from pathlib import Path

import torch.distributed as dist
from torch.utils.data import DataLoader

from .episode_windows import (
    RememEpisodeWindowDataset,
    RememSlotBatchSampler,
)
from .temporal_profile import (
    ABSOLUTE_INDICES,
    DELTA_INDICES,
    apply_recurrent_temporal_profile,
    build_recurrent_dataset,
)


def collate_qwen_worker_batch(examples):
    # Evaluation imports frame helpers without loading the training data stack.
    from .worker_preprocess import collate_worker_batch

    return collate_worker_batch(examples)


def _worker_preprocessed(batch):
    return (
        isinstance(batch, dict)
        and bool(batch.get("_eventvla_worker_preprocessed", False))
    )


def _strip_worker_images(example):
    # qwen_worker_preprocess already converted the images into pixel tensors.
    # Keep metadata/action fields for the recurrent trainer while avoiding a
    # second IPC transfer of the PIL images.
    result = dict(example)
    result.pop("image", None)
    result.pop("memory_keyframe_images", None)
    return result


class RememSequenceCollator:
    """Keep fixed B*K shape and preprocess each valid TBPTT step in the worker."""
    def __call__(self, sequences):
        if not sequences:
            raise ValueError("Cannot collate an empty window batch")
        k = len(sequences[0]["steps"])
        if k < 1 or any(len(item["steps"]) != k for item in sequences):
            raise ValueError("All episode windows must have the same K")

        flat = []
        image_counts = []
        worker_time_batches = []
        time_examples = []
        any_valid = False
        for t in range(k):
            time_batch = []
            valid_batch = []
            for sequence in sequences:
                step = dict(sequence["steps"][t])
                if sequence.get("step_metadata") is not None:
                    metadata = sequence["step_metadata"][t]
                    valid = bool(metadata["step_valid"]) and not metadata["dummy"]
                    step.update(metadata)
                else:
                    valid = not sequence["dummy"] and t < sequence["valid_length"]
                    first = bool(sequence["reset"]) and t == 0 and valid
                    last = valid and sequence["is_last"] and t == sequence["valid_length"] - 1
                    step.update(
                        episode_id=sequence["episode_id"],
                        view_id=sequence["view_id"],
                        trajectory_id=sequence["trajectory_id"],
                        phase=sequence["phase"],
                        timestep=sequence["physical_indices"][t],
                        logical_step=sequence["logical_indices"][t],
                        anchor_index=sequence["logical_indices"][t],
                        slot_id=sequence["slot_id"],
                        chunk_index=sequence["chunk_index"],
                        is_new_episode=first,
                        reset=first,
                        is_last=last,
                        done=last,
                        step_valid=valid,
                        dummy=sequence["dummy"],
                        remem_action_valid_mask=sequence["action_valid_masks"][t],
                        action_valid_mask=sequence["action_valid_masks"][t],
                        source_epoch=sequence.get("source_epoch"),
                        stream_visit_id=sequence.get("stream_visit_id"),
                    )
                step["progress_aux_weight"] = 0.0
                time_batch.append(step)
                image_counts.append(len(step.get("image", [])))
                if valid:
                    valid_batch.append(step)
            flat.extend(time_batch)
            time_examples.append(time_batch)
            if valid_batch:
                worker_time_batches.append(collate_qwen_worker_batch(valid_batch))
                any_valid = True
            else:
                worker_time_batches.append(None)

        # ZeRO-2 can present a rank containing only padding. It still needs a
        # normal model graph, so prepare one all-slot fallback batch in the
        # worker when no real step exists on this rank.
        worker_fallback_batch = None
        if not any_valid:
            worker_fallback_batch = collate_qwen_worker_batch(time_examples[0])

        worker_preprocessed = any(
            _worker_preprocessed(item)
            for item in (*worker_time_batches, worker_fallback_batch)
        )
        if worker_preprocessed:
            flat = [_strip_worker_images(step) for step in flat]
            packed_sequences = []
            for sequence in sequences:
                packed = dict(sequence)
                packed["steps"] = [
                    _strip_worker_images(step) for step in sequence["steps"]
                ]
                packed_sequences.append(packed)
        else:
            packed_sequences = sequences

        return PackedEpisodeBatch(
            flat,
            packed_sequences,
            worker_time_batches=worker_time_batches,
            worker_fallback_batch=worker_fallback_batch,
            image_counts=image_counts,
            worker_preprocessed=worker_preprocessed,
        )


class PackedEpisodeBatch:
    """Time-major B*K examples with worker-prepared Qwen batches.

    This remains list-like for the recurrent trainer. It intentionally does not
    inherit from ``list`` so PyTorch's pin-memory worker calls ``pin_memory``
    below and traverses the worker-produced Qwen tensors as well.
    """
    def __init__(
        self,
        examples,
        sequences,
        *,
        worker_time_batches=None,
        worker_fallback_batch=None,
        image_counts=None,
        worker_preprocessed=False,
    ):
        self._examples = list(examples)
        self.sequences = sequences
        self.batch_slots = len(sequences)
        self.valid_length = len(sequences[0]["steps"])
        self.valid_lengths = [int(x["valid_length"]) for x in sequences]
        self.sequence_shape = (self.batch_slots, self.valid_length)
        self.worker_time_batches = list(worker_time_batches or [])
        self.worker_fallback_batch = worker_fallback_batch
        self.image_counts = list(image_counts or [])
        self.worker_preprocessed = bool(worker_preprocessed)
        self.continuous_refill = all(
            sequence.get("continuous_refill", False) for sequence in sequences
        )
        stream_epochs = {sequence.get("stream_epoch") for sequence in sequences}
        if self.continuous_refill and len(stream_epochs) != 1:
            raise ValueError("Every slot must carry the same completed-epoch counter")
        self.stream_epoch = next(iter(stream_epochs)) if self.continuous_refill else None

    def __len__(self):
        return len(self._examples)

    def __iter__(self):
        return iter(self._examples)

    def __getitem__(self, index):
        return self._examples[index]

    def pin_memory(self, device=None):
        from torch.utils.data._utils.pin_memory import pin_memory
        return PackedEpisodeBatch(
            pin_memory(self._examples, device),
            pin_memory(self.sequences, device),
            worker_time_batches=pin_memory(self.worker_time_batches, device),
            worker_fallback_batch=pin_memory(self.worker_fallback_batch, device),
            image_counts=self.image_counts,
            worker_preprocessed=self.worker_preprocessed,
        )

    def time_slice(self, time_index):
        if not 0 <= time_index < self.valid_length:
            raise IndexError(time_index)
        start = time_index * self.batch_slots
        return list(self._examples[start:start + self.batch_slots])

    def worker_time_batch(self, time_index):
        if not 0 <= time_index < self.valid_length:
            raise IndexError(time_index)
        if not self.worker_time_batches:
            return None
        return self.worker_time_batches[time_index]

    def fallback_batch(self):
        return (
            self.worker_fallback_batch
            if self.worker_fallback_batch is not None
            else self.time_slice(0)
        )

    @property
    def steps(self):
        return [self.time_slice(index) for index in range(self.valid_length)]


__all__ = [
    "ABSOLUTE_INDICES",
    "DELTA_INDICES",
    "PackedEpisodeBatch",
    "RememEpisodeWindowDataset",
    "RememSequenceCollator",
    "RememSlotBatchSampler",
    "apply_recurrent_temporal_profile",
    "build_recurrent_dataset",
    "build_dataloader",
]


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, str):
        return value.lower().strip() not in {"false", "0", "", "off", "no", "none"}
    return bool(value)


def build_dataloader(cfg, dataset_py="lerobot_datasets"):
    """Build a recurrent stream using the configured temporal profile."""
    if dataset_py != "lerobot_datasets":
        raise ValueError("Qwen35 recurrent windows require lerobot_datasets")
    apply_recurrent_temporal_profile(cfg)
    data = cfg.datasets.vla_data
    if not _as_bool(data.get("pack_tbptt_windows", True), True):
        raise ValueError("Qwen35 recurrent windows require pack_tbptt_windows=true")

    mixture = build_recurrent_dataset(cfg)
    dataset = RememEpisodeWindowDataset(
        mixture,
        window_size=1,
        decision_stride=int(data.get("sampling_interval", 50)),
        drop_last=False,
        reset_each_window=False,
        batch_tbptt_video=_as_bool(data.get("batch_tbptt_video", True), True),
    )
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    sampler = RememSlotBatchSampler(
        dataset,
        batch_size=int(data.per_device_batch_size),
        tbptt_steps=int(cfg.trainer.get("tbptt_steps", 1)),
        rank=rank,
        world_size=world_size,
        seed=int(cfg.seed),
        phase_mode=str(data.get("episode_phase_mode", "balanced_random")),
        balance_task_step_counts=_as_bool(data.get("balance_task_step_counts", True), True),
        shuffle_trajectories=_as_bool(data.get("shuffle_trajectories", True), True),
        continuous_refill=_as_bool(data.get("refill_slots_per_step", True), True),
    )
    workers = int(data.get("num_workers", 2))
    loader_kwargs = {"num_workers": workers, "pin_memory": True}
    if workers:
        loader_kwargs["prefetch_factor"] = int(data.get("prefetch_factor", 1))
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=RememSequenceCollator(),
        **loader_kwargs,
    )
    if rank == 0:
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        mixture.save_dataset_statistics(output_dir / "dataset_statistics.json")
    return loader
