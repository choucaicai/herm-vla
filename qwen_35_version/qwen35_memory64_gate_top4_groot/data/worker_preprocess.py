"""Use the current Qwen3.5 data/processor pipeline and record its worker identity."""

import os

from torch.utils.data import get_worker_info

from eventvla.dataloader import build_dataloader
from eventvla.dataloader.qwen_worker_preprocess import (
    configure_qwen_worker_preprocessing,
    preprocess_batch,
)
from .frame_tokens import build_frame_specs


def collate_worker_batch(examples):
    worker = get_worker_info()
    if worker is None:
        raise RuntimeError("Qwen preprocessing requires a DataLoader worker")
    batch = preprocess_batch(examples)
    if not isinstance(batch, dict) or not batch.get("_eventvla_worker_preprocessed"):
        raise RuntimeError("The worker must return processed Qwen inputs")
    batch["_raw_qwen35_worker_pid"] = os.getpid()
    batch["_raw_qwen35_worker_id"] = worker.id
    batch["moba_frame_specs"] = build_frame_specs(
        batch["examples"], batch["qwen_inputs"]["input_ids"], batch["qwen_inputs"]["image_grid_thw"],
        # Preserve every role until the model applies its checkpoint write policy.
        deduplicate=False,
    )
    return batch


def build_worker_dataloader(config):
    loader = build_dataloader(config, dataset_py=config.datasets.vla_data.dataset_py)
    loader.collate_fn = collate_worker_batch
    return loader
