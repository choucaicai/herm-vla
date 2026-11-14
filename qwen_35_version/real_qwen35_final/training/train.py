"""Train EventVLA-structured Qwen3.5 on a rank-aware episode stream."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
import math
import os
from pathlib import Path
import shutil
import socket
import time

# Triton metadata must not race across ranks on the shared project filesystem.
cache_base = os.environ.get("REAL_QWEN35_FINAL_CACHE_BASE")
if cache_base:
    rank_cache = Path(cache_base) / f"rank_{os.environ.get('RANK', '0')}"
    os.environ["TRITON_CACHE_DIR"] = str(rank_cache / "triton")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(rank_cache / "inductor")

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import GradientAccumulationPlugin, InitProcessGroupKwargs, set_seed
from omegaconf import OmegaConf
from transformers import get_scheduler



PROFILE_TIMING = os.environ.get("EVENTVLA_PROFILE_TIMING", "0").strip().lower() in {"1", "true", "yes", "on"}
PROFILE_SAVE = os.environ.get("EVENTVLA_PROFILE_SAVE", "0").strip().lower() in {"1", "true", "yes", "on"}
SINGLE_NODE_INIT = os.environ.get("EVENTVLA_SINGLE_NODE_INIT", "1").strip().lower() in {"1", "true", "yes", "on"}


def _preinitialize_torch_distributed():
    """Initialize the local torch process group before Accelerate/DeepSpeed.

    DeepSpeed 0.16.x can enter its MPI/network discovery path when it owns
    initialization. A torchrun/Accelerate single-node process group gives it
    an already-established NCCL group and keeps initialization on the local
    host (env://).
    """
    if not SINGLE_NODE_INIT or dist.is_initialized():
        return
    if "LOCAL_RANK" not in os.environ or "RANK" not in os.environ:
        return
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    backend = os.environ.get("EVENTVLA_DIST_BACKEND", "nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # This host's elastic agent store can stall in TCPStore::ping. Use a
    # separate classic TCPStore for the worker process group, independently
    # of the launcher's rendezvous store.
    store_port = int(os.environ.get("REAL_PROCESS_GROUP_PORT", int(os.environ["MASTER_PORT"]) + 1))
    if rank == 0:
        print(f"Initializing {world_size}-rank process group on port {store_port}", flush=True)
    store = dist.TCPStore(
        os.environ.get("MASTER_ADDR", "127.0.0.1"), store_port, world_size,
        rank == 0, timeout=timedelta(minutes=5), use_libuv=False,
    )
    dist.init_process_group(
        backend=backend,
        store=store, rank=rank, world_size=world_size,
        timeout=timedelta(minutes=30),
    )
    # Bind DeepSpeed to the already-created local torch group. Without this,
    # DeepSpeed 0.16.x calls its own discovery path again inside Accelerate.
    import deepspeed.comm as ds_comm
    ds_comm.init_distributed(
        dist_backend=backend,
        auto_mpi_discovery=False,
        dist_init_required=False,
    )
    if rank == 0:
        print("Worker process group initialized", flush=True)


# Project loggers construct PartialState during import. Establish the process
# group first, so DeepSpeed cannot initialize a competing communication state.
if __name__ == "__main__":
    _preinitialize_torch_distributed()

from qwen_35_version.real_qwen35_final.data.worker_preprocess import (
    configure_qwen_worker_preprocessing,
)
from qwen_35_version.real_qwen35_final.data import (
    PackedEpisodeBatch, apply_recurrent_temporal_profile, build_dataloader,
)
from qwen_35_version.real_qwen35_final.data.frame_tokens import temporal_image_count
from qwen_35_version.real_qwen35_final.data.temporal_profile import temporal_indices
from eventvla.training.trainer_utils.trainer_tools import (
    build_param_lr_groups,
    normalize_dotlist_args,
)

from qwen_35_version.real_qwen35_final.model.eventvla_qwen35_mix import (
    RealQwen35Final, validate_config, validate_temporal_batch,
)
from qwen_35_version.real_qwen35_final.training.kernel_warmup import warm_delta_kernels


def _cuda_sync():
    """Synchronize only in optional profiling mode to make stage times real wall time."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _get_model_profile_timing(model, accelerator):
    if not PROFILE_TIMING:
        return {}
    try:
        unwrapped = accelerator.unwrap_model(model)
        interface = getattr(unwrapped, "qwen_vl_interface", None)
        return dict(getattr(interface, "_profile_timing", {}) or {})
    except Exception:
        return {}


def _reduce_profile_timing(values, accelerator):
    """Return mean/max timings; every rank enters collectives in the same order."""
    if not values:
        return {}
    if not dist.is_initialized():
        return {key: {"mean": float(value), "max": float(value)} for key, value in values.items()}
    world_size = dist.get_world_size()
    reduced = {}
    for key in sorted(values):
        local = torch.tensor(float(values[key]), dtype=torch.float64, device=accelerator.device)
        total = local.clone()
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        maximum = local.clone()
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        reduced[key] = {"mean": total.item() / world_size, "max": maximum.item()}
    return reduced


def save_weights(accelerator, model, cfg, step, final=False):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        root = Path(cfg.output_dir)
        target = (
            root / "final_model/pytorch_model.pt"
            if final
            else root / f"checkpoints/steps_{step}_pytorch_model.pt"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        torch.save(accelerator.get_state_dict(model), temporary)
        os.replace(temporary, target)
        with (root / "summary.jsonl").open("a") as handle:
            handle.write(json.dumps({"steps": step, "checkpoint": str(target), "final": final}) + "\n")
        if not final:
            checkpoints = sorted(
                (root / "checkpoints").glob("steps_*_pytorch_model.pt"),
                key=lambda path: int(path.name.split("_")[1]),
            )
            keep = int(cfg.trainer.keep_recent_checkpoints)
            for old in checkpoints[:-keep] if keep > 0 else checkpoints:
                old.unlink()
        print(f"Checkpoint saved: {target}", flush=True)
    accelerator.wait_for_everyone()


def _load_resume_checkpoint(model, checkpoint_path: str) -> None:
    checkpoint = Path(checkpoint_path).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"Resume checkpoint must contain a state dict: {checkpoint}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Resume checkpoint does not match RealQwen35Final model: "
            f"missing={list(missing)[:8]}, unexpected={list(unexpected)[:8]}"
        )
    print(f"Loaded resume checkpoint: {checkpoint}", flush=True)


def _scalar_metrics(output, accelerator):
    metrics = {}
    for key, value in output.items():
        if isinstance(value, torch.Tensor) and value.ndim == 0 and key != "total_loss":
            metrics[key] = accelerator.reduce(value.detach().float(), reduction="mean").item()
    return metrics


from qwen_35_version.real_qwen35_final.training.state import (
    detach_slot_state,
    prepare_recurrent_step, update_recurrent_slots,
)

def _example_slot_id(example):
    value = example.get("slot_id", 0)
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def _valid_step_examples(batch, time_index):
    return [
        example
        for example in batch.time_slice(time_index)
        if bool(example.get("step_valid", True))
        and not bool(example.get("dummy", False))
    ]


def _numerical_failure_context(policy, batch, states, tbptt_steps):
    return {
        "examples": [
            [
                {
                    key: example.get(key)
                    for key in ("slot_id", "episode_id", "view_id", "timestep", "reset", "done", "is_last")
                }
                for example in _valid_step_examples(batch, index)
            ]
            for index in range(tbptt_steps)
        ],
        "nonfinite_parameters": [
            name for name, parameter in policy.named_parameters()
            if not bool(torch.isfinite(parameter.detach()).all())
        ],
        "nonfinite_memory_slots": [str(slot) for slot, state in states.items()
                                   if state is not None and not bool(torch.isfinite(state.detach()).all())],
    }


def _valid_step_batch(batch, time_index):
    """Return the worker-prepared batch aligned with valid recurrent slots."""
    raw_examples = _valid_step_examples(batch, time_index)
    if not raw_examples:
        return None
    prepared = batch.worker_time_batch(time_index)
    if prepared is None:
        return raw_examples
    if isinstance(prepared, dict) and prepared.get("_eventvla_worker_preprocessed", False):
        examples = prepared.get("examples", [])
        if len(examples) != len(raw_examples):
            raise RuntimeError(
                f"Worker-preprocessed recurrent batch has {len(examples)} examples; "
                f"expected {len(raw_examples)}"
            )
    return prepared


def _batch_examples(step_batch):
    if isinstance(step_batch, dict) and step_batch.get("_eventvla_worker_preprocessed", False):
        return step_batch["examples"]
    return step_batch


def main(cfg):
    apply_recurrent_temporal_profile(cfg)
    validate_config(cfg)
    absolute_indices, delta_indices = temporal_indices(cfg)
    images_per_sample = temporal_image_count(cfg.datasets.vla_data)
    _preinitialize_torch_distributed()
    accelerator = Accelerator(
        deepspeed_plugin=DeepSpeedPlugin(),
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=1, sync_each_batch=True, sync_with_dataloader=False
        ),
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(minutes=30))],
    )
    plugin = accelerator.state.deepspeed_plugin
    zero_stage = int(plugin.deepspeed_config["zero_optimization"]["stage"])
    if zero_stage not in (1, 2):
        raise ValueError("The two-forward training recipe supports DeepSpeed ZeRO-1 or ZeRO-2")
    set_seed(int(cfg.seed))
    cfg.output_dir = str(Path(cfg.run_root_dir) / cfg.run_id)
    root = Path(cfg.output_dir)
    batch_size = int(cfg.datasets.vla_data.per_device_batch_size)
    k = int(cfg.trainer.tbptt_steps)
    release_unused_cuda_cache_before_backward = bool(
        cfg.trainer.get("release_unused_cuda_cache_before_backward", False)
    )
    if accelerator.is_main_process:
        root.mkdir(parents=True, exist_ok=True)
        (root / "checkpoints").mkdir(exist_ok=True)
        OmegaConf.save(cfg, root / "config.yaml")
        source = root / "source/real_qwen35_final"
        if source.exists():
            raise FileExistsError(f"Refusing to replace an existing run snapshot: {source}")
        shutil.copytree(
            Path(__file__).resolve().parents[1], source,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        runtime = {
            "started_at": datetime.now().astimezone().isoformat(),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "world_size": accelerator.num_processes,
            "deepspeed_zero_stage": zero_stage,
            "deepspeed_zero_config": plugin.deepspeed_config["zero_optimization"],
            "per_device_batch_size": batch_size,
            "batch_slots_per_update": batch_size * accelerator.num_processes,
            "max_decisions_per_update": batch_size * accelerator.num_processes * k,
            "gradient_accumulation_steps": 1,
            "step_unit": "optimizer_update",
            "model": "Qwen3.5-4B + 64 zero-state Gate memory queries + 7-D Piper MLP action head",
            "package": "qwen_35_version.real_qwen35_final",
            "base_release_package": "raw_qwen35_mix_current_kv_action_memory64_all_tokens_zero_state_gate_image_topk",
            "dataset_path": str(cfg.datasets.vla_data.dataset_path),
            "dataset_format": "LeRobot v3.0",
            "camera_order": ["cam_high", "cam_follower_wrist"],
            "action_dim": 7,
            "action_mode": "abs",
            "gripper_mode": "continuous",
            "normalization": "q01/q99 for all seven recorded positions",
            "input": (
                f"two cameras, absolute={list(absolute_indices)}, "
                f"delta={list(delta_indices)}; {images_per_sample} images"
            ),
            "images_per_sample": images_per_sample,
            "absolute_indices": list(absolute_indices),
            "delta_indices": list(delta_indices),
            "sampling_interval": 50,
            "tbptt_steps": k,
            "recurrent_state_mode": "memory_queries",
            "conv_state_policy": "zero_at_every_forward",
            "native_s_policy": "reset_every_decision",
            "cross_decision_s_layers": 0,
            "memory_queries": 64,
            "query_memory_config": OmegaConf.to_container(cfg.framework.memory_queries, resolve=True),
            "memory_initial_state": "zeros",
            "mask_empty_read": True,
            "first_memory_write": "complete",
            "full_attention_kv_policy": "detached_current_visual_images_all_tokens_top4",
            "history_query_scope": "all_tokens",
            "hf_full_attention_kv_policy": "empty_at_every_forward",
            "moba_memory": OmegaConf.to_container(cfg.framework.moba_memory, resolve=True),
            "moba_commit_policy": "after_each_decision_before_next_tbptt_forward",
            "rope_policy": "continuous_per_episode_slot",
            "state_refresh": "none",
            "last_hidden_only": True,
            "padding_policy": "masked_full_batch",
            "padding_state_update": "qkv_zero_g_zero_native_beta",
            "attn_implementation": cfg.framework.qwenvl.attn_implementation,
            "delta_kernel_warmup": True,
            "release_unused_cuda_cache_before_backward": release_unused_cuda_cache_before_backward,
            "action_loss": "L1 normalized by globally valid action elements",
            "processor_in_dataloader_workers": True,
            "batch_window_vision": bool(cfg.trainer.get("batch_window_vision", False)),
            "refill_slots_per_step": bool(cfg.datasets.vla_data.get("refill_slots_per_step", True)),
            "epoch_state_policy": "preserve_active_slots_in_continuous_stream",
            "gradient_checkpointing": False,
            "torch_version": torch.__version__,
            "compiler_cache": {
                "triton": os.environ.get("TRITON_CACHE_DIR"),
                "inductor": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
            },
        }
        (root / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    accelerator.wait_for_everyone()
    accelerator.print(f"Loading Qwen3.5: {cfg.framework.qwenvl.base_vlm}")
    model = RealQwen35Final(cfg)
    resume_checkpoint = cfg.trainer.get("resume_checkpoint")
    init_checkpoint = cfg.trainer.get("init_checkpoint")
    if init_checkpoint and resume_checkpoint:
        raise ValueError("init_checkpoint and resume_checkpoint are mutually exclusive")
    initialization = {"source": "Qwen3.5-4B base model"}
    if init_checkpoint:
        from .initialize import initialize_from_final
        initialization = initialize_from_final(model, str(init_checkpoint))
    resume_step = int(cfg.trainer.get("resume_step", 0))
    if resume_checkpoint:
        if resume_step <= 0:
            raise ValueError("A resumed run requires trainer.resume_step > 0")
        _load_resume_checkpoint(model, str(resume_checkpoint))
    elif resume_step:
        raise ValueError("trainer.resume_checkpoint is required")
    model.train()
    if accelerator.is_main_process:
        runtime["action_query"] = model.action_query_info
        runtime["linear_layers"] = model._layer_types.count("linear_attention")
        runtime["full_attention_layers"] = model._layer_types.count("full_attention")
        runtime["resume_step"] = resume_step
        runtime["initialization"] = initialization
        runtime["resolved_attention_backends"] = {
            "text": model.qwen_vl_interface.model.config.text_config._attn_implementation,
            "vision": model.qwen_vl_interface.model.config.vision_config._attn_implementation,
        }
        (root / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    injection = cfg.framework.memory_buffer.qwen_memory_injection
    configure_qwen_worker_preprocessing(
        enabled=True,
        model_id=cfg.framework.qwenvl.base_vlm,
        action_token=model.action_token,
        chunk_len=model.chunk_len,
        use_image_role_text=bool(injection.get("use_image_role_text", False)),
        cot_prompt=cfg.datasets.vla_data.get("CoT_prompt"),
    )
    loader = build_dataloader(cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
    accelerator.dataloader_config.dispatch_batches = False
    optimizer = torch.optim.AdamW(
        build_param_lr_groups(model, cfg),
        lr=float(cfg.trainer.learning_rate.base),
        betas=tuple(cfg.trainer.optimizer.betas),
        eps=float(cfg.trainer.optimizer.eps),
        weight_decay=float(cfg.trainer.optimizer.weight_decay),
    )
    scheduler = get_scheduler(
        cfg.trainer.lr_scheduler_type, optimizer,
        num_warmup_steps=int(cfg.trainer.num_warmup_steps),
        num_training_steps=int(cfg.trainer.max_train_steps),
        scheduler_specific_kwargs=OmegaConf.to_container(
            cfg.trainer.scheduler_specific_kwargs, resolve=True
        ),
    )
    plugin.deepspeed_config.update(
        train_micro_batch_size_per_gpu=batch_size,
        train_batch_size=batch_size * accelerator.num_processes,
        gradient_accumulation_steps=1,
        gradient_clipping=float(cfg.trainer.gradient_clipping),
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    if model.gradient_accumulation_steps() != 1:
        raise RuntimeError("DeepSpeed accumulation must match the two-step window")
    if accelerator.is_main_process:
        runtime["gradient_accumulation_dtype"] = str(model.get_data_types()[1])
        runtime["resolved_train_batch_size"] = model.train_batch_size()
        (root / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    if resume_checkpoint:
        scheduler.step(resume_step)
    policy = accelerator.unwrap_model(model)
    accelerator.print("Warming DeltaNet forward/backward kernels before TBPTT graphs")
    warmup = warm_delta_kernels(
        policy.qwen_vl_interface.model.config.get_text_config(), accelerator.device
    )
    (root / f"kernel_warmup_rank{accelerator.process_index}.json").write_text(
        json.dumps(warmup, indent=2) + "\n"
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        OmegaConf.save(cfg, root / "config.yaml")
    set_seed(int(cfg.seed) + accelerator.process_index)
    epoch = 0
    iterator = iter(loader)
    state_by_slot = {}
    episode_by_slot = {}
    timestep_by_slot = {}

    def next_batch():
        nonlocal epoch, iterator
        try:
            batch = next(iterator)
        except StopIteration:
            if bool(cfg.datasets.vla_data.get("refill_slots_per_step", True)):
                raise RuntimeError("The continuous slot stream unexpectedly ended")
            epoch += 1
            state_by_slot.clear()
            episode_by_slot.clear()
            timestep_by_slot.clear()
            policy.reset_memory(batch_size=batch_size)
            loader.batch_sampler.set_epoch(epoch)
            iterator = iter(loader)
            batch = next(iterator)
        if batch.continuous_refill:
            epoch = int(batch.stream_epoch)
        return batch

    first_batch = next_batch()
    if not isinstance(first_batch, PackedEpisodeBatch):
        raise TypeError("The loader must return a packed episode window")
    if first_batch.sequence_shape != (batch_size, k):
        raise ValueError(f"Unexpected sequence shape: {first_batch.sequence_shape}")
    if not first_batch.worker_preprocessed:
        raise RuntimeError("Qwen preprocessing must run inside DataLoader workers")
    if any(count != images_per_sample for count in first_batch.image_counts):
        raise ValueError(f"Unexpected image counts: {first_batch.image_counts}")
    validate_temporal_batch(list(first_batch), data_config=cfg.datasets.vla_data)
    prepared = first_batch.worker_time_batch(0) or first_batch.fallback_batch()
    if prepared["_raw_qwen35_worker_pid"] == os.getpid():
        raise RuntimeError("Qwen preprocessing ran in the training process")
    query_counts = (prepared["qwen_inputs"]["input_ids"] == policy.action_token_id).sum(dim=1)
    if not bool((query_counts == 50).all()):
        raise ValueError(f"Invalid action query counts: {query_counts.tolist()}")
    for ids in prepared["qwen_inputs"]["input_ids"]:
        positions = (ids == policy.action_token_id).nonzero().flatten()
        if not bool(((positions[1:] - positions[:-1]) == 1).all()):
            raise ValueError("Diamond queries must remain adjacent")
    if accelerator.is_main_process:
        validation = {
            "sequence_shape": list(first_batch.sequence_shape),
            "action_shape": list(np.stack([item["action"] for item in first_batch]).shape),
            "images_per_sample": first_batch.image_counts,
            "absolute_indices": list(absolute_indices),
            "delta_indices": list(delta_indices),
            "image_metas": first_batch[0].get("image_metas"),
            "action_query": policy.action_query_info,
            "action_query_counts": query_counts.tolist(),
            "worker_pid": prepared["_raw_qwen35_worker_pid"],
            "training_pid": os.getpid(),
            "tensor_devices": {
                key: str(value.device) for key, value in prepared["qwen_inputs"].items()
                if isinstance(value, torch.Tensor)
            },
            "recurrent_state_mode": "memory_queries",
            "refill_slots_per_step": first_batch.continuous_refill,
            "step_metadata": [
                [
                    {key: item.get(key) for key in (
                        "slot_id", "episode_id", "view_id", "stream_visit_id",
                        "source_epoch", "timestep", "reset", "done",
                    )}
                    for item in first_batch.time_slice(index)
                ]
                for index in range(k)
            ],
        }
        (root / "input_validation.json").write_text(
            json.dumps(validation, indent=2, default=str) + "\n"
        )
        print("Validated inputs: " + json.dumps(validation, default=str), flush=True)
    max_steps = int(cfg.trainer.max_train_steps)
    accelerator.print(
        f"Training {max_steps} optimizer updates, world_size={accelerator.num_processes}, "
        f"slots={batch_size * accelerator.num_processes}, tbptt={k}, "
        f"max_decisions={batch_size * accelerator.num_processes * k}, stride=50, state=memory_queries64"
    )
    started = time.monotonic()
    last_log_step = resume_step
    last_log_time = started
    decisions_since_log = 0
    data_wait_since_log = 0.0
    max_data_wait_since_log = 0.0
    last_completed_step = resume_step
    stop_requested = False
    for step in range(resume_step + 1, max_steps + 1):
        update_index = step - resume_step
        tick = time.monotonic()
        data_tick = time.monotonic()
        batch = first_batch if first_batch is not None else next_batch()
        first_batch = None
        data_wait = time.monotonic() - data_tick
        data_wait_since_log += data_wait
        max_data_wait_since_log = max(max_data_wait_since_log, data_wait)
        engine_step_before = model.global_steps
        local_elements = torch.zeros((), device=accelerator.device)
        local_decisions = 0
        numerator = None
        trace = []
        input_state = output = next_state = slot_states = None
        moba_stats_by_time = []
        memory_stats_by_time = []
        with accelerator.accumulate(model):
            optimizer.zero_grad(set_to_none=True)
            step_batches = [_valid_step_batch(batch, index) for index in range(k)]
            if bool(cfg.trainer.get("batch_window_vision", False)):
                with accelerator.autocast():
                    step_batches = model(step_batches, prepare_window_vision=True)
                if update_index <= 3 and accelerator.is_main_process:
                    with (root / "window_vision_validation.jsonl").open("a") as handle:
                        handle.write(json.dumps({
                            "step": step, **policy._window_vision_report,
                        }) + "\n")
            for time_index in range(k):
                step_batch = step_batches[time_index]
                if step_batch is None:
                    continue
                examples = _batch_examples(step_batch)
                if batch.continuous_refill and len(examples) != batch_size:
                    raise RuntimeError("Continuous refill must keep every decoder time step full")
                slots = [_example_slot_id(example) for example in examples]
                input_state, warm, resets = prepare_recurrent_step(
                    examples, state_by_slot, episode_by_slot, timestep_by_slot
                )
                with accelerator.autocast():
                    output, next_state = model(
                        step_batch, recurrent_state=input_state, return_recurrent_state=True
                    )
                if not isinstance(next_state, torch.Tensor) or not next_state.requires_grad:
                    raise RuntimeError("Memory tokens must remain differentiable inside the window")
                moba_stats_by_time.append(dict(policy.moba_memory.last_stats))
                memory_stats_by_time.append(dict(policy.query_memory.last_stats))
                # Historical KV is detached; expose this decision to the next unrolled step.
                policy.commit_moba_memory()
                elements = output["valid_action_elements"].detach()
                weighted = output["total_loss"] * elements
                numerator = weighted if numerator is None else numerator + weighted
                local_elements += elements
                local_decisions += len(examples)
                slot_states = update_recurrent_slots(
                    examples, next_state, state_by_slot, timestep_by_slot
                )
                if update_index <= 20:
                    trace.append({
                        "time_index": time_index,
                        "slots": slots,
                        "timesteps": [int(example["timestep"]) for example in examples],
                        "warm_start": warm,
                        "reset": resets,
                        "done": [bool(example.get("done")) for example in examples],
                        "episode_ids": [example["episode_id"] for example in examples],
                        "visit_ids": [example.get("stream_visit_id") for example in examples],
                        "source_epochs": [example.get("source_epoch") for example in examples],
                        "memory_shape": list(next_state.shape),
                        "memory_requires_grad": True,
                        **policy._last_memory64_report,
                        "language_batch_size": len(examples),
                        "prompt_lengths": step_batch["qwen_inputs"]["attention_mask"].sum(dim=1).tolist(),
                        "valid_action_elements": float(elements.item()),
                        "moba": moba_stats_by_time[-1],
                        "moba_slots": policy.moba_memory.state_summary(),
                    })
            if numerator is None:
                with accelerator.autocast():
                    dummy, _ = model(
                        batch.fallback_batch(), recurrent_state=None, return_recurrent_state=True
                    )
                numerator = dummy["total_loss"] * 0.0
                policy.discard_moba_memory()
            global_elements = accelerator.reduce(local_elements, reduction="sum")
            if global_elements.item() <= 0:
                raise RuntimeError("A distributed update contains no valid action supervision")
            # ZeRO averages gradients across ranks; compensate for uneven tails.
            loss = numerator * accelerator.num_processes / global_elements
            finite = torch.isfinite(loss.detach()).to(torch.int32)
            if accelerator.reduce(finite, reduction="sum").item() != accelerator.num_processes:
                (root / f"loss_failure_rank{accelerator.process_index}.json").write_text(
                    json.dumps({
                        "step": step,
                        "local_loss_is_finite": bool(finite),
                        "local_loss": float(loss.detach()),
                        **_numerical_failure_context(policy, batch, state_by_slot, k),
                    }, indent=2, default=str) + "\n"
                )
                raise FloatingPointError(f"Non-finite loss at step {step}")
            if release_unused_cuda_cache_before_backward:
                torch.cuda.empty_cache()
            if update_index <= 3:
                memory = {
                    "step": step,
                    "stage": "before_backward",
                    "allocated_bytes": torch.cuda.memory_allocated(),
                    "reserved_bytes": torch.cuda.memory_reserved(),
                    "max_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "driver_free_bytes": torch.cuda.mem_get_info()[0],
                }
                with (root / f"memory_rank{accelerator.process_index}.jsonl").open("a") as handle:
                    handle.write(json.dumps(memory) + "\n")
            try:
                accelerator.backward(loss)
            except RuntimeError as error:
                policy.discard_moba_memory()
                failure = {
                    "step": step,
                    "error": str(error),
                    "allocated_bytes": torch.cuda.memory_allocated(),
                    "reserved_bytes": torch.cuda.memory_reserved(),
                    "max_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "driver_free_bytes": torch.cuda.mem_get_info()[0],
                }
                (root / f"backward_failure_rank{accelerator.process_index}.json").write_text(
                    json.dumps(failure, indent=2) + "\n"
                )
                raise
            optimizer.step()
        gradient_norm = model.get_global_grad_norm()
        if gradient_norm is not None:
            gradient_norm = float(gradient_norm)
            if not math.isfinite(gradient_norm) or gradient_norm < 0:
                (root / f"gradient_failure_rank{accelerator.process_index}.json").write_text(
                    json.dumps({
                        "step": step, "gradient_norm": gradient_norm,
                        "loss_before_backward": float(loss.detach()),
                        **_numerical_failure_context(policy, batch, state_by_slot, k),
                    }, indent=2, default=str) + "\n"
                )
                raise FloatingPointError(f"Invalid gradient norm at step {step}: {gradient_norm}")
        if model.global_steps != engine_step_before + 1 or not accelerator.sync_gradients:
            raise RuntimeError("Expected exactly one optimizer update per two-step window")
        if not accelerator.optimizer_step_was_skipped:
            scheduler.step()
        for slot, state in list(state_by_slot.items()):
            state_by_slot[slot] = detach_slot_state(state)
        decisions_since_log += local_decisions
        last_completed_step = step
        if update_index <= 5 or step % int(cfg.trainer.logging_frequency) == 0:
            average = accelerator.reduce(loss.detach().float(), reduction="mean").item()
            local_stop = int(
                accelerator.is_main_process and (root / "SAVE_AND_STOP").is_file()
            )
            counts = accelerator.reduce(
                torch.tensor(
                    [local_decisions, decisions_since_log, local_stop,
                     data_wait, data_wait_since_log],
                    device=accelerator.device,
                    dtype=torch.float64,
                ),
                reduction="sum",
            )
            decisions = counts[0].item()
            stop_requested = bool(counts[2].item())
            log_time = time.monotonic()
            interval_seconds = log_time - last_log_time
            interval_updates = step - last_log_step
            maximum_wait = torch.tensor(
                max_data_wait_since_log, device=accelerator.device, dtype=torch.float64
            )
            if dist.is_initialized():
                dist.all_reduce(maximum_wait, op=dist.ReduceOp.MAX)
            capacity = batch_size * accelerator.num_processes * k
            metrics = {
                "step": step,
                "run_update_index": update_index,
                "resume_step": resume_step,
                "deepspeed_global_steps": model.global_steps,
                "deepspeed_zero_stage": zero_stage,
                "total_loss": average,
                "action_loss": average,
                "gradient_norm": gradient_norm,
                "valid_decisions": decisions,
                "max_decisions_per_update": capacity,
                "batch_fill_fraction": decisions / capacity,
                "interval_valid_decisions": counts[1].item(),
                "interval_batch_fill_fraction": counts[1].item() / (capacity * interval_updates),
                "valid_action_elements": global_elements.item(),
                "action_valid_fraction": global_elements.item() / max(1, decisions * 50 * 7),
                "learning_rates": scheduler.get_last_lr(),
                "learning_rate_groups": {
                    group.get("name", str(index)): lr for index, (group, lr) in enumerate(
                        zip(scheduler.optimizer.param_groups, scheduler.get_last_lr())
                    )
                },
                "step_seconds": time.monotonic() - tick,
                "interval_updates": interval_updates,
                "interval_seconds": interval_seconds,
                "seconds_per_update": interval_seconds / interval_updates,
                "decisions_per_second": counts[1].item() / interval_seconds,
                "data_wait_seconds_mean": counts[3].item() / accelerator.num_processes,
                "interval_data_wait_seconds_per_update_mean":
                    counts[4].item() / (accelerator.num_processes * interval_updates),
                "interval_data_wait_seconds_max": maximum_wait.item(),
                "elapsed_seconds": time.monotonic() - started,
                "timestamp": datetime.now().astimezone().isoformat(),
                "optimizer_step_skipped": accelerator.optimizer_step_was_skipped,
                "epoch": epoch,
                "recurrent_state_mode": "memory_queries",
                "batch_window_vision": bool(cfg.trainer.get("batch_window_vision", False)),
                "refill_slots_per_step": batch.continuous_refill,
                "active_memory_slots_rank0": sum(state is not None for state in state_by_slot.values()),
                "moba_enabled": policy.moba_enabled,
                "moba_cache_delta_indices": (
                    None if policy.moba_cache_delta_indices is None
                    else list(policy.moba_cache_delta_indices)
                ),
                "moba_block_unit": policy.moba_memory.block_unit,
                "moba_topk_history_images": policy.moba_memory.topk,
                "moba_history_images": sum(
                    stats["moba_history_images"] for stats in moba_stats_by_time
                ) / max(len(moba_stats_by_time), 1),
                "moba_history_block_tokens_max": max(
                    (stats["moba_history_block_tokens_max"] for stats in moba_stats_by_time), default=0
                ),
                "moba_history_frames": sum(
                    stats["moba_history_frames"] for stats in moba_stats_by_time
                ) / max(len(moba_stats_by_time), 1),
                "moba_history_frames_max": max(
                    (stats["moba_history_frames_max"] for stats in moba_stats_by_time), default=0
                ),
                "moba_topk_effective": sum(
                    stats["moba_topk_effective"] for stats in moba_stats_by_time
                ) / max(len(moba_stats_by_time), 1),
                "moba_history_attention_layers": max(
                    (stats["moba_history_attention_layers"] for stats in moba_stats_by_time), default=0
                ),
                "moba_slots_rank0": policy.moba_memory.state_summary(),
                "query_memory_stats_rank0": memory_stats_by_time,
                "memory_gate_by_slot_rank0": policy.query_memory.last_gate.cpu().tolist(),
                "memory_initial_state": "zeros",
                "memory_update_rule": policy.query_memory.update_rule,
            }
            last_log_step = step
            last_log_time = log_time
            decisions_since_log = 0
            data_wait_since_log = 0.0
            max_data_wait_since_log = 0.0
            if accelerator.is_main_process:
                root.mkdir(parents=True, exist_ok=True)
                with (root / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(metrics) + "\n")
                print(json.dumps(metrics), flush=True)
        if trace:
            if any(state is not None and state.grad_fn is not None for state in state_by_slot.values()):
                raise RuntimeError("Memory tokens were not detached at the TBPTT boundary")
            if accelerator.is_main_process:
                with (root / "state_validation.jsonl").open("a") as handle:
                    handle.write(json.dumps({
                        "step": step, "trace": trace, "boundary_detached": True,
                        "conv_state_policy": "zero_at_every_forward",
            "native_s_policy": "reset_every_decision",
            "cross_decision_s_layers": 0,
            "memory_queries": 64,
                        "full_attention_kv_policy": "detached_current_visual_images_all_tokens_top4",
            "history_query_scope": "all_tokens",
                        "hf_full_attention_kv_policy": "empty_at_every_forward",
                    }) + "\n")
        # Diagnostics reuse this update's metrics and never consume the episode stream.
        if int(cfg.trainer.get("eval_interval", 0)) > 0 and step % int(cfg.trainer.eval_interval) == 0:
            score = accelerator.reduce(loss.detach().float(), reduction="mean").item()
            if accelerator.is_main_process:
                with (root / "diagnostics.jsonl").open("a") as handle:
                    handle.write(json.dumps({"step": step, "training_action_l1": score}) + "\n")
        del batch, step_batches, loss, numerator, input_state, output, next_state, slot_states
        if stop_requested:
            accelerator.print(f"Save-and-stop requested at optimizer update {step}")
            break
        if bool(cfg.trainer.get("save_checkpoints", True)) and step % int(cfg.trainer.save_interval) == 0:
            save_weights(accelerator, model, cfg, step)
    if bool(cfg.trainer.get("save_checkpoints", True)):
        save_weights(accelerator, model, cfg, last_completed_step, final=True)
    if stop_requested:
        if accelerator.is_main_process:
            (root / "early_stop.json").write_text(json.dumps({
                "step": last_completed_step,
                "reason": "SAVE_AND_STOP",
                "checkpoint_saved": bool(cfg.trainer.get("save_checkpoints", True)),
                "timestamp": datetime.now().astimezone().isoformat(),
            }, indent=2) + "\n")
        accelerator.print(f"Training stopped at update {last_completed_step}: {root}")
    else:
        accelerator.print(f"Training complete: {root}")
    accelerator.end_training()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", required=True)
    args, overrides = parser.parse_known_args()
    config = OmegaConf.merge(
        OmegaConf.load(args.config_yaml),
        OmegaConf.from_dotlist(normalize_dotlist_args(overrides)),
    )
    main(config)
