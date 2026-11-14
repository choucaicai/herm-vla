"""Slot operations for recurrent memory embeddings; native DeltaNet S is local."""
from __future__ import annotations
import torch


def prepare_recurrent_step(examples, state_by_slot, episode_by_slot, timestep_by_slot, *, stride=50):
    slots, warm, resets = [], [], []
    for example in examples:
        slot = int(example.get("slot_id", 0))
        if slot in slots:
            raise ValueError("A time step cannot contain the same memory slot twice")
        episode = (example["episode_id"], example["view_id"], example.get("stream_visit_id"))
        reset = bool(example.get("reset")) or episode_by_slot.get(slot) != episode
        if reset:
            state_by_slot[slot] = None
            timestep_by_slot.pop(slot, None)
        previous = timestep_by_slot.get(slot)
        if previous is not None and int(example["timestep"]) != previous + stride:
            raise RuntimeError("A recurrent slot skipped or reordered a decision")
        episode_by_slot[slot] = episode
        slots.append(slot)
        warm.append(state_by_slot.get(slot) is not None)
        resets.append(reset)
    return merge_slot_states([state_by_slot.get(slot) for slot in slots]), warm, resets


def merge_slot_states(states):
    template = next((state for state in states if state is not None), None)
    if template is None:
        return None
    if not isinstance(template, torch.Tensor) or template.ndim != 3 or template.shape[1] != 64:
        raise ValueError("Each slot must carry one [1,64,D] memory tensor")
    if any(state is not None and state.shape != template.shape for state in states):
        raise ValueError("Memory slot shapes must match")
    return {"tokens": torch.cat([torch.zeros_like(template) if state is None else state for state in states]),
            "valid": torch.tensor([state is not None for state in states], device=template.device, dtype=torch.bool)}


def split_slot_states(state, batch_size):
    if state is None:
        return [None] * batch_size
    if not isinstance(state, torch.Tensor) or state.ndim != 3 or state.shape[:2] != (batch_size, 64):
        raise ValueError("Returned memory must have shape [batch,64,D]")
    return [state[index:index + 1] for index in range(batch_size)]


def update_recurrent_slots(examples, state, state_by_slot, timestep_by_slot):
    states = split_slot_states(state, len(examples))
    for example, slot_state in zip(examples, states):
        slot = int(example.get("slot_id", 0))
        timestep_by_slot[slot] = int(example["timestep"])
        state_by_slot[slot] = None if bool(example.get("done")) or bool(example.get("is_last")) else slot_state
    return states


def detach_slot_state(state):
    return None if state is None else state.detach().clone()


def state_metrics(state):
    return {"memory_tokens": 0 if state is None else int(state.shape[1]),
            "state_shape": None if state is None else list(state.shape),
            "state_bytes": 0 if state is None else state.numel() * state.element_size(),
            "memory_l2": 0.0 if state is None else float(state.detach().float().norm()),
            "cross_decision_s_layers": 0}
