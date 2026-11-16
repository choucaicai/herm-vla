"""Track episode ordering only; no tensor state is retained between decisions."""


def prepare_recurrent_step(examples, state_by_slot, episode_by_slot, timestep_by_slot, *, stride=50):
    slots, resets = [], []
    for example in examples:
        slot = int(example.get("slot_id", 0))
        if slot in slots:
            raise ValueError("Duplicate slot in a decision batch")
        if state_by_slot.get(slot) is not None:
            raise ValueError("Current-only training cannot retain model state")
        episode = (example["episode_id"], example["view_id"], example.get("stream_visit_id"))
        reset = bool(example.get("reset")) or episode_by_slot.get(slot) != episode
        if reset:
            timestep_by_slot.pop(slot, None)
        previous = timestep_by_slot.get(slot)
        if previous is not None and int(example["timestep"]) != previous + stride:
            raise RuntimeError("A slot skipped or reordered a decision")
        episode_by_slot[slot] = episode
        slots.append(slot)
        resets.append(reset)
    return None, [False] * len(slots), resets


def update_recurrent_slots(examples, state, state_by_slot, timestep_by_slot):
    if state is not None:
        raise ValueError("Current-only models must return no recurrent state")
    for example in examples:
        slot = int(example.get("slot_id", 0))
        timestep_by_slot[slot] = int(example["timestep"])
        state_by_slot[slot] = None
    return [None] * len(examples)


def detach_slot_state(state):
    if state is not None:
        raise ValueError("Current-only models cannot retain or detach memory")
    return None
