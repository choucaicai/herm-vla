"""Align detached frame queues with the recurrent stream's stable slot IDs."""

from __future__ import annotations

from .frame_memory import EpisodeSlot, FrameKVMemory


class SlotFrameKVMemory(FrameKVMemory):
    def __init__(self, layer_indices, topk=4, max_history_frames=None):
        super().__init__(layer_indices, topk, max_history_frames)
        self.reset()

    def reset(self, batch_size=1):
        super().reset(batch_size)
        self._slots_by_id = {}
        self._episode_keys = {}
        self._active_slot_ids = []

    def prepare_slots(self, examples):
        if self.pending is not None:
            raise RuntimeError("Commit or discard pending visual KV before another forward")
        slot_ids = [int(sample.get("slot_id", index)) for index, sample in enumerate(examples)]
        if len(set(slot_ids)) != len(slot_ids):
            raise ValueError("A decision batch cannot repeat a visual memory slot")
        active = []
        for slot_id, sample in zip(slot_ids, examples):
            episode = sample.get("episode_id")
            if episode is None:
                raise ValueError("Stateful visual memory requires episode_id")
            identity = (episode, sample.get("view_id"), sample.get("stream_visit_id"))
            reset = bool(sample.get("reset") or sample.get("is_new_episode"))
            if reset or self._episode_keys.get(slot_id) != identity:
                self._slots_by_id[slot_id] = EpisodeSlot(episode_id=episode)
            self._episode_keys[slot_id] = identity
            slot = self._slots_by_id[slot_id]
            if slot.last_timestep is not None and int(sample["timestep"]) <= slot.last_timestep:
                raise ValueError("Episode timesteps must increase within a visual memory slot")
            active.append(slot)
        self.slots = active
        self._active_slot_ids = slot_ids

    def commit(self):
        if self.pending is None:
            return
        super().commit()
        for slot_id, slot in zip(self._active_slot_ids, self.slots):
            self._slots_by_id[slot_id] = slot

    def reset_by_mask(self, reset_mask, episode_ids=None):
        super().reset_by_mask(reset_mask, episode_ids)
        for slot_id, slot in zip(self._active_slot_ids, self.slots):
            self._slots_by_id[slot_id] = slot

    def state_summary(self):
        return [
            dict(summary, slot_id=slot_id)
            for slot_id, summary in zip(self._active_slot_ids, super().state_summary())
        ]
