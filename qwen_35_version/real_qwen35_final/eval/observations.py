"""Build two-camera observations using the checkpoint's temporal profile."""

from collections import deque
from numbers import Integral
import numpy as np

CAMERAS = ("cam_high", "cam_follower_wrist")


class PiperObservationBuffer:
    def __init__(self, *, absolute_indices=(0,), delta_indices=(-15, 0)):
        absolute_indices, delta_indices = tuple(absolute_indices), tuple(delta_indices)
        if any(isinstance(x, bool) or not isinstance(x, Integral)
               for x in (*absolute_indices, *delta_indices)):
            raise ValueError("Temporal indices must be integers")
        if absolute_indices not in ((), (0,)):
            raise ValueError("Piper observations support either no fixed anchor or frame zero")
        if (not delta_indices or delta_indices[-1] != 0
                or len(set(delta_indices)) != len(delta_indices)
                or any(x > 0 for x in delta_indices)):
            raise ValueError("Use unique nonpositive history offsets ending with current frame zero")
        self.absolute_indices = absolute_indices
        self.delta_indices = delta_indices
        self._capacity = 1 + abs(min(delta_indices))
        self.reset()

    @classmethod
    def from_config(cls, config):
        """Use config.yaml saved alongside the checkpoint, including old profiles."""
        image = config["datasets"]["vla_data"]["temporal"]["image"]
        absolute = tuple(image.get("absolute_indices", ()))
        if bool(image["include_initial"]) != bool(absolute):
            raise ValueError("include_initial must match absolute_indices")
        if not bool(image["include_current"]):
            raise ValueError("Piper observations require the current frame")
        return cls(absolute_indices=absolute, delta_indices=image["delta_indices"])

    def reset(self):
        self._first = None
        self._recent = deque(maxlen=self._capacity)
        self.timestep = -1

    def append(self, timestep, images):
        if timestep != self.timestep + 1:
            raise ValueError("Append every 25-Hz frame consecutively, starting at timestep 0")
        frames = {}
        for camera in CAMERAS:
            image = np.asarray(images[camera])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"{camera} must be an RGB uint8 HWC image")
            frames[camera] = image.copy()
        if self.absolute_indices and self._first is None:
            self._first = frames
        self._recent.append((timestep, frames))
        self.timestep = timestep

    def example(self, instruction):
        if self.timestep < 0 or not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("An observation and nonempty instruction are required")
        frames_by_time = dict(self._recent)
        if self._first is not None:
            frames_by_time[0] = self._first
        images, metas = [], []
        profile = [(0, "first", None, 0)] if self.absolute_indices else []
        profile += [(max(0, self.timestep + delta), "current" if delta == 0 else "history",
                     delta, None) for delta in self.delta_indices]
        for time_index, (frame, role, delta, absolute) in enumerate(profile):
            for camera in CAMERAS:
                images.append(frames_by_time[frame][camera].copy())
                metas.append({"role": "anchor", "time_index": time_index, "time_role": role,
                              "delta": delta, "absolute_index": absolute, "view": camera,
                              "video_key": f"video.{camera}", "frame_timestep": frame})
        return {"image": images, "image_metas": metas, "lang": instruction,
                "language": instruction, "timestep": self.timestep, "slot_id": 0}
