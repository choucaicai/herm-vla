"""Build the same [first, t-15, t] two-camera observation used during training."""

from collections import deque
import numpy as np

CAMERAS = ("cam_high", "cam_follower_wrist")


class PiperObservationBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self._first = None
        self._recent = deque(maxlen=16)
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
        if self._first is None:
            self._first = frames
        self._recent.append((timestep, frames))
        self.timestep = timestep

    def example(self, instruction):
        if self.timestep < 0 or not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("An observation and nonempty instruction are required")
        history_t = max(0, self.timestep - 15)
        frames_by_time = dict(self._recent)
        frames_by_time[0] = self._first
        images, metas = [], []
        profile = [(0, "first", None, 0), (history_t, "history", -15, None),
                   (self.timestep, "current", 0, None)]
        for time_index, (frame, role, delta, absolute) in enumerate(profile):
            for camera in CAMERAS:
                images.append(frames_by_time[frame][camera].copy())
                metas.append({"role": "anchor", "time_index": time_index, "time_role": role,
                              "delta": delta, "absolute_index": absolute, "view": camera,
                              "video_key": f"video.{camera}", "frame_timestep": frame})
        return {"image": images, "image_metas": metas, "lang": instruction,
                "language": instruction, "timestep": self.timestep, "slot_id": 0}
