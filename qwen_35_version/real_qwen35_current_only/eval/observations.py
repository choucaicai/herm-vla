"""Construct a current-frame observation without an image-history buffer."""

import numpy as np

CAMERAS = ("cam_high", "cam_follower_wrist")


def current_observation(images, instruction, timestep=0):
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("A nonempty instruction is required")
    if not isinstance(timestep, int) or isinstance(timestep, bool) or timestep < 0:
        raise ValueError("timestep must be a nonnegative integer")
    frames, metas = [], []
    for camera in CAMERAS:
        image = np.asarray(images[camera])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"{camera} must be RGB uint8 HWC")
        frames.append(image.copy())
        metas.append({"role": "anchor", "time_index": 0, "time_role": "current",
                      "delta": 0, "absolute_index": None, "view": camera,
                      "video_key": f"video.{camera}", "frame_timestep": timestep})
    return {"image": frames, "image_metas": metas, "lang": instruction,
            "language": instruction, "timestep": timestep, "slot_id": 0}
