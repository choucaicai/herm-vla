"""Read Piper LeRobot v3 directly, including per-camera video shards.

The source dataset is read-only. Positions retain their recorded units; all
seven action dimensions, including the continuous gripper, use q01/q99 scaling.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from eventvla.dataloader.gr00t_lerobot.datasets import (
    LeRobotSingleDataset, LeRobotMixtureDataset, ModalityConfig,
)
from eventvla.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from eventvla.dataloader.gr00t_lerobot.schema import DatasetMetadata, LeRobotModalityMetadata
from eventvla.dataloader.gr00t_lerobot.transform import ComposedModalityTransform
from eventvla.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor, StateActionTransform,
)

CAMERAS = ("cam_high", "cam_follower_wrist")
POSITION_NAMES = tuple(f"joint_{i}.pos" for i in range(1, 7)) + ("gripper.pos",)


def modality_metadata():
    def positions(original_key):
        return {"position": {"start": 0, "end": 7, "original_key": original_key,
                             "absolute": True, "dtype": "float32"}}
    return LeRobotModalityMetadata.model_validate({
        "state": positions("observation.state"),
        "action": positions("action"),
        "video": {camera: {"original_key": f"observation.images.{camera}"}
                  for camera in CAMERAS},
        "annotation": {"human.action.task_description": {"original_key": "task_index"}},
    })


def position_statistics(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7 or not np.isfinite(values).all():
        raise ValueError("Piper positions must be finite [frames, 7] arrays")
    return {
        "min": values.min(0).tolist(), "max": values.max(0).tolist(),
        "mean": values.mean(0).tolist(), "std": values.std(0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


class PiperLeRobotDataset(LeRobotSingleDataset):
    def __init__(self, *args, **kwargs):
        self._parquet_tables = {}
        super().__init__(*args, **kwargs)

    def _read_table(self, path):
        path = Path(path)
        if path not in self._parquet_tables:
            self._parquet_tables[path] = pd.read_parquet(path)
        return self._parquet_tables[path]

    def _get_lerobot_modality_meta(self):
        return modality_metadata()

    def _get_metadata(self, embodiment_tag):
        info = json.loads((self.dataset_path / "meta/info.json").read_text())
        if info.get("codebase_version") != "v3.0":
            raise ValueError("real_qwen35_final expects a LeRobot v3.0 dataset")
        for key in ("action", "observation.state"):
            feature = info["features"][key]
            if feature["shape"] != [7] or tuple(feature["names"]) != POSITION_NAMES:
                raise ValueError(f"Unexpected Piper field order or shape for {key}: {feature}")
        self.source_info = info
        tables = [self._read_table(p) for p in sorted(self.dataset_path.glob("data/*/*.parquet"))]
        if not tables or sum(len(t) for t in tables) != int(info["total_frames"]):
            raise ValueError("Parquet frame count does not match info.json")
        statistics = {}
        modalities = {}
        for modality, original in (("state", "observation.state"), ("action", "action")):
            values = np.concatenate([np.stack(table[original].to_numpy()) for table in tables])
            statistics[modality] = {"position": position_statistics(values)}
            modalities[modality] = {"position": {
                "absolute": True, "rotation_type": None, "shape": [7], "continuous": True,
            }}
        modalities["video"] = {}
        for camera in CAMERAS:
            feature = info["features"][f"observation.images.{camera}"]
            height, width, channels = feature["shape"]
            fps = feature["info"]["video.fps"]
            if channels != 3 or float(fps) != float(info["fps"]):
                raise ValueError(f"Invalid RGB camera metadata for {camera}")
            modalities["video"][camera] = {
                "resolution": [width, height], "channels": channels, "fps": fps,
            }
        return DatasetMetadata(statistics=statistics, modalities=modalities,
                               embodiment_tag=embodiment_tag)

    def _get_tasks(self):
        tasks = pd.read_parquet(self.dataset_path / "meta/tasks.parquet")
        if "task" not in tasks.columns:
            tasks = tasks.reset_index()
        if "task_index" not in tasks.columns:
            tasks = tasks.reset_index()
        if not {"task", "task_index"}.issubset(tasks.columns):
            raise ValueError("tasks.parquet must contain task text and task_index")
        if tasks["task_index"].duplicated().any():
            raise ValueError("Duplicate task_index in tasks.parquet")
        return tasks.set_index("task_index")[["task"]]

    def _get_all_steps(self):
        # Avoid the shared reader's on-dataset pickle cache and silent skips.
        steps = []
        for episode_id, length in zip(self.trajectory_ids, self.trajectory_lengths):
            table = self.get_trajectory_data_lerobot_v3(int(episode_id))
            for task_id in table["task_index"].unique():
                text = self.tasks.loc[int(task_id), "task"]
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"Missing instruction for task {task_id}")
            for camera in CAMERAS:
                path = self.get_video_path(int(episode_id), camera)
                if not path.is_file():
                    raise FileNotFoundError(path)
            steps.extend((int(episode_id), frame) for frame in range(int(length)))
        return steps

    def _get_trajectories(self):
        ids, lengths = super()._get_trajectories()
        if len(set(ids.tolist())) != len(ids) or len(ids) != self.source_info["total_episodes"]:
            raise ValueError("Invalid episode count or duplicate episode IDs")
        if int(lengths.sum()) != self.source_info["total_frames"]:
            raise ValueError("Episode lengths do not match total_frames")
        for path in sorted(self.dataset_path.glob("meta/episodes/*/*.parquet")):
            for _, row in pd.read_parquet(path).iterrows():
                episode_id = int(row["episode_index"])
                meta = self.trajectory_ids_to_metadata[episode_id]
                meta["videos/files"] = {}
                for camera in CAMERAS:
                    key = f"observation.images.{camera}"
                    prefix = f"videos/{key}"
                    meta["videos/files"][key] = (
                        int(row[f"{prefix}/chunk_index"]), int(row[f"{prefix}/file_index"]),
                    )
                    if key not in meta["videos/from_timestamps"]:
                        raise ValueError(f"Missing video time offset for episode {episode_id}, {key}")
        return ids, lengths

    def get_video_path(self, trajectory_id, key):
        original = self.lerobot_modality_meta.video[key].original_key
        chunk, file_index = self.trajectory_ids_to_metadata[int(trajectory_id)]["videos/files"][original]
        return self.dataset_path / self.video_path_pattern.format(
            video_key=original, chunk_index=chunk, file_index=file_index,
        )

    def get_trajectory_data_lerobot_v3(self, trajectory_id):
        trajectory_id = int(trajectory_id)
        if self.curr_traj_id == trajectory_id and self.curr_traj_data is not None:
            return self.curr_traj_data
        meta = self.trajectory_ids_to_metadata[trajectory_id]
        path = self.dataset_path / self.data_path_pattern.format(
            chunk_index=int(meta["data/chunk_index"]), file_index=int(meta["data/file_index"]),
        )
        table = self._read_table(path)
        episode = table.loc[table["episode_index"] == trajectory_id].reset_index(drop=True)
        expected = int(self.trajectory_lengths[self.get_trajectory_index(trajectory_id)])
        if len(episode) != expected or not np.array_equal(episode["frame_index"], np.arange(expected)):
            raise ValueError(f"Invalid frame sequence for episode {trajectory_id}")
        self.curr_traj_id, self.curr_traj_data = trajectory_id, episode
        return episode


def build_piper_mixture(cfg):
    data = cfg.datasets.vla_data
    if data.get("lerobot_version") != "v3.0" or data.get("action_mode") != "abs":
        raise ValueError("Piper recipe requires lerobot_version=v3.0 and action_mode=abs")
    path = Path(data.dataset_path).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[3] / path
    image = data.temporal.image
    modalities = {
        "video": ModalityConfig(delta_indices=list(image.delta_indices),
                                absolute_indices=list(image.absolute_indices),
                                modality_keys=[f"video.{camera}" for camera in CAMERAS]),
        "state": ModalityConfig(delta_indices=[0], modality_keys=["state.position"]),
        "action": ModalityConfig(delta_indices=list(range(50)), modality_keys=["action.position"]),
        "language": ModalityConfig(delta_indices=[0],
                                   modality_keys=["annotation.human.action.task_description"]),
    }
    keys = ["state.position", "action.position"]
    transforms = ComposedModalityTransform(transforms=[
        StateActionToTensor(apply_to=keys),
        StateActionTransform(apply_to=keys, normalization_modes={key: "q99" for key in keys}),
    ])
    source = PiperLeRobotDataset(
        dataset_path=path, modality_configs=modalities, transforms=transforms,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        video_backend=data.get("video_backend", "pyav"), data_cfg=data,
    )
    return LeRobotMixtureDataset(
        [(source, 1.0)], mode="train", balance_dataset_weights=False,
        balance_trajectory_weights=False, seed=int(cfg.seed), data_cfg=data,
    )
