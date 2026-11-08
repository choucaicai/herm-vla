#!/usr/bin/env python3
"""Convert the prepared RMBench StarVLA LeRobot-v2 data to EventVLA's schema.

The source already uses LeRobot v2.0 parquet/video storage.  This converter keeps
those large files as symlinks, renames the modality keys expected by EventVLA's
AgilexData50Config, and rewrites the statistics wrapper.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

TASKS = [
    "battery_try",
    "blocks_ranking_try",
    "cover_blocks",
    "observe_and_pickup",
    "press_button",
    "put_back_block",
    "rearrange_blocks",
    "swap_T",
    "swap_blocks",
]


def _safe_link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and os.path.realpath(dst) == str(src.resolve()):
            return
        raise FileExistsError(f"Refusing to replace existing path: {dst}")
    try:
        dst.symlink_to(src, target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def convert_task(src: Path, dst: Path) -> dict:
    if not src.is_dir():
        raise FileNotFoundError(src)
    if dst.exists():
        raise FileExistsError(f"Destination already exists; remove it before re-running: {dst}")
    dst.mkdir(parents=True)
    (dst / "meta").mkdir()

    info = json.loads((src / "meta/info.json").read_text())
    info["codebase_version"] = "v2.0"
    info["robot_type"] = "robotwin_mem"
    (dst / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")

    episodes = src / "meta/episodes.jsonl"
    tasks = src / "meta/tasks.jsonl"
    (dst / "meta/episodes.jsonl").write_bytes(episodes.read_bytes())
    (dst / "meta/tasks.jsonl").write_bytes(tasks.read_bytes())

    # EventVLA's AgilexData50Config uses this semantic order while the source
    # vectors remain in the original [left_arm, left_gripper, right_arm,
    # right_gripper] storage order.
    modality = {
        "state": {
            "left_joints": {
                "start": 0, "end": 6, "original_key": "observation.state",
                "absolute": True, "dtype": "float32",
            },
            "right_joints": {
                "start": 7, "end": 13, "original_key": "observation.state",
                "absolute": True, "dtype": "float32",
            },
            "left_gripper": {
                "start": 6, "end": 7, "original_key": "observation.state",
                "absolute": True, "dtype": "float32",
            },
            "right_gripper": {
                "start": 13, "end": 14, "original_key": "observation.state",
                "absolute": True, "dtype": "float32",
            },
        },
        "action": {
            "left_joints": {
                "start": 0, "end": 6, "original_key": "action",
                "absolute": True, "dtype": "float32",
            },
            "right_joints": {
                "start": 7, "end": 13, "original_key": "action",
                "absolute": True, "dtype": "float32",
            },
            "left_gripper": {
                "start": 6, "end": 7, "original_key": "action",
                "absolute": True, "dtype": "float32",
            },
            "right_gripper": {
                "start": 13, "end": 14, "original_key": "action",
                "absolute": True, "dtype": "float32",
            },
        },
        "video": {
            "cam_high": {"original_key": "observation.images.head_camera"},
            "cam_left_wrist": {"original_key": "observation.images.left_camera"},
            "cam_right_wrist": {"original_key": "observation.images.right_camera"},
        },
        "annotation": {
            "human.action.task_description": {
                "original_key": "annotation.human.task_description"
            }
        },
    }
    (dst / "meta/modality.json").write_text(json.dumps(modality, indent=2) + "\n")

    source_stats = json.loads((src / "meta/stats_gr00t.json").read_text())
    if "statistics" in source_stats:
        source_stats = source_stats["statistics"]
    elif "abs" in source_stats:
        source_stats = source_stats["abs"]
    if not {"observation.state", "action"}.issubset(source_stats):
        raise ValueError(f"Unexpected statistics keys in {src}: {list(source_stats)}")
    # EventVLA recognizes the explicit action-mode wrapper.
    (dst / "meta/stats_gr00t.json").write_text(
        json.dumps({"abs": source_stats}, indent=2) + "\n"
    )

    # These are large and already validated. Keep one physical copy.
    _safe_link_or_copy((src / "data").resolve(), dst / "data")
    _safe_link_or_copy((src / "videos").resolve(), dst / "videos")

    # Preserve the existing cache only when present; it contains the same episode
    # ids/lengths and saves EventVLA from rebuilding the step index.
    steps = src / "meta/steps_data_index.pkl"
    if steps.exists():
        (dst / "meta/steps_data_index.pkl").write_bytes(steps.read_bytes())

    return {
        "task": src.name,
        "episodes": int(info["total_episodes"]),
        "frames": int(info["total_frames"]),
        "videos": int(info["total_videos"]),
        "source": str(src),
        "destination": str(dst),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    records = []
    for task in TASKS:
        records.append(convert_task(args.source / task, args.destination / task))
    manifest = {
        "format": "EventVLA LeRobot v2.0-compatible",
        "tasks": TASKS,
        "state_action_storage": "source vectors [left_arm6,left_gripper1,right_arm6,right_gripper1]",
        "eventvla_semantic_order": "[left_joints6,right_joints6,left_gripper1,right_gripper1]",
        "video_mapping": {
            "cam_high": "observation.images.head_camera",
            "cam_left_wrist": "observation.images.left_camera",
            "cam_right_wrist": "observation.images.right_camera",
        },
        "records": records,
    }
    (args.destination / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
