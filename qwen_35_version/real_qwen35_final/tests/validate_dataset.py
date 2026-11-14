"""Validate real Parquet/video data and optional worker preprocessing, without training."""

import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from ..data.temporal_profile import build_recurrent_dataset
from ..data.episode_windows import RememEpisodeWindowDataset, RememWindow
from ..model.eventvla_qwen35_mix import validate_config, validate_temporal_batch
from ..model.normalization import unnormalize_actions


def validate(cfg, output, workers=False):
    validate_config(cfg)
    mixture = build_recurrent_dataset(cfg)
    source = mixture.datasets[0]
    windows = RememEpisodeWindowDataset(mixture, decision_stride=50, reset_each_window=False)
    output.mkdir(parents=True, exist_ok=True)
    mixture.save_dataset_statistics(output / "dataset_statistics.json")
    stats = json.loads((output / "dataset_statistics.json").read_text())["new_embodiment"]["action"]
    assert stats["mask"] == [True] * 7
    shard_ids = {}
    for ep in source.trajectory_ids:
        path = str(source.get_video_path(int(ep), "cam_high"))
        shard_ids.setdefault(path, []).append(int(ep))
    # Decode both ends of each file group, including the second MP4 shard.
    episodes = sorted({ep for ids in shard_ids.values() for ep in (ids[0], ids[-1])})
    records = []
    for ep in episodes:
        table = source.get_trajectory_data(ep)
        for timestep in (0, 50, len(table) - 1):
            sample, mask = windows._load_unit(RememWindow(0, ep, timestep))
            validate_temporal_batch([sample], data_config=cfg.datasets.vla_data)
            assert sample["action"].shape == (50, 7)
            assert np.isfinite(sample["action"]).all()
            assert len(sample["image"]) == 6
            assert int(mask.sum()) == min(50, len(table) - timestep)
            assert sample["lang"] in source.tasks["task"].tolist()
            raw = np.stack(table["action"].to_numpy())[min(timestep, len(table) - 1)]
            restored = unnormalize_actions(sample["action"][:1], stats)[0]
            expected = np.clip(raw, stats["q01"], stats["q99"])
            np.testing.assert_allclose(restored, expected, atol=1e-3)
            for camera in ("cam_high", "cam_follower_wrist"):
                meta = source.trajectory_ids_to_metadata[ep]
                key = f"observation.images.{camera}"
                assert source.get_video_path(ep, camera).is_file()
                records.append({"episode": ep, "timestep": timestep, "camera": camera,
                                "video": str(source.get_video_path(ep, camera)),
                                "episode_video_start": meta["videos/from_timestamps"][key]})
    report = {"passed": True, "episodes": len(source.trajectory_ids),
              "frames": int(source.trajectory_lengths.sum()), "tasks": len(source.tasks),
              "action_shape": [50, 7], "images_per_sample": 6,
              "camera_files": {p: len(ids) for p, ids in shard_ids.items()},
              "normalization_mask": stats["mask"], "decoded_checks": records}
    if workers:
        from ..data import build_dataloader
        from ..data.worker_preprocess import configure_qwen_worker_preprocessing
        cfg.output_dir = str(output / "worker_probe")
        cfg.datasets.vla_data.num_workers = 1
        cfg.datasets.vla_data.per_device_batch_size = 1
        configure_qwen_worker_preprocessing(
            enabled=True, model_id=cfg.framework.qwenvl.base_vlm,
            action_token=cfg.framework.action_token, chunk_len=50, use_image_role_text=True,
        )
        loader = build_dataloader(cfg)
        iterator = iter(loader)
        try:
            batch = next(iterator)
            assert batch.worker_preprocessed and batch.sequence_shape == (1, 2)
            assert all(example["action"].shape == (50, 7) for example in batch)
            assert batch.image_counts == [6, 6]
            report["worker_probe"] = {"sequence_shape": list(batch.sequence_shape),
                                       "image_counts": batch.image_counts, "preprocessed": True}
        finally:
            iterator._shutdown_workers()
    (output / "dataset_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "decoded_checks"}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config/train.yaml"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", action="store_true")
    args = parser.parse_args()
    validate(OmegaConf.load(args.config), Path(args.output), args.workers)


if __name__ == "__main__":
    main()
