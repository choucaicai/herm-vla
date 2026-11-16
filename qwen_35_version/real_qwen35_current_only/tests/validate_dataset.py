"""Validate real Parquet/video data and optional worker preprocessing, without training."""

import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from ..data.temporal_profile import build_recurrent_dataset
from ..data.episode_windows import RememEpisodeWindowDataset, RememSlotBatchSampler, RememWindow
from ..model.eventvla_qwen35_mix import validate_config, validate_temporal_batch
from ..model.normalization import unnormalize_actions


def validate_source(source, source_index, windows, cfg, stats):
    shard_ids = {}
    for ep in source.trajectory_ids:
        paths = tuple(str(source.get_video_path(int(ep), camera))
                      for camera in ("cam_high", "cam_follower_wrist"))
        shard_ids.setdefault(paths, []).append(int(ep))
    # Check each camera's shard boundaries independently for every source.
    episodes = sorted({ep for ids in shard_ids.values() for ep in (ids[0], ids[-1])})
    records = []
    for ep in episodes:
        table = source.get_trajectory_data(ep)
        for timestep in (0, 50, len(table) - 1):
            sample, mask = windows._load_unit(RememWindow(source_index, ep, timestep))
            sample["timestep"] = timestep
            validate_temporal_batch([sample], data_config=cfg.datasets.vla_data)
            assert sample["action"].shape == (50, 7)
            assert np.isfinite(sample["action"]).all()
            assert len(sample["image"]) == 2
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
                records.append({"source_index": source_index, "episode": ep, "timestep": timestep, "camera": camera,
                                "video": str(source.get_video_path(ep, camera)),
                                "episode_video_start": meta["videos/from_timestamps"][key]})
    return {"dataset": str(source.dataset_path), "source_index": source_index,
            "episodes": len(source.trajectory_ids), "frames": int(source.trajectory_lengths.sum()),
            "tasks": source.tasks["task"].tolist(),
            "camera_files": [{"paths": list(paths), "episodes": len(ids)} for paths, ids in shard_ids.items()],
            "decoded_checks": records}


def validate(cfg, output, workers=False):
    validate_config(cfg)
    mixture = build_recurrent_dataset(cfg)
    windows = RememEpisodeWindowDataset(mixture, decision_stride=50, reset_each_window=False)
    output.mkdir(parents=True, exist_ok=True)
    mixture.save_dataset_statistics(output / "dataset_statistics.json")
    stats = json.loads((output / "dataset_statistics.json").read_text())["new_embodiment"]["action"]
    assert stats["mask"] == [True] * 7
    sources = [validate_source(source, index, windows, cfg, stats)
               for index, source in enumerate(mixture.datasets)]
    episodes = sum(item["episodes"] for item in sources)
    assert len(set(windows.episode_ids)) == episodes
    # All sources use the same normalization bounds for training and serving.
    original_stats = [source.metadata.statistics.action["position"] for source in mixture.datasets]
    np.testing.assert_allclose(stats["q01"], np.min([s.q01 for s in original_stats], axis=0))
    np.testing.assert_allclose(stats["q99"], np.max([s.q99 for s in original_stats], axis=0))
    sampler = RememSlotBatchSampler(windows, batch_size=2, tbptt_steps=2, world_size=8,
                                   seed=int(cfg.seed), balance_task_step_counts=True)
    views = sampler._scheduled_views(0)
    counts = [sum(v.decision_count for v in views if v.episode.dataset_index == index)
              for index in range(len(sources))]
    assert min(counts) > 0
    assert max(counts) - min(counts) <= max(v.decision_count for v in views)
    report = {"passed": True, "episodes": episodes,
              "frames": sum(item["frames"] for item in sources),
              "sources": sources, "balanced_decisions_epoch0": counts,
              "unique_instructions": sorted({task for item in sources for task in item["tasks"]}),
              "action_shape": [50, 7], "images_per_sample": 2,
              "normalization_mask": stats["mask"]}
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
            seen = set()
            for batch_number in range(256):
                batch = next(iterator)
                assert batch.worker_preprocessed and batch.sequence_shape == (1, 2)
                assert all(example["action"].shape == (50, 7) for example in batch)
                assert batch.image_counts == [2, 2]
                seen.update(int(example["episode_id"].split(":")[0]) for example in batch)
                if len(seen) == len(sources):
                    break
            assert seen == set(range(len(sources))), seen
            report["worker_probe"] = {"sequence_shape": list(batch.sequence_shape),
                                       "image_counts": batch.image_counts, "preprocessed": True,
                                       "batches": batch_number + 1, "source_indices": sorted(seen)}
        finally:
            iterator._shutdown_workers()
    (output / "dataset_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    concise = dict(report, sources=[{k: v for k, v in source.items() if k != "decoded_checks"}
                                   for source in sources])
    print(json.dumps(concise, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config/train.yaml"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", action="store_true")
    args = parser.parse_args()
    validate(OmegaConf.load(args.config), Path(args.output), args.workers)


if __name__ == "__main__":
    main()
