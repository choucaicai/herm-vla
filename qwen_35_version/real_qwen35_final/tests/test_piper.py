import copy
from pathlib import Path
import unittest
import tempfile

import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn

from ..data.frame_tokens import build_frame_specs, select_cache_frame_specs
from ..data.piper_dataset import PiperLeRobotDataset, modality_metadata, resolve_piper_paths
from ..eval.observations import PiperObservationBuffer
from ..model.eventvla_qwen35_mix import RealQwen35Final, validate_config, validate_temporal_batch
from ..model.normalization import unnormalize_actions
from ..training.initialize import transfer_final_weights

PACKAGE = Path(__file__).resolve().parents[1]


class PiperContractTests(unittest.TestCase):
    def test_dataset_roots_single_multi_and_duplicate_alias(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary) / name for name in ("easy", "hard")]
            for path in paths:
                (path / "meta").mkdir(parents=True)
                (path / "meta/info.json").write_text("{}")
            cfg = OmegaConf.create({"dataset_path": str(paths[0])})
            self.assertEqual(resolve_piper_paths(cfg), [paths[0].resolve()])
            cfg.dataset_paths = [str(path) for path in paths]
            self.assertEqual(resolve_piper_paths(cfg), [path.resolve() for path in paths])
            alias = Path(temporary) / "alias"
            alias.symlink_to(paths[0], target_is_directory=True)
            cfg.dataset_paths = [str(paths[0]), str(alias)]
            with self.assertRaises(ValueError):
                resolve_piper_paths(cfg)
            cfg.dataset_paths = []
            with self.assertRaises(ValueError):
                resolve_piper_paths(cfg)
            cfg.dataset_paths = [str(paths[0]), str(Path(temporary) / "missing")]
            with self.assertRaises(FileNotFoundError):
                resolve_piper_paths(cfg)

    def test_config_rejects_old_shape_and_camera_count(self):
        cfg = OmegaConf.load(PACKAGE / "config/train.yaml")
        validate_config(cfg)
        for key, value in (("framework.action_model.action_dim", 14),
                           ("datasets.vla_data.num_camera_views", 3)):
            bad = copy.deepcopy(cfg)
            OmegaConf.update(bad, key, value)
            with self.assertRaises(ValueError):
                validate_config(bad)

    def test_online_history_and_current_only_kv(self):
        buffer = PiperObservationBuffer()
        for frame in range(51):
            buffer.append(frame, {camera: np.full((4, 4, 3), frame, np.uint8)
                                  for camera in ("cam_high", "cam_follower_wrist")})
        example = buffer.example("Place the blue block into the white box.")
        validate_temporal_batch([example])
        self.assertEqual([int(im[0, 0, 0]) for im in example["image"]], [0, 0, 35, 35, 50, 50])
        ids = torch.tensor([[9, 100, 100, 100, 100, 8] * 6 + [7] * 50])
        specs = build_frame_specs([example], ids, torch.tensor([[1, 4, 4]] * 6),
                                  image_token_id=100, deduplicate=False)
        current = select_cache_frame_specs(specs, [0])
        self.assertEqual(len(current[0]), 1)
        self.assertEqual(current[0][0]["frame_timestep"], 50)
        self.assertEqual(current[0][0]["views"], ["cam_high", "cam_follower_wrist"])
        self.assertEqual(len(current[0][0]["image_spans"]), 2)
        with self.assertRaises(ValueError):
            buffer.append(53, {})
        buffer.reset()
        self.assertEqual(buffer.timestep, -1)

    def test_camera_specific_video_file_indices(self):
        dataset = object.__new__(PiperLeRobotDataset)
        dataset._dataset_path = Path("/fixture")
        dataset._video_path_pattern = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        dataset._lerobot_modality_meta = modality_metadata()
        dataset.trajectory_ids_to_metadata = {60: {
            "data/file_index": 0,
            "videos/files": {"observation.images.cam_high": (1, 2),
                             "observation.images.cam_follower_wrist": (3, 4)},
        }}
        self.assertTrue(str(dataset.get_video_path(60, "cam_high")).endswith("chunk-001/file-002.mp4"))
        self.assertTrue(str(dataset.get_video_path(60, "cam_follower_wrist")).endswith("chunk-003/file-004.mp4"))

    def test_no_first_training_and_serving_profile(self):
        cfg = OmegaConf.load(PACKAGE / "config/train_two_tasks_no_first.yaml")
        validate_config(cfg)
        buffer = PiperObservationBuffer.from_config(cfg)
        for frame in range(51):
            buffer.append(frame, {camera: np.full((4, 4, 3), frame, np.uint8)
                                  for camera in ("cam_high", "cam_follower_wrist")})
            if frame in (0, 10, 16, 50):
                example = buffer.example("Place the yellow block into the white box.")
                validate_temporal_batch([example], data_config=cfg.datasets.vla_data)
                self.assertEqual([int(im[0, 0, 0]) for im in example["image"]],
                                 [max(0, frame - 15)] * 2 + [frame] * 2)
                self.assertTrue(all(meta["absolute_index"] is None for meta in example["image_metas"]))
                self.assertEqual([m["time_role"] for m in example["image_metas"]],
                                 ["history", "history", "current", "current"])
        self.assertIsNone(buffer._first)
        self.assertNotIn(0, dict(buffer._recent))
        ids = torch.tensor([[9, 100, 100, 100, 100, 8] * 4 + [7] * 50])
        specs = build_frame_specs([example], ids, torch.tensor([[1, 4, 4]] * 4),
                                  image_token_id=100, deduplicate=False)
        current = select_cache_frame_specs(specs, [0])
        self.assertEqual([group["frame_timestep"] for group in specs[0]], [35, 50])
        self.assertEqual([group["frame_timestep"] for group in current[0]], [50])
        self.assertEqual(len(current[0][0]["image_spans"]), 2)
        old = OmegaConf.load(PACKAGE / "config/train_two_tasks.yaml")
        with self.assertRaises(ValueError):
            validate_temporal_batch([example], data_config=old.datasets.vla_data)
        buffer.reset()
        self.assertEqual(buffer.timestep, -1)
        self.assertIsNone(buffer._first)

    def test_observation_profile_rejects_future_and_inconsistent_anchors(self):
        for kwargs in ({"delta_indices": (15, 0)}, {"absolute_indices": (5,)},
                       {"delta_indices": (-15,)}, {"delta_indices": (-15, 0, 0)}):
            with self.assertRaises(ValueError):
                PiperObservationBuffer(**kwargs)
        cfg = OmegaConf.load(PACKAGE / "config/train_two_tasks_no_first.yaml")
        cfg.datasets.vla_data.temporal.image.include_initial = True
        with self.assertRaises(ValueError):
            PiperObservationBuffer.from_config(cfg)

    def test_continuous_gripper_denormalization(self):
        stats = {"q01": [-1] * 6 + [0.0], "q99": [1] * 6 + [0.08], "mask": [True] * 7}
        action = np.zeros((3, 7), np.float32)
        action[:, 6] = [-1, 0, 1]
        output = RealQwen35Final.unnormalize_actions(action, stats)
        np.testing.assert_allclose(output[:, 6], [0.0, 0.04, 0.08], atol=1e-7)
        np.testing.assert_array_equal(action[:, 6], [-1, 0, 1])
        with self.assertRaises(ValueError):
            unnormalize_actions(action, dict(stats, mask=[True] * 6 + [False]))

    def test_checkpoint_transfer_only_reinitializes_output(self):
        def model(dim):
            result = nn.Module()
            result.qwen_vl_interface = nn.Linear(3, 3)
            result.query_memory = nn.Linear(3, 3)
            result.action_model = nn.Module()
            result.action_model.model = nn.Module()
            result.action_model.model.fc1 = nn.Linear(3, 4)
            result.action_model.model.fc2 = nn.Linear(4, dim)
            return result
        old, new = model(14), model(7)
        initial = new.action_model.model.fc2.weight.detach().clone()
        transfer_final_weights(new, old.state_dict())
        torch.testing.assert_close(new.query_memory.weight, old.query_memory.weight)
        torch.testing.assert_close(new.action_model.model.fc2.weight, initial)
        bad = dict(old.state_dict())
        bad.pop("query_memory.weight")
        with self.assertRaises(ValueError):
            transfer_final_weights(new, bad)


if __name__ == "__main__":
    unittest.main()
