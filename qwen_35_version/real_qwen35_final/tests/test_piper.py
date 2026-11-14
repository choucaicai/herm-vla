import copy
from pathlib import Path
import unittest

import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn

from ..data.frame_tokens import build_frame_specs, select_cache_frame_specs
from ..data.piper_dataset import PiperLeRobotDataset, modality_metadata
from ..eval.observations import PiperObservationBuffer
from ..model.eventvla_qwen35_mix import RealQwen35Final, validate_config, validate_temporal_batch
from ..model.normalization import unnormalize_actions
from ..training.initialize import transfer_final_weights

PACKAGE = Path(__file__).resolve().parents[1]


class PiperContractTests(unittest.TestCase):
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
