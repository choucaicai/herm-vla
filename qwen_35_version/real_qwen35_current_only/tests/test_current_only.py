import copy
from pathlib import Path
import unittest

import numpy as np
from omegaconf import OmegaConf

from ..eval.observations import current_observation
from ..model.eventvla_qwen35_mix import validate_config, validate_temporal_batch
from ..model.normalization import unnormalize_actions
from ..training.state import prepare_recurrent_step, update_recurrent_slots, detach_slot_state

PACKAGE = Path(__file__).resolve().parents[1]


class CurrentOnlyContractTests(unittest.TestCase):
    def test_rejects_each_history_channel(self):
        cfg = OmegaConf.load(PACKAGE / 'config/train.yaml')
        validate_config(cfg)
        for key, value in [
            ('datasets.vla_data.temporal.image.absolute_indices', [0]),
            ('datasets.vla_data.temporal.image.delta_indices', [-15, 0]),
            ('framework.memory_queries.enabled', True),
            ('framework.memory_queries.count', 64),
            ('framework.moba_memory.enabled', True),
            ('framework.moba_memory.cache_delta_indices', [0]),
            ('framework.recurrent_state_mode', 'memory_queries'),
            ('datasets.vla_data.include_state', True),
            ('framework.action_model.action_dim', 14),
        ]:
            with self.subTest(key=key):
                invalid = copy.deepcopy(cfg)
                OmegaConf.update(invalid, key, value)
                with self.assertRaises(ValueError):
                    validate_config(invalid)

    def test_current_observation_needs_no_history(self):
        images = {'cam_high': np.full((4, 4, 3), 12, np.uint8),
                  'cam_follower_wrist': np.full((4, 4, 3), 34, np.uint8)}
        sample = current_observation(images, 'Pick the red block.', timestep=200)
        validate_temporal_batch([sample])
        self.assertEqual(len(sample['image']), 2)
        self.assertEqual([m['frame_timestep'] for m in sample['image_metas']], [200, 200])
        images['cam_high'][:] = 255
        self.assertEqual(int(sample['image'][0][0, 0, 0]), 12)
        bad = copy.deepcopy(sample)
        bad['image_metas'][0]['frame_timestep'] = 150
        with self.assertRaises(ValueError):
            validate_temporal_batch([bad])
        bad = copy.deepcopy(sample)
        bad['image_metas'].reverse()
        with self.assertRaises(ValueError):
            validate_temporal_batch([bad])

    def test_slot_bookkeeping_rejects_tensor_state(self):
        states, episodes, timesteps = {}, {}, {}
        example = {'slot_id': 0, 'episode_id': '0:0', 'view_id': 0,
                   'stream_visit_id': 0, 'timestep': 0, 'reset': True}
        state, warm, _ = prepare_recurrent_step([example], states, episodes, timesteps)
        self.assertIsNone(state)
        self.assertEqual(warm, [False])
        update_recurrent_slots([example], None, states, timesteps)
        example.update(timestep=50, reset=False)
        self.assertIsNone(prepare_recurrent_step([example], states, episodes, timesteps)[0])
        with self.assertRaises(ValueError):
            update_recurrent_slots([example], np.zeros((1, 64, 2560)), states, timesteps)
        with self.assertRaises(ValueError):
            detach_slot_state(np.zeros(1))
        states[0] = np.zeros(1)
        with self.assertRaises(ValueError):
            prepare_recurrent_step([example], states, episodes, timesteps)

    def test_continuous_gripper(self):
        stats = {'q01': [-1] * 6 + [0], 'q99': [1] * 6 + [0.08], 'mask': [True] * 7}
        action = np.zeros((3, 7), np.float32)
        action[:, 6] = [-1, 0, 1]
        np.testing.assert_allclose(unnormalize_actions(action, stats)[:, 6], [0, 0.04, 0.08], atol=1e-7)


if __name__ == '__main__':
    unittest.main()
