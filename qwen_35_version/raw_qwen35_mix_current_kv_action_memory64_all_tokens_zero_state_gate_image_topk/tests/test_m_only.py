"""Final-model M-only: keep 64Q Gate memory, never persist historical KV."""
import copy
from pathlib import Path
import unittest

from omegaconf import OmegaConf
import torch

from ..model.eventvla_qwen35_mix import validate_config
from ..training.state import detach_slot_state
from .helpers import forward, sample, tiny_image_inputs, tiny_vision_policy


class MOnlyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(827)
        self.policy, _ = tiny_vision_policy()
        self.policy._initialize_moba({
            "block_unit": "image", "topk_history_images": 4,
            "cache_delta_indices": [], "attention_backend": "dense_reference",
        })
        self.inputs = tiny_image_inputs([2])

    def assert_empty_history(self):
        policy = self.policy
        self.assertEqual(policy.moba_cache_delta_indices, ())
        self.assertTrue(all(not slot.frames for slot in policy.moba_memory.slots))
        stats = policy.moba_memory.last_stats
        for key in ("moba_history_frames", "moba_history_images", "moba_new_frames",
                    "moba_topk_effective", "moba_history_attention_layers"):
            self.assertEqual(stats[key], 0, key)
        report = policy._last_memory64_report
        self.assertEqual(report["memory_tokens"], 64)
        self.assertEqual(report["cross_decision_s_layers"], 0)
        self.assertEqual(report["native_s_reset_layers"], 3)
        self.assertEqual(report["memory_update_rule"], "gated")

    def test_recipe_changes_only_kv_write_policy_and_run_name(self):
        folder = Path(__file__).resolve().parents[1] / "config"
        final = OmegaConf.load(folder / "train.yaml")
        monly = OmegaConf.load(folder / "train_m_only.yaml")
        validate_config(monly)
        self.assertEqual(monly.framework.recurrent_state_mode, "memory_queries")
        self.assertEqual(monly.framework.memory_queries.count, 64)
        self.assertEqual(monly.framework.memory_queries.native_s_policy, "reset_every_decision")
        self.assertEqual(list(monly.framework.moba_memory.cache_delta_indices), [])
        monly.run_id = final.run_id
        monly.framework.moba_memory.cache_delta_indices = [0]
        self.assertEqual(OmegaConf.to_container(monly), OmegaConf.to_container(final))

    def test_three_decisions_carry_only_64_query_vectors(self):
        policy, state = self.policy, None
        with torch.no_grad():
            for timestep in (0, 50, 100):
                _, state = forward(policy, self.inputs, [sample(timestep)], state)
                self.assertEqual(tuple(state.shape), (1, 64, 64))
                self.assertTrue(bool(torch.isfinite(state).all()))
                self.assertEqual(policy._last_query_memory_layout.has_memory.tolist(), [timestep > 0])
                self.assertTrue(all(not frames for frames in policy.moba_memory.pending.history))
                self.assertTrue(all(not frames for frames in policy.moba_memory.pending.new_frames))
                policy.commit_moba_memory()
                self.assert_empty_history()
                if timestep == 0:
                    self.assertTrue(bool((policy.query_memory.last_gate == 1).all()))
        with self.assertRaisesRegex(ValueError, "Native S"):
            policy._build_recurrent_cache(state)

    def test_warm_tbptt_trains_gate_and_feedback_without_historical_kv(self):
        policy = self.policy
        with torch.no_grad():
            _, warm = forward(policy, self.inputs, [sample(0)])
            policy.commit_moba_memory()
        _, first = forward(policy, self.inputs, [sample(50)], detach_slot_state(warm))
        first.retain_grad()
        policy.commit_moba_memory()
        hidden, second = forward(policy, self.inputs, [sample(100)], first)
        policy.commit_moba_memory()
        hidden[:, -2:].square().mean().backward()
        self.assertGreater(float(first.grad.abs().sum()), 0)
        for name, parameter in policy.query_memory.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0, name)
        self.assertFalse(detach_slot_state(second).requires_grad)
        self.assert_empty_history()

    def test_query_memory_changes_predictions_with_no_historical_kv(self):
        policy = self.policy.eval()
        with torch.no_grad():
            _, warm = forward(policy, self.inputs, [sample(0)])
            policy.commit_moba_memory()
            position_state = copy.deepcopy(policy.moba_memory)
            carried, _ = forward(policy, self.inputs, [sample(50)], warm)
            policy.commit_moba_memory()
            policy.moba_memory = copy.deepcopy(position_state)
            zeroed, _ = forward(policy, self.inputs, [sample(50)], torch.zeros_like(warm))
            policy.commit_moba_memory()
        self.assertGreater(float((carried[:, -2:] - zeroed[:, -2:]).abs().max()), 0)
        self.assert_empty_history()


if __name__ == "__main__":
    unittest.main()
