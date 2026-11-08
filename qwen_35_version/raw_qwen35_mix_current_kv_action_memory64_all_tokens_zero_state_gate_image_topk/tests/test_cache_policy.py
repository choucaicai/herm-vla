"""Check configurable KV writes against worker metadata and real Qwen layers."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch

from ..data.frame_tokens import build_frame_specs, select_cache_frame_specs
from ..data.worker_preprocess import collate_worker_batch
from ..eval.policy import RecurrentServingPolicy
from ..eval.server_policy import load_policy
from ..model.eventvla_qwen35_mix import validate_config
from .helpers import history3_sample
from .helpers import tiny_image_inputs, tiny_vision_policy


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class CachePolicyTests(unittest.TestCase):
    def test_default_configs_keep_input_profiles_and_write_current_only(self):
        profiles = {
            "train.yaml": ([0], [-15, 0]),
            "train_delta25.yaml": ([], [-25, 0]),
            "train_history3.yaml": ([], [-30, -15, 0]),
        }
        for name, (absolute, delta) in profiles.items():
            with self.subTest(config=name):
                cfg = OmegaConf.load(PACKAGE_ROOT / "config" / name)
                validate_config(cfg)
                self.assertEqual(list(cfg.datasets.vla_data.temporal.image.absolute_indices), absolute)
                self.assertEqual(list(cfg.datasets.vla_data.temporal.image.delta_indices), delta)
                self.assertEqual(list(cfg.framework.moba_memory.cache_delta_indices), [0])
                del cfg.framework.moba_memory.cache_delta_indices
                validate_config(cfg)
        policy, _ = tiny_vision_policy()
        policy._initialize_moba({"attention_backend": "dense_reference"})
        self.assertEqual(policy.moba_cache_delta_indices, (0,))

    def test_invalid_write_policies_fail_before_training(self):
        cfg = OmegaConf.load(PACKAGE_ROOT / "config/train_history3.yaml")
        for invalid in (0, "0", ["0"], [True], [0.0], [0, 0], [-25], [15]):
            with self.subTest(policy=invalid):
                changed = copy.deepcopy(cfg)
                changed.framework.moba_memory.cache_delta_indices = invalid
                with self.assertRaisesRegex(ValueError, "cache_delta_indices"):
                    validate_config(changed)
        for valid in ([0], [-30], [-15, 0], [-30, -15, 0], None, []):
            with self.subTest(policy=valid):
                cfg.framework.moba_memory.cache_delta_indices = valid
                validate_config(cfg)

    def test_worker_preserves_clipped_roles_until_write_selection(self):
        inputs = tiny_image_inputs([2])
        inputs["input_ids"][inputs["input_ids"] == 100] = 248056
        batch = {
            "_eventvla_worker_preprocessed": True,
            "examples": [history3_sample(0)], "qwen_inputs": inputs,
        }
        with patch("qwen_35_version.raw_qwen35_mix_current_kv_action_memory64_all_tokens_zero_state_gate_image_topk.data.worker_preprocess.get_worker_info",
                   return_value=SimpleNamespace(id=2)), patch(
            "qwen_35_version.raw_qwen35_mix_current_kv_action_memory64_all_tokens_zero_state_gate_image_topk.data.worker_preprocess.preprocess_batch",
            return_value=batch,
        ):
            result = collate_worker_batch(batch["examples"])
        specs = result["moba_frame_specs"]
        self.assertEqual([group["delta"] for group in specs[0]], [-30, -15, 0])
        self.assertEqual([group["frame_timestep"] for group in specs[0]], [0, 0, 0])
        historical = select_cache_frame_specs(specs, [-30])[0]
        current = select_cache_frame_specs(specs, [0])[0]
        self.assertEqual(historical[0]["delta"], -30)
        self.assertEqual(current[0]["delta"], 0)
        self.assertEqual(historical[0]["token_indices"], [1, 2, 3, 4, 7, 8, 9, 10, 13, 14, 15, 16])
        self.assertEqual(current[0]["token_indices"], [37, 38, 39, 40, 43, 44, 45, 46, 49, 50, 51, 52])
        self.assertEqual(select_cache_frame_specs(specs, [-30, -15])[0][0]["delta"], -15)
        self.assertEqual(select_cache_frame_specs(specs, None)[0][0]["delta"], 0)

    def _run_decision(self, policy, inputs, timestep, state, worker_specs):
        examples = [history3_sample(timestep, slot=4)]
        policy._moba_examples = examples
        policy._moba_frame_specs = (
            build_frame_specs(examples, inputs["input_ids"], inputs["image_grid_thw"],
                              image_token_id=100, deduplicate=False)
            if worker_specs else None
        )
        try:
            hidden, state = policy._forward_qwen_with_state(inputs, state)
        finally:
            policy._moba_examples = None
            policy._moba_frame_specs = None
        pending = policy.moba_memory.pending
        token_spans = [indices.tolist() for _, indices in pending.new_frames[0]]
        policy.commit_moba_memory()
        return hidden, state, token_spans

    def test_real_forward_raw_and_worker_paths_obey_all_write_policies(self):
        expectations = [
            ([0], [0, 50], [0, 50, 100]),
            ([-30], [0, 20], [0, 20, 70]),
            ([-15, 0], [0, 35, 50], [0, 35, 50, 85, 100]),
            (None, [0, 20, 35, 50], [0, 20, 35, 50, 70, 85, 100]),
            ([], [], []),
        ]
        for write_deltas, after_two, after_three in expectations:
            cached_by_path = []
            hidden_by_path = []
            for worker_specs in (False, True):
                with self.subTest(policy=write_deltas, worker=worker_specs):
                    torch.manual_seed(86)
                    policy, _ = tiny_vision_policy()
                    policy._initialize_moba({
                        "attention_backend": "dense_reference",
                        "cache_delta_indices": write_deltas,
                    })
                    inputs = tiny_image_inputs([2])
                    with torch.no_grad():
                        hidden, state, spans = self._run_decision(policy, inputs, 0, None, worker_specs)
                        if write_deltas == [0]:
                            self.assertEqual(spans, [[101, 102, 103, 104, 107, 108, 109, 110, 113, 114, 115, 116]])
                        hidden, state, _ = self._run_decision(policy, inputs, 50, state, worker_specs)
                        self.assertEqual(list(policy.moba_memory.slots[0].frames), after_two)
                        hidden, state, _ = self._run_decision(policy, inputs, 100, state, worker_specs)
                    slot = policy.moba_memory.slots[0]
                    self.assertEqual(list(slot.frames), after_three)
                    self.assertTrue(all(torch.isfinite(item).all() for item in state if item is not None))
                    for entry in slot.frames.values():
                        for key, value, mean in entry.layers.values():
                            self.assertEqual(key.shape[-2], 12)
                            self.assertEqual(value.shape[-2], 12)
                            self.assertFalse(any(tensor.requires_grad for tensor in (key, value, mean)))
                    cached_by_path.append([entry.layers[3] for entry in slot.frames.values()])
                    hidden_by_path.append(hidden)
            torch.testing.assert_close(hidden_by_path[0], hidden_by_path[1])
            for raw, worker in zip(cached_by_path[0], cached_by_path[1]):
                for first, second in zip(raw, worker):
                    torch.testing.assert_close(first, second)

    def test_current_only_capacity_and_episode_reset(self):
        policy, _ = tiny_vision_policy()
        policy._initialize_moba({"attention_backend": "dense_reference", "max_history_frames": 2})
        inputs = tiny_image_inputs([2])
        state = None
        with torch.no_grad():
            for timestep in (0, 50, 100):
                _, state, _ = self._run_decision(policy, inputs, timestep, state, True)
        self.assertEqual(list(policy.moba_memory.slots[0].frames), [50, 100])
        policy.reset_memory()
        self.assertEqual(list(policy.moba_memory.slots[0].frames), [])

    def test_server_metadata_exposes_checkpoint_write_policy(self):
        for deltas in ((0,), (-15, 0), None, ()):
            with self.subTest(policy=deltas):
                model = SimpleNamespace(
                    config=OmegaConf.load(PACKAGE_ROOT / "config/train_history3.yaml"),
                    moba_cache_delta_indices=deltas, chunk_len=50,
                    action_token="\u25c6", action_token_id=158778,
                    query_memory=SimpleNamespace(update_rule="gated"),
                )
                with patch.object(RecurrentServingPolicy, "from_checkpoint", return_value=model):
                    _, metadata = load_policy("/tmp/current_kv_policy_test.pt")
                self.assertEqual(metadata["moba_cache_delta_indices"],
                                 None if deltas is None else list(deltas))
                self.assertEqual(metadata["framework"], "RawQwen35MixCurrentKVActionMemory64AllTokensZeroStateGateImageTopK")
                self.assertEqual(metadata["memory_initial_state"], "zeros")
                self.assertTrue(metadata["memory_mask_empty_read"])
                self.assertTrue(metadata["memory_force_first_write"])
                self.assertEqual(metadata["memory_update_rule"], "gated")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(CachePolicyTests)
    )
    if args.output:
        Path(args.output).write_text(json.dumps({
            "passed": result.wasSuccessful(), "tests": result.testsRun,
            "failures": len(result.failures), "errors": len(result.errors),
        }, indent=2) + "\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
