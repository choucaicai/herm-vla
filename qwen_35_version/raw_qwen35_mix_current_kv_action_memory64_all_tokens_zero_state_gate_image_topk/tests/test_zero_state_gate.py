"""Zero-state, gate endpoints, recurrent gradients and mixed-episode regressions."""
import copy
import unittest
from unittest.mock import patch

import torch

from ..model.query_memory import QueryMemory
from ..training.state import detach_slot_state, merge_slot_states, prepare_recurrent_step
from .helpers import tiny_vision_policy, tiny_image_inputs, sample, forward
from .test_memory64 import text_inputs


class ZeroStateGateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(236)
        self.q = QueryMemory(8)
        self.inputs = text_inputs()

    def expand(self, state=None):
        return self.q.expand(self.inputs, state, action_token_id=110, pad_token_id=0)

    def hidden(self, layout):
        return torch.randn(1, layout.valid_mask.shape[1], 8, requires_grad=True)

    def candidate(self, hidden, layout):
        selected = hidden.gather(1, layout.write_indices[..., None].expand(-1, -1, 8))
        return self.q.feedback_projection(self.q.feedback_norm(selected))

    def test_cold_state_zero_and_prefix_excluded_from_all_attention_masks(self):
        expanded, layout = self.expand()
        self.assertFalse(layout.previous_memory.requires_grad)
        self.assertEqual(int(layout.previous_memory.count_nonzero()), 0)
        self.assertEqual(layout.has_memory.tolist(), [False])
        self.assertFalse(bool(layout.valid_mask[layout.read_mask].any()))
        self.assertFalse(bool(layout.history_query_mask[layout.read_mask].any()))
        self.assertTrue(bool(layout.valid_mask[layout.write_mask | layout.action_mask].all()))
        self.assertFalse(hasattr(self.q, "initial_memory"))
        self.assertTrue(self.q.write_queries.requires_grad)
        self.assertGreater(float(self.q.write_queries.std()), 0)

    def test_first_write_is_complete_even_with_closed_gate(self):
        _, layout = self.expand()
        hidden = self.hidden(layout)
        output = self.q.write(hidden, layout, force_gate=0)
        torch.testing.assert_close(output, self.candidate(hidden, layout), atol=0, rtol=0)
        self.assertTrue(bool((self.q.last_gate == 1).all()))

    def test_warm_gate_endpoints_and_initial_ratio(self):
        previous = torch.randn(1, 64, 8, requires_grad=True)
        _, layout = self.expand(previous)
        hidden = self.hidden(layout)
        candidate = self.candidate(hidden, layout)
        closed = self.q.write(hidden, layout, force_gate=0)
        torch.testing.assert_close(closed, previous, atol=0, rtol=0)
        opened = self.q.write(hidden, layout, force_gate=1)
        torch.testing.assert_close(opened, candidate, atol=0, rtol=0)
        output = self.q.write(hidden, layout)
        torch.testing.assert_close(self.q.last_gate, torch.full((1, 64), 0.2))
        torch.testing.assert_close(output, 0.8 * previous + 0.2 * candidate)
        output.square().mean().backward()
        for name, parameter in self.q.write_gate.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0, name)
        self.assertGreater(float(previous.grad.abs().sum()), 0)

    def test_slot_gates_depend_on_each_slots_content(self):
        with torch.no_grad():
            self.q.write_gate.weight[0, 0] = 2
        previous = torch.randn(1, 64, 8)
        _, layout = self.expand(previous)
        self.q.write(self.hidden(layout), layout)
        self.assertEqual(self.q.last_gate.shape, (1, 64))
        self.assertGreater(float(self.q.last_gate.std()), 0.01)

    def test_mixed_slots_mask_and_update_independently(self):
        x = self.inputs
        x = {k: v.repeat(1, 2, 1) if k == "position_ids" else
             v.repeat(2, 1, 1) if k == "inputs_embeds" else v.repeat(2, 1)
             for k, v in x.items()}
        previous = torch.randn(1, 64, 8, requires_grad=True)
        expanded, layout = self.q.expand(x, merge_slot_states([previous, None]),
                                        action_token_id=110, pad_token_id=0)
        self.assertEqual(layout.has_memory.tolist(), [True, False])
        self.assertEqual((layout.read_mask & layout.valid_mask).sum(-1).tolist(), [64, 0])
        self.assertEqual((layout.write_mask & layout.valid_mask).sum(-1).tolist(), [64, 64])
        h = torch.randn(2, expanded["input_ids"].shape[1], 8, requires_grad=True)
        out = self.q.write(h, layout, force_gate=0)
        torch.testing.assert_close(out[:1], previous, atol=0, rtol=0)
        selected = h.gather(1, layout.write_indices[..., None].expand(-1, -1, 8))
        candidate = self.q.feedback_projection(self.q.feedback_norm(selected))
        torch.testing.assert_close(out[1:], candidate[1:], atol=0, rtol=0)
        out.sum().backward()
        self.assertGreater(float(previous.grad.abs().sum()), 0)

    def test_zero_replace_control_and_forced_one_have_identical_states(self):
        control = copy.deepcopy(self.q)
        control.update_rule = "replace"
        previous = torch.randn(1, 64, 8)
        _, layout = self.expand(previous)
        hidden = self.hidden(layout)
        torch.testing.assert_close(self.q.write(hidden, layout, force_gate=1),
                                   control.write(hidden, layout), atol=0, rtol=0)

    def test_bfloat16_interpolation_is_finite_and_preserves_gate_endpoints(self):
        self.q.to(torch.bfloat16)
        x = dict(self.inputs, inputs_embeds=self.inputs["inputs_embeds"].to(torch.bfloat16))
        previous = torch.randn(1, 64, 8, dtype=torch.bfloat16)
        expanded, layout = self.q.expand(x, previous, action_token_id=110, pad_token_id=0)
        hidden = torch.randn(1, expanded["input_ids"].shape[1], 8, dtype=torch.bfloat16, requires_grad=True)
        output = self.q.write(hidden, layout)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(bool(torch.isfinite(output).all()))
        torch.testing.assert_close(self.q.write(hidden, layout, force_gate=0), previous, atol=0, rtol=0)

    def test_old_learned_initial_state_cannot_silently_load(self):
        state = dict(self.q.state_dict(), initial_memory=torch.zeros(64, 8))
        with self.assertRaisesRegex(RuntimeError, "initial_memory"):
            self.q.load_state_dict(state, strict=True)

    def test_invalid_gate_configuration_rejected(self):
        for ratio in [0, 1, -0.1, float("nan")]:
            with self.assertRaises(ValueError):
                QueryMemory(8, gate_initial_write_ratio=ratio)
        with self.assertRaises(ValueError):
            QueryMemory(8, update_rule="unknown")


class NativeGatedMemoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(827)
        self.policy, _ = tiny_vision_policy()
        self.policy._initialize_moba({"attention_backend": "dense_reference"})
        self.inputs = tiny_image_inputs([2])

    def test_gate_gradients_cross_warm_tbptt_window_and_boundary_detaches(self):
        p = self.policy
        with torch.no_grad():
            _, warm = forward(p, self.inputs, [sample(0)])
            p.commit_moba_memory()
        _, first = forward(p, self.inputs, [sample(50)], detach_slot_state(warm))
        first.retain_grad()
        p.commit_moba_memory()
        hidden, second = forward(p, self.inputs, [sample(100)], first)
        p.commit_moba_memory()
        hidden[:, -2:].square().mean().backward()
        self.assertGreater(float(first.grad.abs().sum()), 0)
        for name, parameter in p.query_memory.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0, name)
        detached = detach_slot_state(second)
        self.assertFalse(detached.requires_grad)
        torch.testing.assert_close(detached, second, atol=0, rtol=0)

    def test_forced_one_matches_zero_replace_across_two_decisions(self):
        p = self.policy.eval()
        initial_history = copy.deepcopy(p.moba_memory)
        write = p.query_memory.write
        def forced(hidden, layout):
            return write(hidden, layout, force_gate=1)
        results = []
        with torch.no_grad():
            for rule in ["forced_one", "replace"]:
                p.moba_memory = copy.deepcopy(initial_history)
                state = None
                p.query_memory.update_rule = "replace" if rule == "replace" else "gated"
                outputs = []
                with patch.object(p.query_memory, "write", side_effect=forced if rule == "forced_one" else write):
                    for t in [0, 50]:
                        hidden, state = forward(p, self.inputs, [sample(t)], state)
                        p.commit_moba_memory()
                        outputs.append((hidden, state))
                results.append(outputs)
        for first, second in zip(*results):
            for x, y in zip(first, second):
                torch.testing.assert_close(x, y, atol=0, rtol=0)

    def test_episode_change_clears_memory_read_and_visual_history_together(self):
        p = self.policy
        inputs = tiny_image_inputs([1, 4])
        with torch.no_grad():
            _, memory = forward(p, inputs, [sample(slot=2), sample(slot=7, episode="b")])
            p.commit_moba_memory()
            states = {2: memory[:1], 7: memory[1:]}
            episodes = {2: ("a", 0, 0), 7: ("b", 0, 0)}
            times = {2: 0, 7: 0}
            examples = [sample(50, slot=2), sample(0, slot=7, episode="new")]
            state, _, _ = prepare_recurrent_step(examples, states, episodes, times)
            forward(p, inputs, examples, state)
        layout, context = p._last_query_memory_layout, p.moba_memory.pending
        self.assertEqual(layout.has_memory.tolist(), [True, False])
        self.assertEqual([len(x) for x in context.history], [1, 0])
        self.assertFalse(bool(context.valid_tokens[1][layout.read_mask[1]].any()))
        self.assertTrue(bool((p.query_memory.last_gate[1] == 1).all()))
        for route in context.route_indices.values():
            self.assertTrue(bool((route[1] == -1).all()))


if __name__ == "__main__":
    unittest.main()
