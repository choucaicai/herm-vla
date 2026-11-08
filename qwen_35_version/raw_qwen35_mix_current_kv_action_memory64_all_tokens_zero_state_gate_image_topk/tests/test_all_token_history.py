"""History access must reach every valid category while preserving causality."""
import copy
import unittest
import torch
from ..model.moba_attention import sparse_frame_attention
from ..model.window_vision import prepare_window_vision_inputs
from ..training.state import merge_slot_states
from .helpers import tiny_vision_policy, tiny_image_inputs, history3_sample, forward


class AllTokenHistoryTests(unittest.TestCase):
    def test_history_values_affect_every_valid_token_but_not_padding(self):
        torch.manual_seed(134)
        query = torch.zeros(2, 4, 9, 8, requires_grad=True)
        key = torch.zeros(2, 2, 9, 8, requires_grad=True)
        value = torch.zeros_like(key, requires_grad=True)
        history_key = torch.zeros(2, 2, 192, 8)
        history_value = torch.ones_like(history_key)
        valid = torch.ones(2, 9, dtype=torch.bool)
        valid[1, :3] = False
        lengths = torch.tensor([[192], [192]])
        means = torch.zeros(2, 2, 1, 8)
        packed = history_key, history_value, means, lengths, 192
        output, route = sparse_frame_attention(
            query, key, value, packed, valid, history_query_mask=valid,
            backend='dense_reference',
        )
        active = valid[:, None, :, None].expand_as(output)
        self.assertTrue(bool((output[active] > 0.9).all()))
        self.assertTrue(bool((output[~active] == 0).all()))
        routed = valid[:, None, :, None].expand_as(route)
        self.assertTrue(bool((route[routed] == 0).all()))
        self.assertTrue(bool((route[~routed] == -1).all()))
        output.square().mean().backward()
        self.assertTrue(bool(torch.isfinite(query.grad).all()))

    def test_warm_memory_write_cannot_read_future_action_queries(self):
        torch.manual_seed(8)
        policy, backbone = tiny_vision_policy()
        policy._initialize_moba({'attention_backend': 'dense_reference'})
        policy.eval()
        raw = tiny_image_inputs([2])
        with torch.no_grad():
            _, memory = forward(policy, raw, [history3_sample(0)])
            policy.commit_moba_memory()
            snapshot = copy.deepcopy(policy.moba_memory)
            prepared, _ = prepare_window_vision_inputs(backbone, [raw])
            inputs = prepared[0]
            hidden, state = forward(policy, inputs, [history3_sample(50)], memory)
            policy.discard_moba_memory()
            policy.moba_memory = snapshot
            altered = dict(inputs, inputs_embeds=inputs['inputs_embeds'].clone())
            altered['inputs_embeds'][:, -2:] += 5
            changed, changed_state = forward(policy, altered, [history3_sample(50)], memory)
            policy.discard_moba_memory()
        torch.testing.assert_close(state, changed_state, atol=1e-7, rtol=1e-6)
        self.assertGreater(float((hidden[:, -2:] - changed[:, -2:]).abs().max()), 1e-3)

    def test_warm_batch_routes_no_padding_and_resets_only_changed_episode(self):
        torch.manual_seed(9)
        policy, _ = tiny_vision_policy()
        policy._initialize_moba({'attention_backend': 'dense_reference'})
        inputs = tiny_image_inputs([1, 5])
        _, memory = forward(policy, inputs, [history3_sample(0, slot=2), history3_sample(0, slot=7)])
        policy.commit_moba_memory()
        state = merge_slot_states([memory[:1], None])
        _, _ = forward(policy, inputs, [history3_sample(50, slot=2), history3_sample(0, slot=7, episode='new')], state)
        context = policy.moba_memory.pending
        layout = policy._last_query_memory_layout
        torch.testing.assert_close(context.history_query_mask, context.valid_tokens)
        self.assertEqual(int(context.history_query_mask[0, :4].sum()), 0)
        self.assertTrue(bool(context.history_query_mask[0][layout.read_mask[0]].all()))
        self.assertFalse(bool(context.history_query_mask[1][layout.read_mask[1]].any()))
        for route in context.route_indices.values():
            enabled = context.valid_tokens[0, None, :, None].expand_as(route[0])
            self.assertTrue(bool((route[0][enabled] >= 0).all()))
            self.assertTrue(bool((route[0][~enabled] == -1).all()))
            self.assertTrue(bool((route[1] == -1).all()))
        policy.discard_moba_memory()


if __name__ == '__main__':
    unittest.main()
