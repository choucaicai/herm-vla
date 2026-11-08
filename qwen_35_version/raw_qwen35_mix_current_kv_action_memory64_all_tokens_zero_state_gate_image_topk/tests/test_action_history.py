"""Verify action-only history selection, attention isolation and padding."""

from pathlib import Path
import unittest

from omegaconf import OmegaConf
import torch
from torch.nn import functional as F

from ..model.eventvla_qwen35_mix import action_checkpoint_config, validate_config
from ..model.moba_attention import make_sparse_mask, route_frames, sparse_frame_attention


class ActionHistoryTests(unittest.TestCase):
    def tensors(self, batch=3, sequence=129, frames=6):
        torch.manual_seed(402)
        heads, kv_heads, dim, width = 4, 2, 16, 64
        query = torch.randn(batch, heads, sequence, dim, requires_grad=True)
        key = torch.randn(batch, kv_heads, sequence, dim, requires_grad=True)
        value = torch.randn_like(key, requires_grad=True)
        history_key = torch.randn(batch, kv_heads, frames * width, dim)
        history_value = torch.randn_like(history_key)
        lengths = torch.full((batch, frames), width, dtype=torch.long)
        lengths[0, -1] = 3
        if batch > 1:
            lengths[1, 2:] = 0
        means = history_key.reshape(batch, kv_heads, frames, width, dim).mean(-2)
        valid = torch.ones(batch, sequence, dtype=torch.bool)
        if batch > 1:
            valid[1, :7] = False
        action = torch.zeros_like(valid)
        action[0, 61:111] = True  # straddles a sparse query-block boundary
        if batch > 1:
            action[1, 1:4] = True  # padding must never become an active query
            action[1, 79:129] = True
        return query, key, value, (history_key, history_value, means, lengths, width), valid, action

    def test_routes_match_independent_per_action_head_scores(self):
        q, _, _, packed, valid, action = self.tensors()
        _, _, means, lengths, _ = packed
        route = route_frames(q, means, lengths, 4, action & valid)
        active = (action & valid)[:, None, :, None].expand_as(route)
        self.assertTrue(bool((route[~active] == -1).all()))
        for b, token in (action & valid).nonzero().tolist():
            for head in range(q.shape[1]):
                scores = q[b, head, token].detach() @ means[b, head // 2].T
                scores *= q.shape[-1] ** -0.5
                scores[lengths[b] == 0] = -float("inf")
                expected = scores.topk(4).indices
                expected[lengths[b, expected] == 0] = -1
                torch.testing.assert_close(route[b, head, token], expected)

    def test_joint_attention_and_gradients_match_independent_dense_mask(self):
        q, k, v, packed, valid, action = self.tensors()
        hk, hv, _, lengths, width = packed
        output, route = sparse_frame_attention(
            q, k, v, packed, valid, history_query_mask=action, backend="dense_reference",
        )
        history_tokens = hk.shape[-2]
        mask = torch.zeros(q.shape[0], q.shape[1], q.shape[2], history_tokens + q.shape[2], dtype=torch.bool)
        for b in range(q.shape[0]):
            for head in range(q.shape[1]):
                for token in range(q.shape[2]):
                    if action[b, token] and valid[b, token]:
                        for frame in route[b, head, token].tolist():
                            if frame >= 0:
                                mask[b, head, token, frame * width:frame * width + lengths[b, frame]] = True
                    mask[b, head, token, history_tokens:history_tokens + token + 1] = valid[b, :token + 1]
                    if not valid[b, token]:
                        mask[b, head, token, history_tokens + token] = True
        qr, kr, vr = [tensor.detach().clone().requires_grad_() for tensor in (q, k, v)]
        reference = F.scaled_dot_product_attention(
            qr, torch.cat((hk, kr), -2), torch.cat((hv, vr), -2), attn_mask=mask, enable_gqa=True,
        ) * valid[:, None, :, None]
        torch.testing.assert_close(output, reference, atol=2e-6, rtol=2e-5)
        weight = torch.randn_like(output)
        (output * weight).mean().backward()
        (reference * weight).mean().backward()
        for actual, expected in ((q.grad, qr.grad), (k.grad, kr.grad), (v.grad, vr.grad)):
            torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)

    def test_history_cannot_directly_change_non_action_outputs_or_receive_their_gradient(self):
        q, k, v, packed, valid, action = self.tensors()
        hk, hv, means, lengths, width = packed
        hv.requires_grad_()
        original, _ = sparse_frame_attention(q, k, v, packed, valid, history_query_mask=action, backend="dense_reference")
        changed, _ = sparse_frame_attention(
            q, k, v, (hk, hv + 20, means, lengths, width), valid,
            history_query_mask=action, backend="dense_reference",
        )
        inactive = (~action & valid)[:, None, :, None].expand_as(original)
        active = (action & valid)[:, None, :, None].expand_as(original)
        torch.testing.assert_close(original[inactive], changed[inactive], atol=0, rtol=0)
        self.assertGreater(float((original[active] - changed[active]).abs().max()), 1)
        original[inactive].square().mean().backward()
        self.assertEqual(int(hv.grad.count_nonzero()), 0)

    def test_empty_action_mask_and_empty_history_slot_equal_local_attention(self):
        q, k, v, packed, valid, action = self.tensors()
        action.zero_()
        output, route = sparse_frame_attention(q, k, v, packed, valid, history_query_mask=action, backend="dense_reference")
        sequence = q.shape[-2]
        causal = torch.arange(sequence)[:, None] >= torch.arange(sequence)[None, :]
        mask = causal[None, None] & valid[:, None, None, :]
        mask = mask | (~valid[:, None, :, None] & torch.eye(sequence, dtype=torch.bool)[None, None])
        reference = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True) * valid[:, None, :, None]
        torch.testing.assert_close(output, reference, atol=2e-6, rtol=2e-5)
        self.assertTrue(bool((route == -1).all()))
        hk, hv, means, lengths, width = packed
        lengths.zero_()
        action.fill_(True)
        empty, empty_route = sparse_frame_attention(q, k, v, packed, valid, history_query_mask=action, backend="dense_reference")
        torch.testing.assert_close(empty, reference, atol=2e-6, rtol=2e-5)
        self.assertTrue(bool((empty_route == -1).all()))

    def test_block_mask_has_no_history_blocks_without_action_queries(self):
        q, _, _, packed, valid, action = self.tensors(batch=1)
        _, _, means, lengths, width = packed
        action.zero_()
        action[0, 128] = True
        route = route_frames(q, means, lengths, 4, action)
        block, _ = make_sparse_mask(route, lengths, valid, width)
        for head in range(q.shape[1]):
            for query_block in (0, 1):
                count = int(block.kv_num_blocks[0, head, query_block])
                self.assertTrue(bool((block.kv_indices[0, head, query_block, :count] >= lengths.shape[1]).all()))
            count = int(block.kv_num_blocks[0, head, 2])
            indices = block.kv_indices[0, head, 2, :count]
            self.assertEqual(int((indices < lengths.shape[1]).sum()), 4)

    def test_config_requires_memory_queries_and_own_checkpoint(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("train.yaml", "train_relative3.yaml", "train_history3.yaml", "train_delta25.yaml"):
            cfg=OmegaConf.load(root / "config" / name)
            validate_config(cfg)
            self.assertEqual(cfg.framework.moba_memory.history_query_scope,"all_tokens")
            cfg.framework.moba_memory.history_query_scope="action"
            with self.assertRaisesRegex(ValueError,"all_tokens"):validate_config(cfg)
        cfg=OmegaConf.load(root/"config/train.yaml")
        validate_config(action_checkpoint_config(cfg))
        cfg.framework.name="RawQwen35MixCurrentKVAction"
        with self.assertRaises(ValueError):action_checkpoint_config(cfg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
