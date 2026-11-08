"""Final-model Top-6/8 budgets, exact masks and original Top-4 regression."""
from pathlib import Path
import unittest

from omegaconf import OmegaConf
import torch
from torch.nn import functional as F

from ..model.eventvla_qwen35_mix import validate_config
from ..model.frame_memory import FrameKVMemory, retrieval_settings
from ..model.moba_attention import sparse_frame_attention
from .reference_gate.model.moba_attention import sparse_frame_attention as original_attention


class TopKBudgetTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(643)

    def test_recipes_change_only_budget_and_run_name(self):
        folder = Path(__file__).resolve().parents[1] / "config"
        base = OmegaConf.load(folder / "train.yaml")
        for k in (6, 8):
            with self.subTest(k=k):
                cfg = OmegaConf.load(folder / f"train_top{k}.yaml")
                validate_config(cfg)
                self.assertEqual(retrieval_settings(cfg.framework.moba_memory), ("image", k))
                self.assertEqual(FrameKVMemory([3], topk=k, block_unit="image").topk, k)
                cfg.run_id = base.run_id
                cfg.framework.moba_memory.topk_history_images = 4
                self.assertEqual(OmegaConf.to_container(cfg), OmegaConf.to_container(base))
        for k in (0, 9):
            with self.assertRaises(ValueError):
                retrieval_settings(dict(block_unit="image", topk_history_images=k))
            with self.assertRaises(ValueError):
                FrameKVMemory([3], topk=k, block_unit="image")

    def inputs(self, images=10):
        batch, heads, kv_heads, seq, dim, width = 2, 4, 2, 7, 16, 64
        q = torch.randn(batch, heads, seq, dim, requires_grad=True)
        ck = torch.randn(batch, kv_heads, seq, dim, requires_grad=True)
        cv = torch.randn_like(ck, requires_grad=True)
        hk = torch.randn(batch, kv_heads, images * width, dim)
        hv = torch.randn_like(hk)
        lengths = torch.full((batch, images), width, dtype=torch.long)
        lengths[1] = 0
        means = hk.reshape(batch, kv_heads, images, width, dim).mean(-2)
        valid = torch.ones(batch, seq, dtype=torch.bool)
        valid[1, :2] = False
        return q, ck, cv, (hk, hv, means, lengths, width), valid

    def test_all_six_or_eight_choices_are_attended_with_valid_gradients(self):
        for k in (6, 8):
            with self.subTest(k=k):
                q, ck, cv, packed, valid = self.inputs()
                hk, hv, means, lengths, width = packed
                output, route = sparse_frame_attention(q, ck, cv, packed, valid,
                    history_query_mask=valid, topk=k, backend="dense_reference")
                self.assertEqual(route.shape[-1], k)
                self.assertTrue(bool((route[1] == -1).all()))
                # Independent exact reference from every selected image, not mask_mod.
                mask = torch.zeros(2, 4, 7, hk.shape[-2] + 7, dtype=torch.bool)
                for b in range(2):
                    for h in range(4):
                        for t in range(7):
                            for image in route[b, h, t].tolist():
                                if image >= 0:
                                    mask[b, h, t, image * width:image * width + int(lengths[b, image])] = True
                            mask[b, h, t, hk.shape[-2]:hk.shape[-2] + t + 1] = valid[b, :t + 1]
                            if not valid[b, t]:
                                mask[b, h, t, hk.shape[-2] + t] = True
                qr, kr, vr = [x.detach().clone().requires_grad_() for x in (q, ck, cv)]
                expected = F.scaled_dot_product_attention(qr, torch.cat([hk, kr], -2),
                    torch.cat([hv, vr], -2), attn_mask=mask, enable_gqa=True) * valid[:, None, :, None]
                torch.testing.assert_close(output, expected, atol=0, rtol=0)
                weight = torch.randn_like(output)
                (output * weight).sum().backward()
                (expected * weight).sum().backward()
                for x, ref in zip((q, ck, cv), (qr, kr, vr)):
                    self.assertTrue(bool(torch.isfinite(x.grad).all()))
                    torch.testing.assert_close(x.grad, ref.grad, atol=0, rtol=0)
                # The sixth/eighth selected image must affect the corresponding query.
                image = int(route[0, 0, 0, -1])
                changed = hv.clone()
                changed[0, :, image * width:(image + 1) * width] += 100
                alternate, _ = sparse_frame_attention(q, ck, cv,
                    (hk, changed, means, lengths, width), valid,
                    history_query_mask=valid, topk=k, backend="dense_reference")
                self.assertGreater(float((alternate[0, 0, 0] - output[0, 0, 0]).abs().max()), 0.1)

    def test_available_history_caps_effective_budget(self):
        q, ck, cv, packed, valid = self.inputs(images=3)
        for k in (6, 8):
            _, route = sparse_frame_attention(q, ck, cv, packed, valid,
                history_query_mask=valid, topk=k, backend="dense_reference")
            self.assertEqual(route.shape[-1], 3)
            self.assertTrue(bool((route[1] == -1).all()))

    def test_top4_outputs_and_gradients_match_original_implementation(self):
        q, ck, cv, packed, valid = self.inputs()
        output, route = sparse_frame_attention(q, ck, cv, packed, valid,
            history_query_mask=valid, topk=4, backend="dense_reference")
        qr, kr, vr = [x.detach().clone().requires_grad_() for x in (q, ck, cv)]
        reference, rr = original_attention(qr, kr, vr, packed, valid,
            history_query_mask=valid, topk=4, backend="dense_reference")
        torch.testing.assert_close(route, rr, atol=0, rtol=0)
        torch.testing.assert_close(output, reference, atol=0, rtol=0)
        weight = torch.randn_like(output)
        (output * weight).sum().backward()
        (reference * weight).sum().backward()
        for actual, expected in zip((q, ck, cv), (qr, kr, vr)):
            torch.testing.assert_close(actual.grad, expected.grad, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
