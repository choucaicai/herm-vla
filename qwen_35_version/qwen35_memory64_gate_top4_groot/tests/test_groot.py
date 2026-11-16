import copy
import io
import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from ..model.groot_head import GR00TActionHead, masked_flow_mse
from ..model.starvla.GR00T_ActionHeader import DiTConfig, FlowmatchingActionHead


class GR00TTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.cfg = OmegaConf.create({'framework': {'action_model': {
            'action_model_type': 'DiT-B', 'action_dim': 14, 'action_horizon': 5,
            'hidden_size': 32, 'state_dim': 0, 'num_inference_timesteps': 4,
            'num_target_vision_tokens': 2, 'add_pos_embed': True, 'max_seq_len': 16,
            'noise_beta_alpha': 1.5, 'noise_beta_beta': 1.0, 'noise_s': .999,
            'num_timestep_buckets': 1000, 'checkpoint_blocks': False,
            'repeated_diffusion_steps': 1,
            'diffusion_model_cfg': {'cross_attention_dim': 24, 'num_layers': 4,
                'output_dim': 32, 'dropout': 0., 'final_dropout': True,
                'interleave_self_attention': True, 'norm_type': 'ada_norm',
                'positional_embeddings': None}}}})
        self.shape_patch = patch.dict(DiTConfig, {'DiT-B': {
            'input_embedding_dim': 32, 'num_attention_heads': 4, 'attention_head_dim': 8}})
        self.shape_patch.start()
        self.addCleanup(self.shape_patch.stop)

    def inputs(self):
        return (torch.randn(2, 9, 24, requires_grad=True), torch.ones(2, 9, dtype=torch.bool),
                torch.randn(2, 5, 14), torch.ones(2, 5, 1, dtype=torch.bool),
                torch.randn(2, 5, 14), torch.tensor([.2, .7]))

    def test_upstream_loss_and_four_step_inference_parity(self):
        head = GR00TActionHead(self.cfg).eval()
        upstream = FlowmatchingActionHead(self.cfg).eval()
        upstream.load_state_dict(head.state_dict(), strict=True)
        condition, valid, target, mask, noise, time = self.inputs()
        actual, _ = head.training_loss(condition, valid, target, mask, noise=noise, time=time)
        with patch('torch.randn', return_value=noise), patch.object(upstream, 'sample_time', return_value=time):
            expected = upstream(condition, target)
        torch.testing.assert_close(actual, expected)
        actual = head.sample(condition, valid, noise=noise)
        with patch('torch.randn', return_value=noise.clone()):
            expected = upstream.predict_action(condition)
        torch.testing.assert_close(actual, expected)
        self.assertEqual([b.attn1.to_k.in_features for b in head.model.transformer_blocks],
                         [24, 32, 24, 32])

    def test_padding_and_invalid_tail_are_invariant(self):
        head = GR00TActionHead(self.cfg).eval()
        condition, valid, target, mask, noise, time = self.inputs()
        valid[0, :2] = False
        valid[1, -3:] = False
        mask[0, -2:] = False
        loss, _ = head.training_loss(condition, valid, target, mask, noise=noise, time=time)
        changed = condition.detach().clone().masked_fill(~valid[..., None], 1000.)
        bad = target.masked_fill(~mask.expand_as(target), float('nan'))
        other, _ = head.training_loss(changed, valid, bad, mask, noise=noise, time=time)
        torch.testing.assert_close(loss, other)
        torch.testing.assert_close(head.sample(condition, valid, noise=noise),
                                   head.sample(changed, valid, noise=noise))
        loss.backward()
        self.assertGreater(condition.grad[valid].abs().sum().item(), 0)
        self.assertEqual(condition.grad[~valid].abs().sum().item(), 0)
        for name, p in head.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_checkpoint_preserves_gradients_with_dropout(self):
        self.cfg.framework.action_model.diffusion_model_cfg.dropout = .2
        direct = GR00TActionHead(self.cfg).train()
        saved = copy.deepcopy(direct)
        saved.model.gradient_checkpointing = True
        condition, valid, target, mask, noise, time = self.inputs()
        other = condition.detach().clone().requires_grad_()
        torch.manual_seed(99)
        a, _ = direct.training_loss(condition, valid, target, mask, noise=noise, time=time)
        torch.manual_seed(99)
        b, _ = saved.training_loss(other, valid, target, mask, noise=noise, time=time)
        a.backward(); b.backward()
        torch.testing.assert_close(a, b)
        for p, q in zip(direct.parameters(), saved.parameters()):
            torch.testing.assert_close(p.grad, q.grad)
        torch.testing.assert_close(condition.grad, other.grad)

    def test_eight_repetitions_average_without_changing_example_weight(self):
        head = GR00TActionHead(self.cfg).eval()
        repeated = copy.deepcopy(head)
        repeated.repeats = 8
        condition, valid, target, mask, noise, time = self.inputs()
        a, _ = head.training_loss(condition, valid, target, mask, noise=noise, time=time)
        b, predictions = repeated.training_loss(condition, valid, target, mask,
                                                noise=noise.repeat(8, 1, 1), time=time.repeat(8))
        torch.testing.assert_close(a, b)
        self.assertEqual(predictions.shape, (16, 5, 14))
        a_grad = torch.autograd.grad(a, condition, retain_graph=True)[0]
        b_grad = torch.autograd.grad(b, condition)[0]
        torch.testing.assert_close(a_grad, b_grad)

    def test_checkpoint_roundtrip_and_euler_direction(self):
        head = GR00TActionHead(self.cfg).eval()
        condition, valid, target, mask, noise, time = self.inputs()
        buffer = io.BytesIO()
        torch.save(head.state_dict(), buffer)
        buffer.seek(0)
        restored = GR00TActionHead(self.cfg).eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        torch.testing.assert_close(head.sample(condition, valid, noise=noise),
                                   restored.sample(condition, valid, noise=noise))
        before = noise.clone()
        with patch.object(head, 'velocity', return_value=target-noise) as velocity:
            actual = head.sample(condition, valid, noise=noise)
        torch.testing.assert_close(actual, target)
        torch.testing.assert_close(noise, before)
        self.assertEqual(velocity.call_count, 4)

    def test_global_mask_normalization(self):
        prediction = torch.randn(3, 5, 14, requires_grad=True)
        target = torch.randn_like(prediction)
        mask = torch.ones_like(prediction, dtype=torch.bool)
        mask[1, 2:] = False
        mask[2] = False
        full = masked_flow_mse(prediction, target, mask)
        split = sum(masked_flow_mse(prediction[i:i+1], target[i:i+1], mask[i:i+1])
                    * mask[i:i+1].sum() for i in range(3)) / mask.sum()
        torch.testing.assert_close(full, split)
        full.backward()
        self.assertEqual(prediction.grad[~mask].abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
