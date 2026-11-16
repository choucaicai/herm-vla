"""Masked RMBench adapter for the pinned StarVLA Qwen-GR00T action head.

Only this stateless expert is activation-checkpointed. Qwen and its recurrent
memory execute once per decision, outside the flow-matching repetitions.
"""
from __future__ import annotations

import torch

from .starvla.GR00T_ActionHeader import FlowmatchingActionHead


def expanded_action_mask(mask, reference):
    mask = torch.as_tensor(mask, device=reference.device, dtype=torch.bool)
    if mask.ndim == 2:
        mask = mask.unsqueeze(-1)
    if mask.shape[:2] != reference.shape[:2] or mask.shape[-1] not in (1, reference.shape[-1]):
        raise ValueError("Action mask must be [B,H], [B,H,1], or [B,H,D]")
    return mask.expand_as(reference)


def masked_flow_mse(prediction, velocity, mask):
    valid = expanded_action_mask(mask, prediction)
    error = torch.where(valid, prediction.float() - velocity.float(), 0.0)
    return error.square().sum() / valid.sum().clamp_min(1)


class GR00TActionHead(FlowmatchingActionHead):
    def __init__(self, config):
        super().__init__(config)
        if self.state_encoder is not None:
            raise ValueError("The final RMBench recipe disables proprioceptive state input")
        self.model.gradient_checkpointing = bool(self.config.checkpoint_blocks)
        self.repeats = int(self.config.repeated_diffusion_steps)
        if self.repeats < 1 or int(self.num_inference_timesteps) < 1:
            raise ValueError("Diffusion repetitions and inference steps must be positive")

    def _validate_conditions(self, condition, valid):
        if not isinstance(condition, torch.Tensor) or condition.ndim != 3:
            raise ValueError("GR00T requires the final Qwen hidden state [B,S,C]")
        if condition.shape[-1] != int(self.config.diffusion_model_cfg.cross_attention_dim):
            raise ValueError("Qwen condition width does not match the action expert")
        if valid.shape != condition.shape[:2] or not bool(valid.bool().any(-1).all()):
            raise ValueError("Each Qwen condition must contain valid tokens")

    def velocity(self, condition, valid, actions, time):
        dtype = self.action_encoder.layer1.weight.dtype
        condition, actions = condition.to(dtype), actions.to(dtype)
        timesteps = (time * self.num_timestep_buckets).long()
        features = self.action_encoder(actions, timesteps)
        if self.config.add_pos_embed:
            positions = torch.arange(actions.shape[1], device=actions.device)
            features = features + self.position_embedding(positions)[None]
        future = self.future_tokens.weight[None].expand(actions.shape[0], -1, -1)
        features = torch.cat((future, features), dim=1)
        # Diffusers expects additive attention bias, not a 0/1 padding mask.
        bias = torch.zeros(valid.shape, device=valid.device, dtype=dtype)
        bias = bias.masked_fill(~valid.bool(), float('-inf'))[:, None, :]
        hidden = self.model(features, condition, timestep=timesteps,
                            encoder_attention_mask=bias)
        return self.action_decoder(hidden)[:, -self.action_horizon:]

    def training_loss(self, condition, valid, actions, action_mask, *, noise=None, time=None):
        self._validate_conditions(condition, valid)
        actions = actions.to(device=condition.device, dtype=condition.dtype)
        if actions.shape != (condition.shape[0], self.action_horizon, self.action_dim):
            raise ValueError("Training action shape does not match the configured horizon")
        mask = expanded_action_mask(action_mask, actions)
        # Invalid episode-tail targets must not leak through action self-attention.
        actions = torch.where(mask, actions, 0.0)
        # Match StarVLA's repeat ordering and average over all independent samples.
        condition = condition.repeat(self.repeats, 1, 1)
        valid = valid.repeat(self.repeats, 1)
        actions = actions.repeat(self.repeats, 1, 1)
        mask = mask.repeat(self.repeats, 1, 1)
        noise = torch.randn_like(actions) if noise is None else noise.to(actions)
        time = (self.sample_time(actions.shape[0], actions.device, actions.dtype)
                if time is None else time.to(device=actions.device, dtype=actions.dtype))
        if noise.shape != actions.shape or time.shape != (actions.shape[0],):
            raise ValueError("Noise and time must include all diffusion repetitions")
        t = time[:, None, None]
        prediction = self.velocity(condition, valid, (1-t)*noise + t*actions, time)
        return masked_flow_mse(prediction, actions-noise, mask), prediction

    @torch.no_grad()
    def sample(self, condition, valid, *, noise=None, steps=None):
        self._validate_conditions(condition, valid)
        steps = int(self.num_inference_timesteps if steps is None else steps)
        if steps < 1:
            raise ValueError("Sampling requires at least one Euler step")
        dtype = self.action_encoder.layer1.weight.dtype
        shape = (condition.shape[0], self.action_horizon, self.action_dim)
        actions = (torch.randn(shape, device=condition.device, dtype=dtype) if noise is None
                   else noise.to(device=condition.device, dtype=dtype).clone())
        if actions.shape != shape:
            raise ValueError("Sampling noise has the wrong action shape")
        for step in range(steps):
            time = torch.full((shape[0],), step/steps, device=condition.device)
            actions = actions + self.velocity(condition, valid, actions, time) / steps
        return actions
