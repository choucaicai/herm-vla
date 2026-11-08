"""Validate the real Qwen3.5 zero-state gate on recorded abs0 + [-15,0] images."""
from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
import socket
import time
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import torch
from accelerate import PartialState

from ..model.eventvla_qwen35_mix import RawQwen35MixCurrentKVActionMemory64AllTokensZeroStateGateImageTopK
from ..model.window_vision import prepare_window_vision_inputs
from ..training.state import detach_slot_state
from eventvla.training.trainer_utils.trainer_tools import build_param_lr_groups


def build_samples(cfg):
    from ..data.temporal_profile import build_recurrent_dataset
    from ..data.episode_windows import RememEpisodeWindowDataset, RememWindow
    mixture = build_recurrent_dataset(cfg)
    source = mixture.datasets[0]
    trajectory = next(int(i) for i, n in zip(source.trajectory_ids, source.trajectory_lengths) if n > 200)
    windows = RememEpisodeWindowDataset(mixture, decision_stride=50, reset_each_window=False)
    result = []
    for index in range(4):
        base, _ = windows._load_unit(RememWindow(0, trajectory, index * 50))
        batch = []
        for slot in range(2):
            sample = copy.deepcopy(base)
            sample.update(episode_id=f"memory64-real-slot-{slot}", slot_id=slot,
                          view_id=0, stream_visit_id=0, timestep=index * 50,
                          done=False, is_last=False, reset=index == 0,
                          is_new_episode=index == 0)
            if slot:
                sample['lang'] += ' Carefully.'
            assert len(sample['image']) == 9
            assert [m['delta'] for m in sample['image_metas']] == [None] * 3 + [-15] * 3 + [0] * 3
            assert [m.get('absolute_index') for m in sample['image_metas']] == [0] * 3 + [None] * 6
            batch.append(sample)
        result.append(batch)
    return result, {'trajectory_id': trajectory, 'timesteps': [0, 50, 100, 150], 'source_tag': str(source.tag)}



def check_context(policy, timestep, *, cold=False):
    context = policy.moba_memory.pending
    layout = policy._last_query_memory_layout
    torch.testing.assert_close(context.history_query_mask, context.valid_tokens)
    assert bool((layout.write_mask.sum(-1) == 64).all())
    assert bool((layout.action_mask.sum(-1) == 50).all())
    reads = (layout.read_mask & layout.valid_mask).sum(-1)
    assert reads.tolist() == ([0, 0] if cold else [64, 64])
    assert layout.has_memory.tolist() == ([False, False] if cold else [True, True])
    if cold:
        assert not bool(layout.previous_memory.any())
        assert not layout.previous_memory.requires_grad
        assert bool((policy.query_memory.last_gate == 1).all())
    for route in context.route_indices.values():
        active = context.valid_tokens[:, None, :, None].expand_as(route)
        assert bool((route[active] >= 0).all())
        assert bool((route[~active] == -1).all())
    assert len(context.route_indices) == (0 if timestep == 0 else 8)
    assert policy.moba_memory.block_unit == 'image'
    assert context.stats()['moba_topk_effective'] == min(4, 3 * (timestep // 50))
    if timestep:
        assert context.stats()['moba_history_block_tokens_max'] == 64
        assert all(len(ids) == 3 * (timestep // 50) for ids in context.history_block_ids)
        for layer in policy.moba_memory.layer_indices:
            reference = torch.stack([frames[0].layers[layer][0] for frames in context.history])
            packed = context.pack_history(layer, reference)
            assert packed[-1] == 64 and bool((packed[-2] == 64).all())
    assert policy._last_memory64_report['native_s_reset_layers'] == 24
    assert policy._last_memory64_report['cross_decision_s_layers'] == 0
    policy.commit_moba_memory()
    for slot in policy.moba_memory.slots:
        assert list(slot.frames) == list(range(0, timestep + 1, 50))
        for entry in slot.frames.values():
            assert len(entry.layers) == 8
            assert len(entry.image_spans) == 3
            assert all(stop - start == 64 for _, start, stop in entry.image_spans)
            assert all(not mean.requires_grad and mean.grad_fn is None for mean in entry.image_means.values())
            for key, value, mean in entry.layers.values():
                assert key.shape[-2] == value.shape[-2] == 192
                assert all(not x.requires_grad and x.grad_fn is None for x in (key, value, mean))
    return dict(policy._last_memory64_report, cache_detached=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    start = time.monotonic()
    PartialState()
    torch.manual_seed(42)
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    cfg = OmegaConf.load(args.config)
    assert list(cfg.datasets.vla_data.temporal.image.absolute_indices) == [0]
    assert list(cfg.datasets.vla_data.temporal.image.delta_indices) == [-15, 0]
    assert cfg.framework.memory_queries.update_rule == 'gated'
    print('Loading fresh Qwen3.5-4B; no optimizer or checkpoint writes', flush=True)
    policy = RawQwen35MixCurrentKVActionMemory64AllTokensZeroStateGateImageTopK(cfg).to(device='cuda', dtype=torch.bfloat16).train()
    assert 'initial_memory' not in policy.query_memory.state_dict()
    assert 'initial_memory' not in dict(policy.query_memory.named_parameters())
    from ..training.kernel_warmup import warm_delta_kernels
    warmup = warm_delta_kernels(policy.qwen_vl_interface.model.config.get_text_config(), torch.device('cuda', torch.cuda.current_device()))
    groups = build_param_lr_groups(policy, cfg)
    grouped = [id(p) for group in groups for p in group['params']]
    assert len(grouped) == len(set(grouped)) == len(list(policy.parameters()))
    assert all(parameter.requires_grad for parameter in policy.parameters())
    assert any(group['name'] == 'query_memory' for group in groups)
    del groups, grouped
    samples, provenance = build_samples(cfg)
    decisions = []
    # First write is unconditional. Establish history without retaining a graph.
    with torch.no_grad():
        output, state = policy(samples[0], return_recurrent_state=True)
        decisions.append(check_context(policy, 0, cold=True))
        state = detach_slot_state(state)
    batches = []
    for examples in samples[1:3]:
        inputs, raw = policy._prepare_qwen_inputs(examples)
        batches.append({'_eventvla_worker_preprocessed': True, 'examples': raw, 'qwen_inputs': inputs})
    prepared = policy(batches, prepare_window_vision=True)
    assert policy._window_vision_report['vision_encoder_calls'] == 1
    assert policy._window_vision_report['valid_decisions'] == 4
    losses = []
    first = None
    for index, batch in enumerate(prepared):
        output, state = policy(batch, recurrent_state=state, return_recurrent_state=True)
        assert state.shape == (2, 64, 2560) and state.requires_grad
        assert bool(torch.isfinite(state).all()) and bool(torch.isfinite(output['total_loss']))
        if index == 0:
            first = state
            first.retain_grad()
        decisions.append(check_context(policy, (index + 1) * 50))
        losses.append(output['total_loss'])
        print(json.dumps({'phase': 'warm_forward', 'timestep': (index + 1) * 50,
                          'loss': float(output['total_loss'].detach()),
                          'gate': policy.query_memory.last_stats}), flush=True)
    losses[1].backward()
    assert first.grad is not None and bool(torch.isfinite(first.grad).all())
    memory_grad = float(first.grad.float().abs().sum())
    assert memory_grad > 0
    gradients = {}
    for name, parameter in policy.query_memory.named_parameters():
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), name
        gradients[name] = float(parameter.grad.float().abs().sum())
        assert gradients[name] > 0, name
    assert policy.qwen_vl_interface.model.model.visual.patch_embed.proj.weight.grad is not None
    detached = detach_slot_state(state)
    torch.testing.assert_close(detached, state, atol=0, rtol=0)
    assert not detached.requires_grad and detached.grad_fn is None
    policy.zero_grad(set_to_none=True)
    del output, state, first, losses, batches, prepared
    gc.collect()
    torch.cuda.empty_cache()
    policy.eval()
    write = policy.query_memory.write

    def force_one(hidden, layout):
        return write(hidden, layout, force_gate=1)

    def replace_reference(hidden, layout):
        selected = hidden.gather(1, layout.write_indices[..., None].expand(-1, -1, hidden.shape[-1]))
        candidate = policy.query_memory.feedback_projection(policy.query_memory.feedback_norm(selected))
        policy.query_memory._record_update(
            torch.ones((*candidate.shape[:2], 1), device=candidate.device),
            layout.previous_memory.float(), candidate.float(), layout.has_memory,
        )
        return candidate

    endpoints = []
    with torch.no_grad():
        for callback in [force_one, replace_reference]:
            policy.reset_memory(batch_size=2)
            state = None
            trajectory = []
            with patch.object(policy.query_memory, 'write', side_effect=callback):
                for batch in samples[:2]:
                    response, state = policy.predict_action(batch, recurrent_state=state, return_recurrent_state=True)
                    policy.commit_moba_memory()
                    assert response['normalized_actions'].shape == (2, 50, 14)
                    trajectory.append((response['normalized_actions'], state.clone()))
            endpoints.append(trajectory)
        action_error, state_error = 0.0, 0.0
        for actual, expected in zip(*endpoints):
            action_error = max(action_error, float(np.max(np.abs(actual[0] - expected[0]))))
            state_error = max(state_error, float((actual[1] - expected[1]).float().abs().max()))
            np.testing.assert_array_equal(actual[0], expected[0])
            torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)
        policy.reset_memory(batch_size=2)
        inputs, _ = policy._prepare_qwen_inputs(samples[0])
        prepared, _ = prepare_window_vision_inputs(policy.qwen_vl_interface.model.model, [inputs])
        x = prepared[0]
        def cold_forward(current):
            policy._moba_examples, policy._moba_frame_specs = samples[0], None
            try:
                hidden, memory = policy._forward_qwen_with_state(current)
                policy.discard_moba_memory()
                return hidden, memory
            finally:
                policy._moba_examples = policy._moba_frame_specs = None
        hidden, first = cold_forward(x)
        altered = dict(x, inputs_embeds=x['inputs_embeds'].clone())
        action_mask = x['input_ids'] == policy.action_token_id
        altered['inputs_embeds'][action_mask] += 2
        changed, second = cold_forward(altered)
        causality_error = float((first - second).float().abs().max())
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        assert float((hidden[action_mask] - changed[action_mask]).float().abs().max()) > 1e-3
        policy.reset_memory(batch_size=2)
        assert all(not slot.frames for slot in policy.moba_memory.slots)
    report = {
        'passed': True, 'host': socket.gethostname(), 'device': torch.cuda.get_device_name(),
        'initialization': 'pretrained Qwen3.5-4B; no VLA checkpoint loaded',
        'batch_size': 2, 'tbptt_steps': 2, 'absolute_indices': [0], 'delta_indices': [-15, 0],
        'initial_memory': 'fixed_zero_runtime_tensor_not_parameter', 'cold_read_prefix_masked': True,
        'first_write_complete': True, 'gate_granularity': 'slot', 'gate_initial_write_ratio': 0.2,
        'batch_window_vision': True, 'memory_shape': [2, 64, 2560],
        'native_s_reset_layers': 24, 'cross_decision_s_layers': 0, 'history_attention_layers': 8,
        'history_query_scope': 'all_valid_tokens', 'historical_KV_detached': True,
        'memory_gradient_from_second_warm_loss': memory_grad, 'query_memory_gradients_l1': gradients,
        'gate_one_vs_zero_replace_action_max_error': action_error,
        'gate_one_vs_zero_replace_state_max_error': state_error,
        'future_action_causality_error': causality_error, 'boundary_detach': True,
        'all_parameters_in_optimizer_groups': True, 'optimizer_updates': 0, 'checkpoint_writes': 0,
        'sample_provenance': provenance, 'decisions': decisions, 'delta_kernel_warmup': warmup,
        'peak_cuda_memory_bytes': torch.cuda.max_memory_allocated(), 'seconds': time.monotonic() - start,
        'comparison_scope': 'Gate=1 matches the zero-state masked-read replacement control, not the old learned-initial-state checkpoint',
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ['decisions', 'delta_kernel_warmup']}), flush=True)


if __name__ == '__main__':
    main()
