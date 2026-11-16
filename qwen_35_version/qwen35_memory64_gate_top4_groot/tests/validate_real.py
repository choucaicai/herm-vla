"""Real 4B forward/backward and serving validation; no optimizer updates."""
from __future__ import annotations
import argparse, copy, gc, json, socket, time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from omegaconf import OmegaConf
from accelerate import PartialState
from ..model.eventvla_qwen35_mix import Qwen35Memory64GateTop4GR00T
from ..eval.policy import RecurrentServingPolicy
from ..training.state import detach_slot_state
from ..training.kernel_warmup import warm_delta_kernels
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
        for slot in range(1):
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
    assert reads.tolist() == ([0] if cold else [64])
    assert layout.has_memory.tolist() == ([False] if cold else [True])
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
    parser=argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    PartialState()
    cfg=OmegaConf.load(args.config)
    print('Loading pretrained Qwen3.5-4B and fresh GR00T expert', flush=True)
    policy=Qwen35Memory64GateTop4GR00T(cfg).to(device='cuda', dtype=torch.bfloat16).train()
    warm_delta_kernels(policy.qwen_vl_interface.model.config.get_text_config(), torch.device('cuda', 0))
    groups=build_param_lr_groups(policy,cfg)
    grouped=[id(p) for group in groups for p in group['params']]
    assert len(grouped)==len(set(grouped))==len(list(policy.parameters()))
    del groups,grouped
    samples,provenance=build_samples(cfg)
    decisions=[]
    with torch.no_grad():
        output,state=policy(samples[0],return_recurrent_state=True)
        decisions.append(check_context(policy,0,cold=True))
        state=detach_slot_state(state)
    batches=[]
    for examples in samples[1:3]:
        inputs,raw=policy._prepare_qwen_inputs(examples)
        batches.append({'_eventvla_worker_preprocessed':True,'examples':raw,'qwen_inputs':inputs})
    prepared=policy(batches,prepare_window_vision=True)
    assert policy._window_vision_report['vision_encoder_calls']==1
    losses=[]
    for i,batch in enumerate(prepared):
        output,state=policy(batch,recurrent_state=state,return_recurrent_state=True)
        assert state.shape==(1,64,2560) and state.requires_grad
        assert torch.isfinite(output['total_loss'])
        if i==0:first=state;first.retain_grad()
        losses.append(output['total_loss'])
        decisions.append(check_context(policy,(i+1)*50))
        assert policy._last_condition_report['layers']==1
        assert policy._last_condition_report['valid_memory_read']==[64]
        print('REAL_FORWARD',i,float(output['total_loss'].detach()),policy._last_condition_report,flush=True)
    losses[1].backward()
    assert first.grad is not None and first.grad.abs().sum()>0
    memory_grad=float(first.grad.float().abs().sum())
    gradient_report={}
    for prefix,module in [('memory',policy.query_memory),('expert',policy.action_model)]:
        for name,p in module.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(),(prefix,name)
        gradient_report[prefix]=sum(float(p.grad.float().abs().sum()) for p in module.parameters())
        assert gradient_report[prefix]>0
    vision=policy.qwen_vl_interface.model.model.visual.patch_embed.proj.weight
    assert vision.grad is not None and torch.isfinite(vision.grad).all() and vision.grad.abs().sum()>0
    policy.zero_grad(set_to_none=True)
    del output,state,first,losses,batches,prepared
    gc.collect();torch.cuda.empty_cache()
    policy.eval();policy.reset_memory(batch_size=1)
    counts=[]
    with patch.object(policy.query_memory,'write',wraps=policy.query_memory.write) as write:
        with patch.object(policy.qwen_vl_interface.model.model,'forward',wraps=policy.qwen_vl_interface.model.model.forward) as qwen:
            with patch.object(policy.moba_memory,'stage',wraps=policy.moba_memory.stage) as stage:
                response,state=policy.predict_action(samples[0],return_recurrent_state=True,sampling_steps=4)
                assert write.call_count==qwen.call_count==stage.call_count==1
                counts=[write.call_count,qwen.call_count,stage.call_count]
    assert response['normalized_actions'].shape==(1,50,14)
    assert np.isfinite(response['normalized_actions']).all()
    policy.commit_moba_memory()
    policy.reset_memory(batch_size=1)
    # Exercise actual serving wrapper without loading another 4B model.
    policy.__class__=RecurrentServingPolicy
    policy.reset_memory()
    a=policy.predict_action(samples[0],episode_id='serve-a',decision_id=0)
    before=copy.deepcopy(policy.moba_memory.state_summary())
    b=policy.predict_action(samples[0],episode_id='serve-a',decision_id=0)
    np.testing.assert_array_equal(a['normalized_actions'],b['normalized_actions'])
    assert before==policy.moba_memory.state_summary()
    c=policy.predict_action(samples[1],episode_id='serve-a',decision_id=1)
    assert c['recurrent_memory']['warm_start']
    policy.predict_action(samples[0],episode_id='serve-b',decision_id=0)
    assert len(policy.moba_memory.slots[0].frames)==1
    report={'passed':True,'host':socket.gethostname(),'initialization':'fresh pretrained Qwen3.5-4B; random GR00T',
            'batch_size':1,'tbptt_steps':2,'condition_layers':1,'dit_layers':16,'dit_width':768,'actions':[1,50,14],
            'memory_gradient_from_next_decision':memory_grad,'gradient_l1':gradient_report,
            'four_denoise_steps_write_qwen_stage_calls':counts,
            'duplicate_serving_and_episode_reset':True,'historical_kv_detached':True,
            'optimizer_parameter_coverage':True,'optimizer_updates':0,'sample_provenance':provenance,
            'peak_cuda_memory_bytes':torch.cuda.max_memory_allocated()}
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
