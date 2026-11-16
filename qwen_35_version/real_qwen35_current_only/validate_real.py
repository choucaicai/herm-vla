"""Validate current-only observations, memory isolation and real-model gradients."""
import argparse
import copy
import gc
import json
from pathlib import Path
import socket
import time

import numpy as np
import torch
from accelerate import PartialState
from omegaconf import OmegaConf

from .model.eventvla_qwen35_mix import RealQwen35CurrentOnly, validate_config
from .data.temporal_profile import build_recurrent_dataset
from .data.episode_windows import RememEpisodeWindowDataset, RememWindow
from .training.kernel_warmup import warm_delta_kernels
from eventvla.training.trainer_utils.trainer_tools import build_param_lr_groups


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    start=time.monotonic()
    PartialState()
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    cfg=OmegaConf.load(args.config)
    validate_config(cfg)
    invalid=[]
    for path,value in [
        ('datasets.vla_data.temporal.image.absolute_indices',[0]),
        ('datasets.vla_data.temporal.image.delta_indices',[-15,0]),
        ('framework.memory_queries.enabled',True),
        ('framework.memory_queries.count',64),
        ('framework.moba_memory.enabled',True),
        ('framework.moba_memory.cache_delta_indices',[0]),
        ('framework.recurrent_state_mode','memory_queries'),
    ]:
        bad=copy.deepcopy(cfg);OmegaConf.update(bad,path,value)
        try:validate_config(bad)
        except ValueError:invalid.append(path)
        else:raise AssertionError(path)
    from .eval.policy import CurrentOnlyServingPolicy
    policy=CurrentOnlyServingPolicy(cfg).to(device='cuda',dtype=torch.bfloat16).eval()
    assert not hasattr(policy,'query_memory') and not hasattr(policy,'moba_memory')
    assert not any('query_memory' in n for n,_ in policy.named_parameters())
    groups=build_param_lr_groups(policy,cfg)
    ids=[id(p) for g in groups for p in g['params']]
    assert len(ids)==len(set(ids))==len(list(policy.parameters()))
    warmup=warm_delta_kernels(policy.qwen_vl_interface.model.config.get_text_config(),torch.device('cuda',0))
    mixture=build_recurrent_dataset(cfg);source=mixture.datasets[0]
    trajectory=next(int(i) for i,n in zip(source.trajectory_ids,source.trajectory_lengths) if n>200)
    windows=RememEpisodeWindowDataset(mixture,decision_stride=50,reset_each_window=False)
    samples=[]
    for index in range(3):
        base,_=windows._load_unit(RememWindow(0,trajectory,index*50))
        batch=[]
        for slot in range(2):
            x=copy.deepcopy(base)
            x.update(episode_id=f'current-only-validation-{slot}',slot_id=slot,view_id=0,
                     stream_visit_id=0,timestep=index*50,done=False,is_last=False,
                     reset=index==0,is_new_episode=index==0)
            if slot:x['lang']+=' Carefully.'
            assert len(x['image'])==2
            assert [m['delta'] for m in x['image_metas']]==[0]*2
            assert all(m.get('absolute_index') is None and m['time_role']=='current' for m in x['image_metas'])
            batch.append(x)
        samples.append(batch)
    assert samples[0][0]['action'].shape == (50, 7)
    reports=[]
    with torch.no_grad():
        reference,state=RealQwen35CurrentOnly.predict_action(policy, samples[2],return_recurrent_state=True)
        assert state is None
        for batch in samples:
            response,state=RealQwen35CurrentOnly.predict_action(policy, batch,return_recurrent_state=True)
            assert state is None
            report=dict(policy._last_current_only_report)
            assert report['input_tokens']==report['expanded_tokens']
            assert report['action_query_counts']==[50,50]
            assert report['native_s_reset_layers']==24
            reports.append(report)
        np.testing.assert_array_equal(reference['normalized_actions'],response['normalized_actions'])
        assert np.isfinite(response['normalized_actions']).all()
        try:policy(samples[0],recurrent_state=torch.zeros(2,64,2560,device='cuda'))
        except ValueError:pass
        else:raise AssertionError('Accepted cross-decision memory')
    stats_path=Path(args.output).with_suffix('.statistics.json')
    mixture.save_dataset_statistics(stats_path)
    policy.norm_stats=json.loads(stats_path.read_text())
    served=policy.predict_action([samples[2][0]],episode_id='probe',decision_id=2)
    assert served['actions'].shape==(1,50,7) and np.isfinite(served['actions']).all()
    policy.predict_action([samples[0][0]],episode_id='other',decision_id=0)
    repeated=policy.predict_action([samples[2][0]],episode_id='probe',decision_id=2)
    np.testing.assert_array_equal(served['actions'],repeated['actions'])
    policy.train()
    batches=[]
    for examples in samples[:2]:
        inputs,raw=policy._prepare_qwen_inputs(examples)
        batches.append({'_eventvla_worker_preprocessed':True,'examples':raw,'qwen_inputs':inputs})
    prepared=policy(batches,prepare_window_vision=True)
    assert policy._window_vision_report['vision_encoder_calls']==1
    losses=[]
    for batch in prepared:
        output,state=policy(batch,return_recurrent_state=True)
        assert state is None and torch.isfinite(output['total_loss'])
        losses.append(output['total_loss'])
    loss=sum(losses)/2
    loss.backward()
    gradients={}
    for name,p in policy.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(),name
            gradients[name]=float(p.grad.float().abs().sum())
    for prefix in ['action_model.','qwen_vl_interface.model.model.visual.','qwen_vl_interface.model.model.language_model.']:
        assert sum(v for n,v in gradients.items() if n.startswith(prefix))>0,prefix
    report={'passed':True,'host':socket.gethostname(),'device':torch.cuda.get_device_name(),
            'images_per_decision':2,'absolute_indices':[],'delta_indices':[0],
            'memory_read_tokens':0,'memory_write_tokens':0,'history_kv_enabled':False,
            'returned_state':None,'native_s_reset_layers':24,'cross_decision_s_layers':0,
            'action_queries':50,'output_shape':list(response['normalized_actions'].shape),
            'repeat_after_unrelated_decisions_max_error':0.0,'rejected_configs':invalid,
            'all_parameters_in_optimizer':True,'finite_gradient_tensors':len(gradients),
            'finite_gradient_l1':sum(gradients.values()),'two_decision_action_loss':float(loss.detach()),
            'source_trajectory':trajectory,'source_tag':str(source.tag),'decisions':reports,
            'kernel_warmup':warmup,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),
            'elapsed_seconds':time.monotonic()-start}
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('decisions','kernel_warmup')}),flush=True)
    policy.zero_grad(set_to_none=True)
    del losses,loss,prepared,batches,output,policy,groups
    gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
