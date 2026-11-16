"""Strict checkpoint serving for the trained current-only, stateless policy."""
import json
import os
import logging
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch

from eventvla.model.framework.share_tools import read_mode_config
from ..model.eventvla_qwen35_mix import RealQwen35CurrentOnly, validate_temporal_batch


class CurrentOnlyServingPolicy(RealQwen35CurrentOnly):
    @classmethod
    def from_checkpoint(cls, checkpoint):
        checkpoint=Path(checkpoint).resolve()
        config,stats=read_mode_config(checkpoint)
        config=OmegaConf.create(config)
        if os.environ.get("QWEN35_BASE_MODEL"):
            config.framework.qwenvl.base_vlm=os.environ["QWEN35_BASE_MODEL"]
        model=cls(config)
        weights=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
        model.load_state_dict(weights,strict=True)
        model.norm_stats=stats
        model.unnormalize_actions(np.zeros((1,7)),stats['new_embodiment']['action'])
        model.eval()
        assert not hasattr(model,'query_memory') and not hasattr(model,'moba_memory')
        return model

    @torch.inference_mode()
    def predict_action(self, examples=None, episode_id=None, decision_id=None, **kwargs):
        if examples is None or len(examples)!=1:
            raise ValueError('Serve one current environment observation per request')
        if kwargs.get('recurrent_state') is not None:
            raise ValueError('Current-only serving rejects recurrent state')
        validate_temporal_batch(examples,data_config=self.config.datasets.vla_data)
        response,state=super().predict_action(examples,return_recurrent_state=True)
        assert state is None
        actions=response['normalized_actions']
        assert actions.shape==(1,50,7) and np.isfinite(actions).all()
        response['actions']=self.unnormalize_actions(actions,self.norm_stats['new_embodiment']['action'])
        native=dict(self._last_current_only_report)
        assert native['memory_read_tokens']==native['memory_write_tokens']==native['history_kv_tokens']==0
        assert native['input_tokens']==native['expanded_tokens']
        assert native['action_query_counts']==[50]
        assert native['native_s_reset_layers']==24 and native['cross_decision_s_layers']==0
        assert native['initial_full_attention_tokens']==0
        report={**native,'episode_id':episode_id,'decision_id':decision_id,
                'timestep':int(examples[0]['timestep']),'images':len(examples[0]['image']),
                'absolute_indices':[],'delta_indices':[0],'state_mode':'none','state_shape':None,
                'passed':True}
        response['current_only']=report
        if decision_id is None or decision_id<3 or decision_id%10==0:
            logging.info('PIPER_AUDIT_RUNTIME %s',json.dumps(report,sort_keys=True))
        return response


def metadata_for(model, checkpoint):
    return {'framework':'RealQwen35CurrentOnly','model_class':'CurrentOnlyServingPolicy',
            'input_mode':'current_only_2_images','memory':'none','memory_query_count':0,
            'native_s_policy':'reset_every_decision','conv_state_policy':'zero_at_every_forward',
            'full_attention_kv_policy':'local_per_decision_only','history_kv_enabled':False,
            'checkpoint':str(Path(checkpoint).resolve()),'num_views':2,
            'absolute_indices':[],'delta_indices':[0],'sampling_interval':50,
            'action_horizon':50,'action_dim':7,'action_token':model.action_token,
            'action_token_id':int(model.action_token_id),'strict_checkpoint_load':True,
            'camera_order':['cam_high','cam_follower_wrist'],'action_mode':'abs','gripper_mode':'continuous',
            'action_names':[f'joint_{i}.pos' for i in range(1,7)]+['gripper.pos'],
            'fps':25,'actions_field':'denormalized absolute positions [1,50,7]'}
