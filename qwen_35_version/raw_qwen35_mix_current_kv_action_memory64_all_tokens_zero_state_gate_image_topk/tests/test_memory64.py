"""Query memory, native causality, KV isolation, and two-step autograd tests."""
from pathlib import Path
import copy
import unittest
import torch
from omegaconf import OmegaConf
from ..model.query_memory import QueryMemory
from ..model.window_vision import prepare_window_vision_inputs
from ..model.eventvla_qwen35_mix import validate_config
from ..model.moba_attention import sparse_frame_attention
from ..training.state import (merge_slot_states,split_slot_states,detach_slot_state,
                              prepare_recurrent_step,update_recurrent_slots)
from .helpers import tiny_vision_policy,tiny_image_inputs,sample,forward


def text_inputs(left=0,right=0):
    ids=torch.tensor([[0]*left+[7]*8+[110]*50+[0]*right])
    valid=ids!=0
    positions=torch.zeros(3,1,ids.shape[1],dtype=torch.long)
    positions[:,0,left:left+58]=torch.arange(58)[None]
    return {"input_ids":ids,"attention_mask":valid.long(),"mm_token_type_ids":torch.zeros_like(ids),
            "position_ids":positions,"inputs_embeds":torch.randn(1,ids.shape[1],8,requires_grad=True)}


class MemoryLayoutTests(unittest.TestCase):
    def test_layout_preserves_original_tokens_and_action_ids(self):
        memory=QueryMemory(8)
        for left,right in [(0,0),(4,0),(0,5)]:
            x=text_inputs(left,right);e,layout=memory.expand(x,None,action_token_id=110,pad_token_id=0)
            torch.testing.assert_close(e["input_ids"].gather(1,layout.original_indices),x["input_ids"])
            torch.testing.assert_close(e["inputs_embeds"].gather(1,layout.original_indices[...,None].expand(-1,-1,8)),x["inputs_embeds"])
            self.assertEqual(int(layout.read_mask.sum()),64)
            self.assertEqual(int(layout.write_mask.sum()),64)
            self.assertEqual(int(layout.action_mask.sum()),50)
            self.assertEqual(int(layout.history_query_mask.sum()),122)
            torch.testing.assert_close(layout.history_query_mask,e['attention_mask'].bool())
            self.assertEqual(int((layout.history_query_mask&layout.read_mask).sum()),0)
            self.assertEqual(int((e["input_ids"]==110).sum()),50)
            self.assertEqual(e["input_ids"].shape[1],x["input_ids"].shape[1]+128)

    def test_three_axis_positions_shift_without_changing_image_geometry(self):
        memory=QueryMemory(8);x=text_inputs(3)
        x["position_ids"][1,0,5:9]=torch.tensor([2,2,3,3])
        x["position_ids"][2,0,5:9]=torch.tensor([2,3,2,3])
        e,l=memory.expand(x,None,action_token_id=110,pad_token_id=0)
        for i in range(x["input_ids"].shape[1]):
            expected=x["position_ids"][:,0,i]+(0 if i<3 else 64 if i<11 else 128)
            torch.testing.assert_close(e["position_ids"][:,0,l.original_indices[0,i]],expected)
        torch.testing.assert_close(e["position_ids"][:,0,l.read_mask[0]],torch.arange(64)[None].expand(3,-1))

    def test_frame_indices_follow_only_prefix_shift(self):
        q=QueryMemory(8);x=text_inputs();_,l=q.expand(x,None,action_token_id=110,pad_token_id=0)
        spec=[[{'token_indices':[1,2,3],'frame_timestep':0,'delta':0}]]
        self.assertEqual(l.remap_frame_specs(spec)[0][0]['token_indices'],[65,66,67])
        with self.assertRaises(ValueError):l.remap_frame_specs([[dict(spec[0][0],token_indices=[9])]])

    def test_cold_slots_are_fixed_zero_without_initial_memory_parameter(self):
        q=QueryMemory(8)
        state=torch.full((1,64,8),3.,requires_grad=True)
        merged=merge_slot_states([state,None])
        read,has_memory=q.read(merged,torch.zeros(2,4,8))
        self.assertTrue(bool((read[0]==3).all()));self.assertTrue(bool((read[1]==0).all()))
        self.assertEqual(has_memory.tolist(),[True,False])
        read.sum().backward()
        self.assertIsNotNone(state.grad)
        self.assertNotIn('initial_memory',dict(q.named_parameters()))
        self.assertNotIn('initial_memory',q.state_dict())

    def test_invalid_state_and_incomplete_actions_rejected(self):
        q=QueryMemory(8);x=text_inputs()
        with self.assertRaises(ValueError):q.read(torch.zeros(1,32,8),x['inputs_embeds'])
        with self.assertRaises(ValueError):QueryMemory(8,count=32)
        x['input_ids'][0,-1]=7
        with self.assertRaises(ValueError):q.expand(x,None,action_token_id=110,pad_token_id=0)

    def test_exact_training_contract(self):
        cfg=OmegaConf.load(Path(__file__).resolve().parents[1]/'config/train.yaml')
        validate_config(cfg)
        self.assertEqual(cfg.trainer.tbptt_steps,2)
        self.assertTrue(cfg.datasets.vla_data.batch_tbptt_video)
        self.assertEqual(cfg.framework.moba_memory.cache_grad,'detached')
        for path,value in [('framework.memory_queries.count',32),('framework.recurrent_state_mode','s_only'),
                           ('framework.memory_queries.native_s_policy','carry'),('framework.moba_memory.cache_grad','attached'),
                           ('framework.memory_queries.initial_state','learned'),('framework.memory_queries.mask_empty_read',False),
                           ('framework.memory_queries.force_first_write',False),('framework.memory_queries.gate_granularity','channel'),
                           ('framework.memory_queries.gate_initial_write_ratio',0.0)]:
            c=copy.deepcopy(cfg);OmegaConf.update(c,path,value)
            with self.assertRaises(ValueError):validate_config(c)


class SlotTests(unittest.TestCase):
    def test_split_merge_detach_preserves_gradient_then_cuts_boundary(self):
        state=torch.randn(2,64,8,requires_grad=True)
        a,b=split_slot_states(state,2);m=merge_slot_states([b,a]);m['tokens'].square().sum().backward()
        self.assertTrue(bool((state.grad!=0).all()))
        detached=detach_slot_state(a)
        self.assertIsNone(detached.grad_fn);self.assertFalse(detached.requires_grad)
        self.assertNotEqual(detached.data_ptr(),a.data_ptr())

    def test_episode_change_resets_one_slot_and_keeps_other(self):
        a=torch.randn(1,64,8,requires_grad=True);b=torch.randn_like(a)
        states={2:a,7:b};episodes={2:('a',0,0),7:('b',0,0)};times={2:0,7:0}
        batch=[sample(50,slot=2,episode='a'),sample(0,slot=7,episode='new')]
        result,warm,resets=prepare_recurrent_step(batch,states,episodes,times)
        self.assertEqual(warm,[True,False]);self.assertEqual(resets,[False,True])
        self.assertEqual(result['valid'].tolist(),[True,False])
        output=torch.randn(2,64,8,requires_grad=True)
        batch[1]['done']=True
        update_recurrent_slots(batch,output,states,times)
        self.assertIsNone(states[7]);self.assertEqual(times[2],50)
        with self.assertRaises(RuntimeError):prepare_recurrent_step([sample(150,slot=2)],states,episodes,times)


class NativeHybridTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.policy,self.backbone=tiny_vision_policy()
        self.policy._initialize_moba({'attention_backend':'dense_reference'})
        self.inputs=tiny_image_inputs([2])

    def test_read_write_action_image_and_text_all_route(self):
        p=self.policy
        _,memory=forward(p,self.inputs,[sample(0)])
        p.commit_moba_memory()
        hidden,next_memory=forward(p,self.inputs,[sample(50)],memory)
        layout=p._last_query_memory_layout;context=p.moba_memory.pending
        self.assertEqual(next_memory.shape,(1,64,64))
        self.assertEqual(hidden.shape[:2],self.inputs['input_ids'].shape)
        torch.testing.assert_close(context.history_query_mask,layout.history_query_mask)
        torch.testing.assert_close(context.history_query_mask,context.valid_tokens)
        self.assertTrue(bool(context.history_query_mask[layout.read_mask].all()))
        original_active=context.history_query_mask.gather(1,layout.original_indices)
        self.assertTrue(bool(original_active[self.inputs['attention_mask'].bool()].all()))
        for route in context.route_indices.values():
            active=layout.history_query_mask[:,None,:,None].expand_as(route)
            self.assertTrue(bool((route[active]>=0).all()))
            self.assertTrue(bool((route[~active]==-1).all()))
        self.assertEqual(p._last_memory64_report['cross_decision_s_layers'],0)
        self.assertEqual(p._last_memory64_report['native_s_reset_layers'],3)
        with self.assertRaises(ValueError):p._build_recurrent_cache(memory)
        p.commit_moba_memory()
        for entry in p.moba_memory.slots[0].frames.values():
            for tensors in entry.layers.values():
                for tensor in tensors:self.assertFalse(tensor.requires_grad);self.assertIsNone(tensor.grad_fn)

    def test_second_decision_loss_backpropagates_to_first_memory(self):
        p=self.policy
        _,memory=forward(p,self.inputs,[sample(0)]);memory.retain_grad();p.commit_moba_memory()
        hidden,next_memory=forward(p,self.inputs,[sample(50)],memory);p.commit_moba_memory()
        hidden[:,-2:].square().mean().backward()
        self.assertIsNotNone(memory.grad);self.assertGreater(float(memory.grad.abs().sum()),0)
        for parameter in p.query_memory.parameters():
            self.assertIsNotNone(parameter.grad);self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
        self.assertGreater(float(p.query_memory.feedback_projection.weight.grad.abs().sum()),0)
        self.assertTrue(bool(torch.isfinite(self.backbone.visual.patch_embed.proj.weight.grad).all()))
        self.assertIsNotNone(next_memory.grad_fn)

    def test_write_memory_cannot_read_future_action_queries(self):
        p=self.policy;p.eval()
        prepared,_=prepare_window_vision_inputs(self.backbone,[self.inputs]);inputs=prepared[0]
        with torch.no_grad():
            hidden,a=forward(p,inputs,[sample(0)]);p.discard_moba_memory()
            changed=dict(inputs);changed['inputs_embeds']=inputs['inputs_embeds'].clone()
            changed['inputs_embeds'][:,-2:]+=torch.randn_like(changed['inputs_embeds'][:,-2:])*5
            altered,b=forward(p,changed,[sample(0)]);p.discard_moba_memory()
        torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-6)
        self.assertGreater(float((hidden[:,-2:]-altered[:,-2:]).abs().max()),1e-3)

    def test_current_images_can_change_written_memory(self):
        p=self.policy;p.eval()
        with torch.no_grad():
            _,a=forward(p,self.inputs,[sample(0)]);p.discard_moba_memory()
            changed=dict(self.inputs);changed['pixel_values']=self.inputs['pixel_values']+2
            _,b=forward(p,changed,[sample(0)]);p.discard_moba_memory()
        self.assertGreater(float((a-b).abs().max()),1e-7)

    def test_window_vision_matches_separate_encoding(self):
        p=self.policy;p.eval()
        with torch.no_grad():
            _,a=forward(p,self.inputs,[sample(0)]);p.discard_moba_memory()
            prepared,_=prepare_window_vision_inputs(self.backbone,[self.inputs])
            _,b=forward(p,prepared[0],[sample(0)]);p.discard_moba_memory()
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)

    def test_padded_batch_matches_independent_slot_forwards(self):
        p=self.policy;p.eval();inputs=tiny_image_inputs([1,5])
        with torch.no_grad():
            _,batch=forward(p,inputs,[sample(0,slot=3),sample(0,slot=8)]);p.discard_moba_memory()
            single=tiny_image_inputs([1]);single['pixel_values']=inputs['pixel_values'][:144].clone()
            _,one=forward(p,single,[sample(0,slot=12)]);p.discard_moba_memory()
        torch.testing.assert_close(batch[:1],one,atol=1e-5,rtol=1e-4)
