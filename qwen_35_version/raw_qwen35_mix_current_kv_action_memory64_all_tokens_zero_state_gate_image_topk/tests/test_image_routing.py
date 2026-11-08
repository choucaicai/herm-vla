"""Independent camera routing, exact masks, cache lifetimes and TBPTT gradients."""
import argparse
import importlib
import json
from pathlib import Path
import unittest

from omegaconf import OmegaConf
import torch

from ..data.frame_tokens import build_frame_specs, select_cache_frame_specs
from ..model.frame_memory import FrameKVMemory, MemoryContext, retrieval_settings
from ..model.moba_attention import route_frames, sparse_frame_attention
from ..model.eventvla_qwen35_mix import validate_config
from .helpers import tiny_vision_policy, tiny_image_inputs, sample, history3_sample, forward
from .test_memory64 import NativeHybridTests
from .test_moba import attention_check


class ImageRoutingTests(unittest.TestCase):
    def test_configuration_requires_unambiguous_image_topk(self):
        cfg=OmegaConf.load(Path(__file__).resolve().parents[1]/'config/train.yaml')
        validate_config(cfg)
        self.assertEqual(retrieval_settings(cfg.framework.moba_memory),('image',4))
        self.assertNotIn('topk_history_frames',cfg.framework.moba_memory)
        for bad in [dict(block_unit='image',topk_history_frames=4),
                    dict(block_unit='frame_time',topk_history_images=4),
                    dict(block_unit='image',topk_history_images=5),dict(block_unit='camera')]:
            with self.subTest(settings=bad),self.assertRaises(ValueError):retrieval_settings(bad)

    def _capture(self,memory,timestep):
        example=history3_sample(timestep)
        ids=torch.tensor([[9]+[248056]*36+[7]])
        specs=build_frame_specs([example],ids,torch.tensor([[1,4,4]]*9),deduplicate=False)
        specs=select_cache_frame_specs(specs,[0])
        memory.prepare_slots([example])
        context=MemoryContext(memory,[example],specs,torch.ones_like(ids,dtype=torch.bool),[100+timestep])
        keys=torch.arange(2*ids.shape[1]*8,dtype=torch.float32).reshape(1,2,ids.shape[1],8).requires_grad_()
        values=(keys.detach()+1000).requires_grad_()
        context.capture(3,keys,values);memory.stage(context)
        return context,keys,values

    def test_cache_splits_three_cameras_without_duplicate_kv_storage(self):
        memory=FrameKVMemory([3],block_unit='image')
        _,keys,values=self._capture(memory,0);memory.commit()
        context,_,_=self._capture(memory,50)
        packed=context.pack_history(3,keys)
        pk,pv,means,lengths,width=packed
        self.assertEqual(width,64)
        self.assertEqual(lengths.tolist(),[[4,4,4]])
        self.assertEqual(context.history_block_ids,[[(0,'camera_0'),(0,'camera_1'),(0,'camera_2')]])
        entry=context.history[0][0]
        self.assertEqual(entry.layers[3][0].shape[-2],12)
        for camera in range(3):
            begin=25+camera*4
            expected=keys[0,:,begin:begin+4]
            torch.testing.assert_close(pk[0,:,camera*64:camera*64+4],expected)
            torch.testing.assert_close(pv[0,:,camera*64:camera*64+4],values[0,:,begin:begin+4])
            torch.testing.assert_close(means[0,:,camera],expected.detach().mean(1))
            self.assertEqual(int(pk[0,:,camera*64+4:(camera+1)*64].count_nonzero()),0)
        self.assertEqual(context.stats()['moba_history_frames'],1)
        self.assertEqual(context.stats()['moba_history_images'],3)
        self.assertEqual(context.stats()['moba_topk_effective'],3)
        for tensor in [*entry.layers[3],entry.image_means[3],pk,pv,means]:
            self.assertFalse(tensor.requires_grad);self.assertIsNone(tensor.grad_fn)

    def test_capacity_and_resets_still_count_physical_times(self):
        memory=FrameKVMemory([3],block_unit='image',max_history_frames=2)
        for t in [0,50,100]:self._capture(memory,t);memory.commit()
        self.assertEqual(list(memory.slots[0].frames),[50,100])
        context,keys,_=self._capture(memory,150)
        self.assertEqual(len(context.history_block_ids[0]),6)
        self.assertEqual(context.stats()['moba_topk_effective'],4)
        self.assertEqual(context.pack_history(3,keys)[-2].tolist(),[[4]*6])
        memory.discard();memory.reset_by_mask(torch.tensor([True]))
        self.assertEqual(list(memory.slots[0].frames),[])

    def test_top4_is_per_query_head_and_never_expands_to_other_views(self):
        width,dim=64,16
        keys=torch.zeros(1,1,6,width,dim)
        keys[0,0,:,:,0]=torch.tensor([8,7,6,5,4,3])[:,None]
        keys[0,0,:,:,1]=torch.arange(6)[:,None]
        hk=keys.reshape(1,1,6*width,dim)
        hv=torch.randn_like(hk)
        means=keys.float().mean(-2)
        query=torch.zeros(1,2,2,dim,requires_grad=True)
        with torch.no_grad():
            query[0,0,0,0]=1;query[0,0,1,1]=1
            query[0,1,0,1]=1;query[0,1,1,0]=1
        valid=torch.ones(1,2,dtype=torch.bool);lengths=torch.full((1,6),width)
        route=route_frames(query,means,lengths,4,valid)
        self.assertEqual(set(route[0,0,0].tolist()),{0,1,2,3})
        self.assertEqual(set(route[0,0,1].tolist()),{2,3,4,5})
        torch.testing.assert_close(route[0,0,1],route[0,1,0])
        ck=torch.randn(1,1,2,dim,requires_grad=True);cv=torch.randn_like(ck,requires_grad=True)
        def run(values):
            return sparse_frame_attention(query,ck,cv,(hk,values,means,lengths,width),valid,
                                          history_query_mask=valid,topk=4,backend='dense_reference')[0]
        output=run(hv);changed=hv.clone();changed[:,:,4*width:]+=1000
        alternate=run(changed)
        torch.testing.assert_close(output[0,0,0],alternate[0,0,0],atol=0,rtol=0)
        self.assertGreater(float((output[0,1,0]-alternate[0,1,0]).abs().max()),1)
        output.square().mean().backward()
        for tensor in [query,ck,cv]:self.assertTrue(bool(torch.isfinite(tensor.grad).all()))

    def test_worker_and_raw_specs_match_after_memory_token_insertion(self):
        states=[];endpoints=[];cache=[]
        for worker in [False,True]:
            torch.manual_seed(123);policy,_=tiny_vision_policy()
            policy._initialize_moba(dict(block_unit='image',topk_history_images=4,attention_backend='dense_reference'))
            inputs=tiny_image_inputs([2]);state=None
            for timestep in [0,50,100]:
                examples=[sample(timestep)]
                policy._moba_examples=examples
                policy._moba_frame_specs=build_frame_specs(examples,inputs['input_ids'],inputs['image_grid_thw'],image_token_id=100,deduplicate=False) if worker else None
                hidden,state=policy._forward_qwen_with_state(inputs,state)
                context=policy.moba_memory.pending
                if timestep==100:
                    self.assertEqual(len(context.history_block_ids[0]),6)
                    self.assertEqual(next(iter(context.route_indices.values())).shape[-1],4)
                policy.commit_moba_memory()
            endpoints.append(hidden);states.append(state)
            cache.append([entry.layers[3][0] for entry in policy.moba_memory.slots[0].frames.values()])
        torch.testing.assert_close(endpoints[0],endpoints[1],atol=0,rtol=0)
        torch.testing.assert_close(states[0],states[1],atol=0,rtol=0)
        for a,b in zip(*cache):torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_frame_time_reference_and_cold_image_path_preserve_original_outputs(self):
        original=importlib.import_module('qwen_35_version.raw_qwen35_mix_current_kv_action_memory64_all_tokens_zero_state_gate_image_topk.tests.reference_gate.tests.helpers')
        reference,backbone=original.tiny_vision_policy()
        reference._initialize_moba(dict(attention_backend='dense_reference'))
        inputs=tiny_image_inputs([2])
        for unit in ['frame_time','image']:
            policy,model=tiny_vision_policy();model.load_state_dict(backbone.state_dict());policy.load_state_dict(reference.state_dict())
            settings=dict(block_unit=unit,attention_backend='dense_reference')
            settings['topk_history_images' if unit=='image' else 'topk_history_frames']=4
            policy._initialize_moba(settings);reference.reset_memory();state=expected=None
            with torch.no_grad():
                for t in ([0,50,100] if unit=='frame_time' else [0]):
                    actual,state=forward(policy,inputs,[sample(t)],state)
                    target,expected=original.forward(reference,inputs,[original.sample(t)],expected)
                    torch.testing.assert_close(actual,target,atol=0,rtol=0)
                    torch.testing.assert_close(state,expected,atol=0,rtol=0)
                    policy.commit_moba_memory();reference.commit_moba_memory()


class ImageNativeHybridTests(NativeHybridTests):
    def setUp(self):
        torch.manual_seed(42)
        self.policy,self.backbone=tiny_vision_policy()
        self.policy._initialize_moba(dict(block_unit='image',topk_history_images=4,attention_backend='dense_reference'))
        self.inputs=tiny_image_inputs([2])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',action='store_true');parser.add_argument('--output');args=parser.parse_args()
    torch.set_num_threads(1)
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls) for cls in [ImageRoutingTests,ImageNativeHybridTests])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)
    checks=[attention_check(width=64)]
    if args.gpu:
        for n in [3,6,9]:checks.append(attention_check('cuda',n,torch.bfloat16,width=64))
    report={'passed':True,'tests':result.testsRun,'block_unit':'image','topk_history_images':4,'attention_checks':checks}
    print(json.dumps(report,indent=2))
    if args.output:Path(args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
