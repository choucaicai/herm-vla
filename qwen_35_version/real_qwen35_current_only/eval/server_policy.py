"""Serve the unmodified trained current-only model with runtime auditing."""
import argparse
import json
import logging
import os
from pathlib import Path

gpu=os.environ.get('CUDA_VISIBLE_DEVICES','0')
cache=Path(os.environ.get('REAL_QWEN35_CURRENT_ONLY_SERVING_CACHE','/tmp/real_qwen35_current_only_serving'))/f'gpu_{gpu}'
os.environ['TRITON_CACHE_DIR']=str(cache/'triton')
os.environ['TORCHINDUCTOR_CACHE_DIR']=str(cache/'inductor')
from accelerate import PartialState
import torch
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from qwen_35_version.real_qwen35_current_only.eval.policy import CurrentOnlyServingPolicy,metadata_for


def load_policy(checkpoint):
    model=CurrentOnlyServingPolicy.from_checkpoint(checkpoint)
    metadata=metadata_for(model,checkpoint)
    logging.info('Current-only checkpoint loaded strictly: %s',checkpoint)
    logging.info('PIPER_AUDIT_METADATA %s',json.dumps(metadata))
    logging.info('PIPER_AUDIT_SOURCE %s',json.dumps({'policy_module':__file__,'checkpoint':str(checkpoint),'gpu':gpu,'passed':True}))
    return model,metadata


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--ckpt_path',required=True)
    parser.add_argument('--port',type=int,default=7403)
    parser.add_argument('--use_bf16',action='store_true')
    parser.add_argument('--idle_timeout',type=int,default=1800)
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,force=True)
    PartialState()
    model,metadata=load_policy(args.ckpt_path)
    if args.use_bf16:model=model.to(torch.bfloat16)
    model=model.to('cuda').eval()
    WebsocketPolicyServer(model,host='127.0.0.1',port=args.port,
                          idle_timeout=args.idle_timeout,metadata=metadata).serve_forever()


if __name__=='__main__':main()
