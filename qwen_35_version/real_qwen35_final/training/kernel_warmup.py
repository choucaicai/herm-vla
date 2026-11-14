"""Tune DeltaNet backward kernels before retaining full-model TBPTT graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def warm_delta_kernels(text_config, device):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    heads = int(text_config.linear_num_value_heads)
    key_dim = int(text_config.linear_key_head_dim)
    value_dim = int(text_config.linear_value_head_dim)
    shape = (1, 128, heads)
    torch.cuda.empty_cache()
    with torch.random.fork_rng(devices=[device.index]), torch.enable_grad():
        inputs = []
        for _ in range(2):
            q = torch.randn(*shape, key_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn_like(q, requires_grad=True)
            v = torch.randn(*shape, value_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
            g = (-torch.rand(*shape, device=device) * 0.05).requires_grad_()
            beta = (torch.rand(*shape, device=device, dtype=torch.bfloat16) * 0.1).requires_grad_()
            inputs.append((q, k, v, g, beta))
        output1, state1 = chunk_gated_delta_rule(
            *inputs[0][:3], g=inputs[0][3], beta=inputs[0][4],
            initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True,
        )
        state1.retain_grad()
        output2, state2 = chunk_gated_delta_rule(
            *inputs[1][:3], g=inputs[1][3], beta=inputs[1][4],
            initial_state=state1, output_final_state=True, use_qk_l2norm_in_kernel=True,
        )
        loss = output1.float().square().mean() + output2.float().square().mean()
        loss.backward()
        torch.cuda.synchronize(device)
        tensors = [value.grad for step in inputs for value in step] + [state1.grad]
        if any(value is None or not bool(torch.isfinite(value).all()) for value in tensors):
            raise RuntimeError("DeltaNet kernel warmup returned invalid gradients")
        if state1.grad.abs().sum().item() == 0:
            raise RuntimeError("CUDA DeltaNet does not propagate gradients through S")
        report = {
            "passed": True,
            "shape": [1, 128, heads, key_dim, value_dim],
            "state_gradient_l1": state1.grad.abs().sum().item(),
            "device": str(device),
        }
        del inputs, output1, output2, state1, state2, loss, tensors, q, k, v, g, beta
    torch.cuda.empty_cache()
    return report


if __name__ == "__main__":
    from transformers import AutoConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = AutoConfig.from_pretrained(args.model)
    device = torch.device("cuda", torch.cuda.current_device())
    result = warm_delta_kernels(config.get_text_config(), device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
