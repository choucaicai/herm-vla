from __future__ import annotations

import argparse
import json
import time
import unittest
from unittest import mock
from pathlib import Path
import warnings

import torch
from torch.nn import functional as F

from qwen_35_version.raw_qwen35_mix_current_kv_action_memory64_all_tokens_zero_state_gate_image_topk.data.frame_tokens import build_frame_specs
from qwen_35_version.raw_qwen35_mix_current_kv_action_memory64_all_tokens_zero_state_gate_image_topk.model.frame_memory import (
    FrameEntry, FrameKVMemory, MemoryContext,
)
from qwen_35_version.raw_qwen35_mix_current_kv_action_memory64_all_tokens_zero_state_gate_image_topk.model.moba_attention import (
    _filter_legacy_dynamo_config_warning, sparse_frame_attention,
)


def metadata(include_initial=True):
    result = []
    roles = [("history", -30, None), ("history", -15, None), ("current", 0, None)]
    if include_initial:
        roles.insert(0, ("first", None, 0))
    for group, (role, delta, absolute) in enumerate(roles):
        for camera in range(3):
            result.append({
                "time_index": group, "time_role": role, "delta": delta,
                "absolute_index": absolute, "view": f"camera_{camera}",
            })
    return result


class FrameTests(unittest.TestCase):
    def test_relative_only_frames_and_clipped_deduplication(self):
        image_id = 248056
        ids = torch.tensor([[0, 99] + [image_id] * 36 + [10]])
        grid = torch.tensor([[1, 4, 4]] * 9)
        sample = {"timestep": 50, "image_metas": metadata(include_initial=False)}
        groups = build_frame_specs([sample], ids, grid)[0]
        self.assertEqual([group["frame_timestep"] for group in groups], [20, 35, 50])
        self.assertEqual([group["delta"] for group in groups], [-30, -15, 0])
        self.assertEqual([group["token_indices"] for group in groups], [
            list(range(2, 14)), list(range(14, 26)), list(range(26, 38)),
        ])
        sample["timestep"] = 0
        groups = build_frame_specs([sample], ids, grid)[0]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["frame_timestep"], 0)
        self.assertEqual(groups[0]["delta"], 0)
        self.assertEqual(groups[0]["token_indices"], list(range(26, 38)))

    def test_initial_role_exclusion_and_clipped_frame_deduplication(self):
        image_id = 248056
        ids = torch.tensor([[0, 99] + [image_id] * 48 + [10]])
        grid = torch.tensor([[1, 4, 4]] * 12)
        sample = {"timestep": 0, "image_metas": metadata()}
        groups = build_frame_specs([sample], ids, grid)[0]
        self.assertEqual([group["frame_timestep"] for group in groups], [0])
        self.assertEqual(groups[0]["delta"], 0)
        self.assertEqual(groups[0]["token_indices"], list(range(38, 50)))
        sample["timestep"] = 50
        groups = build_frame_specs([sample], ids, grid)[0]
        self.assertEqual([group["frame_timestep"] for group in groups], [20, 35, 50])
        self.assertEqual([len(group["token_indices"]) for group in groups], [12] * 3)
        self.assertTrue(all(min(group["token_indices"]) >= 14 for group in groups))

    def test_transaction_detachment_slot_reset_and_repeat_time(self):
        memory = FrameKVMemory([3, 7])
        samples = [
            {"episode_id": "a", "timestep": 10},
            {"episode_id": "b", "timestep": 20},
        ]
        memory.prepare_slots(samples)
        specs = [[{"frame_timestep": sample["timestep"], "token_indices": [1, 2, 3]}]
                 for sample in samples]
        valid = torch.ones(2, 8, dtype=torch.bool)
        context = MemoryContext(memory, samples, specs, valid, [12, 12])
        keys = torch.randn(2, 4, 8, 256, requires_grad=True)
        values = torch.randn_like(keys, requires_grad=True)
        context.capture(3, keys, values)
        with self.assertRaises(RuntimeError):
            memory.stage(context)
        context.capture(7, keys, values)
        memory.stage(context)
        self.assertEqual(len(memory.slots[0].frames), 0)
        memory.commit()
        self.assertEqual(list(memory.slots[0].frames), [10])
        cached_key, cached_value, _ = memory.slots[0].frames[10].layers[3]
        for cached, source in [(cached_key, keys), (cached_value, values)]:
            self.assertFalse(cached.requires_grad)
            self.assertIsNone(cached.grad_fn)
            self.assertNotEqual(cached.untyped_storage().data_ptr(), source.untyped_storage().data_ptr())
        expected_key, expected_value = cached_key.clone(), cached_value.clone()
        with torch.no_grad():
            keys.add_(1)
            values.add_(1)
        torch.testing.assert_close(cached_key, expected_key, atol=0, rtol=0)
        torch.testing.assert_close(cached_value, expected_value, atol=0, rtol=0)
        memory.reset_by_mask(torch.tensor([True, False]), ["new-a", "b"])
        self.assertEqual(len(memory.slots[0].frames), 0)
        self.assertEqual(list(memory.slots[1].frames), [20])
        samples = [{"episode_id": "new-a", "timestep": 30},
                   {"episode_id": "b", "timestep": 40}]
        memory.prepare_slots(samples)
        specs = [[{"frame_timestep": 25, "token_indices": [1, 2, 3]}],
                 [{"frame_timestep": 20, "token_indices": [1, 2, 3]}]]
        context = MemoryContext(memory, samples, specs, valid, [40, 40])
        self.assertEqual(len(context.new_frames[1]), 0)
        for layer in [3, 7]:
            context.capture(layer, keys, values)
        memory.stage(context)
        memory.commit()
        self.assertEqual(list(memory.slots[1].frames), [20])
        with self.assertRaises(ValueError):
            memory.prepare_slots(samples)

    def test_pack_history_preserves_variable_widths_and_empty_slots(self):
        torch.manual_seed(19)
        context = object.__new__(MemoryContext)
        widths = [[], [1, 65], [], [127, 192, 64]]
        context.history = []
        for slot_widths in widths:
            entries = []
            for frame, width in enumerate(slot_widths):
                key = torch.randn(2, width, 8, dtype=torch.bfloat16)
                value = torch.randn_like(key)
                entry = FrameEntry(frame)
                entry.layers[3] = (key, value, key.float().mean(dim=1))
                entries.append(entry)
            context.history.append(entries)
        reference = torch.empty(4, 2, 11, 8, dtype=torch.bfloat16)
        keys, values, means, lengths, width = context.pack_history(3, reference)
        self.assertEqual(width, 192)
        self.assertEqual(tuple(keys.shape), (4, 2, 576, 8))
        self.assertEqual(lengths.tolist(), [[0, 0, 0], [1, 65, 0], [0, 0, 0], [127, 192, 64]])
        self.assertEqual(means.dtype, torch.float32)
        for sample, entries in enumerate(context.history):
            for frame in range(3):
                offset = frame * width
                if frame < len(entries):
                    key, value, mean = entries[frame].layers[3]
                    count = key.shape[1]
                    torch.testing.assert_close(keys[sample, :, offset:offset + count], key, atol=0, rtol=0)
                    torch.testing.assert_close(values[sample, :, offset:offset + count], value, atol=0, rtol=0)
                    torch.testing.assert_close(means[sample, :, frame], mean, atol=0, rtol=0)
                else:
                    count = 0
                    self.assertEqual(int(means[sample, :, frame].count_nonzero()), 0)
                self.assertEqual(int(keys[sample, :, offset + count:offset + width].count_nonzero()), 0)
                self.assertEqual(int(values[sample, :, offset + count:offset + width].count_nonzero()), 0)
        context.history = [[], [], [], []]
        self.assertIsNone(context.pack_history(3, reference))

    def test_warning_filter_is_specific_to_empty_legacy_config(self):
        message = "Skipping serialization of skipfiles_inline_module_allowlist value {}"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _filter_legacy_dynamo_config_warning()
            for text, module in [
                (message, "torch.utils._config_module"),
                ("An unrelated compiler warning", "torch.utils._config_module"),
                (message.replace("{}", "{'custom': 1}"), "torch.utils._config_module"),
                (message, "other_module"),
            ]:
                warnings.warn_explicit(text, UserWarning, "_config_module.py", 342, module=module)
        self.assertEqual(len(caught), 3)
        self.assertEqual(str(caught[0].message), "An unrelated compiler warning")


def attention_check(device="cpu", frame_count=6, dtype=torch.float32, width=192):
    torch.manual_seed(71 + frame_count)
    batch, heads, kv_heads, sequence, dim = 2, 16, 4, 97, 256
    query = torch.randn(batch, heads, sequence, dim, device=device, dtype=dtype, requires_grad=True)
    key = torch.randn(batch, kv_heads, sequence, dim, device=device, dtype=dtype, requires_grad=True)
    value = torch.randn_like(key, requires_grad=True)
    history_key = torch.randn(batch, kv_heads, frame_count * width, dim, device=device, dtype=dtype)
    history_value = torch.randn_like(history_key)
    lengths = torch.full((batch, frame_count), width, device=device, dtype=torch.long)
    lengths[1] = 0
    means = history_key.reshape(batch, kv_heads, frame_count, width, dim).float().mean(-2)
    valid = torch.ones(batch, sequence, device=device, dtype=torch.bool)
    valid[1, :7] = False
    enabled_queries = valid.clone()
    packed = history_key, history_value, means, lengths, width
    backend = "flex" if device == "cuda" else "dense_reference"
    started = time.monotonic()
    output, route = sparse_frame_attention(
        query, key, value, packed, valid, topk=4, backend=backend,
        history_query_mask=enabled_queries,
    )
    # Independently enumerate each selected time and the causal current keys.
    mask = torch.zeros(batch, heads, sequence, frame_count * width + sequence,
                       device=device, dtype=torch.bool)
    for b in range(batch):
        for h in range(heads):
            for q in range(sequence):
                for frame in route[b, h, q].tolist():
                    if frame < 0:
                        continue
                    count = int(lengths[b, frame])
                    mask[b, h, q, frame * width:frame * width + count] = True
                mask[b, h, q, frame_count * width:frame_count * width + q + 1] = valid[b, :q + 1]
                if not bool(valid[b, q]):
                    mask[b, h, q, frame_count * width + q] = True
    qr = query.detach().clone().requires_grad_()
    kr = key.detach().clone().requires_grad_()
    vr = value.detach().clone().requires_grad_()
    reference = F.scaled_dot_product_attention(
        qr, torch.cat([history_key, kr], -2), torch.cat([history_value, vr], -2),
        attn_mask=mask, enable_gqa=True,
    ) * valid[:, None, :, None]
    weight = torch.randn_like(output)
    (output.float() * weight.float()).mean().backward()
    (reference.float() * weight.float()).mean().backward()
    tolerance = 0.015 if dtype == torch.bfloat16 else 2e-5
    torch.testing.assert_close(output, reference, atol=tolerance, rtol=0.03)
    grad_errors = []
    for actual, expected in [(query.grad, qr.grad), (key.grad, kr.grad), (value.grad, vr.grad)]:
        if not torch.isfinite(actual).all():
            raise AssertionError("Non-finite sparse attention gradient")
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=0.08)
        grad_errors.append(float((actual - expected).abs().max()))
    if device == "cuda":
        torch.cuda.synchronize()
    return {
        "backend": backend, "frame_count": frame_count,
        "topk": route.shape[-1], "max_output_error": float((output - reference).abs().max()),
        "max_gradient_errors": grad_errors, "seconds": time.monotonic() - started,
        "empty_slot_and_left_padding": True,
        "history_query_scope": "all_tokens",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.15)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.gpu:
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FrameTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    records = [attention_check()]
    if args.gpu:
        for frames in [1, 6, 9]:
            records.append(attention_check("cuda", frames, torch.bfloat16))
    summary = {"passed": True, "attention_checks": records}
    print(json.dumps(summary, indent=2), flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
