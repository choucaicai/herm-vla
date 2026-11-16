"""Strict-load a saved GR00T checkpoint and execute a serving decision."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from accelerate import PartialState

from ..eval.policy import RecurrentServingPolicy
from .validate_real import build_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    PartialState()
    model = RecurrentServingPolicy.from_checkpoint(args.checkpoint).to('cuda', dtype=torch.bfloat16).eval()
    samples, _ = build_samples(model.config)
    a = model.predict_action(samples[0], episode_id='checkpoint-check', decision_id=0)
    b = model.predict_action(samples[0], episode_id='checkpoint-check', decision_id=0)
    assert a['normalized_actions'].shape == (1, 50, 14)
    assert np.isfinite(a['normalized_actions']).all()
    np.testing.assert_array_equal(a['normalized_actions'], b['normalized_actions'])
    report = {'passed': True, 'checkpoint': args.checkpoint, 'strict_load': True,
              'shape': list(a['normalized_actions'].shape), 'duplicate_request_identical': True,
              'memory_frames_after_duplicate': len(model.moba_memory.slots[0].frames)}
    assert report['memory_frames_after_duplicate'] == 1
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
