"""Validate current-only Piper checkpoint configuration, shapes and normalization."""

import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch

from eventvla.model.tools import read_mode_config
from ..model.eventvla_qwen35_mix import validate_config
from ..model.normalization import unnormalize_actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    path = Path(args.checkpoint).resolve()
    config, stats = read_mode_config(path)
    validate_config(OmegaConf.create(config))
    state = torch.load(path, map_location='meta', weights_only=True, mmap=True)
    if not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError('Expected a tensor state dictionary')
    if any(name.startswith(('query_memory.', 'moba_memory.')) for name in state):
        raise ValueError('A current-only checkpoint must not contain memory parameters')
    for name, shape in {'action_model.model.fc2.weight': (7, 5120),
                        'action_model.model.fc2.bias': (7,)}.items():
        if name not in state or tuple(state[name].shape) != shape:
            raise ValueError(f'Invalid action parameter: {name}')
    if not any(name.startswith('qwen_vl_interface.') for name in state):
        raise ValueError('Missing Qwen backbone')
    unnormalize_actions(np.zeros((1, 7)), stats['new_embodiment']['action'])
    print(json.dumps({'passed': True, 'checkpoint': str(path), 'images_per_sample': 2,
                      'memory_parameters': 0, 'tensor_count': len(state),
                      'strict_load': 'enforced by serving policy'}, indent=2))


if __name__ == '__main__':
    main()
