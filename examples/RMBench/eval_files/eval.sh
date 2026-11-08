#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
task="${1:?Usage: eval.sh <task> [task_config] [ckpt_setting] [seed] [gpu]}"
exec bash "${ROOT}/scripts/eval_final.sh"   --tasks "${task}" --task-config "${2:-demo_clean}"   --ckpt-setting "${3:-herm_vla_final}" --seed "${4:-0}" --gpus "${5:-0}"   --checkpoint "${POLICY_CKPT_PATH:?Set POLICY_CKPT_PATH to the trained checkpoint}"
