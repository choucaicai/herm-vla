#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
checkpoint="${1:-${POLICY_CKPT_PATH:-}}"
if [[ -z "${checkpoint}" ]]; then
  echo "Usage: $0 <checkpoint> [gpu=0] [port=7402]" >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${2:-${GPU_ID:-0}}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
args=(--ckpt_path "${checkpoint}" --port "${3:-${PORT:-7402}}")
if [[ "${USE_BF16:-1}" == 1 ]]; then args+=(--use_bf16); fi
exec "${EVENTVLA_PYTHON:-python}" -m qwen_35_version.raw_qwen35_mix_current_kv_action_memory64_all_tokens_zero_state_gate_image_topk.eval.server_policy "${args[@]}"
