#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${REPO_ROOT}/qwen_35_version/raw_qwen35_mix_current_kv_action_memory64_all_tokens_zero_state_gate_image_topk/scripts/train_raw_qwen35_mix_current_kv_action_memory64_all_tokens_zero_state_gate_image_topk.sh" "$@"
