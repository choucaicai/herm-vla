#!/usr/bin/env bash
# Exact final-model component ablation: carry 64Q Gate M; no historical visual KV.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CONFIG_YAML="${REPO_ROOT}/qwen_35_version/raw_qwen35_mix_current_kv_action_memory64_all_tokens_zero_state_gate_image_topk/config/train_m_only.yaml"
printf '%s\n' 'Ablation=M-only; 64Q zero-state slot Gate; native S resets per decision; historical KV writes/reads=0.'
exec bash "${REPO_ROOT}/scripts/train_final.sh" "$@"
