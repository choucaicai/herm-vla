#!/usr/bin/env bash
# Final-model retrieval-budget ablation; K counts individual camera images.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TOPK="${1:?Usage: train_topk.sh 6|8 [training overrides...]}"
shift
case "${TOPK}" in
  6|8) ;;
  *) printf 'Top-k ablations require 6 or 8\n' >&2; exit 1 ;;
esac
export CONFIG_YAML="${REPO_ROOT}/qwen_35_version/raw_qwen35_mix_current_kv_action_memory64_all_tokens_zero_state_gate_image_topk/config/train_top${TOPK}.yaml"
printf 'Final-model ablation: 64Q Gate M + per-image historical KV Top-%s.\n' "${TOPK}"
exec bash "${REPO_ROOT}/scripts/train_final.sh" "$@"
