#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${REPO_ROOT}/qwen_35_version/raw_qwen35_mix_current_kv_action_memory64_all_tokens_zero_state_gate_image_topk"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec bash "${REPO_ROOT}/scripts/eval_rmbench_v2.sh"   --model-root "${MODEL_ROOT}"   --policy-name qwen_35_version.raw_qwen35_mix_current_kv_action_memory64_all_tokens_zero_state_gate_image_topk.eval.model2robotwin_interface   --unnorm-key new_embodiment --action-mode abs   --test-num 25 --sampling-interval 50 --seed 0   --task-config demo_clean --instruction-type unseen   --swap-rb --swap-rb-tasks cover_blocks "$@"
