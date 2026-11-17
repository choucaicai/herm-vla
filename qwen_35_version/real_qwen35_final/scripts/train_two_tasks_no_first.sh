#!/usr/bin/env bash
set -euo pipefail
PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CONFIG_YAML="${CONFIG_YAML:-${PACKAGE_ROOT}/config/train_two_tasks_no_first.yaml}"
export RUN_ID="${RUN_ID:-real_piper_easy_hard_no_first_qwen35_40k_$(date +%Y%m%d_%H%M%S)}"
exec bash "${PACKAGE_ROOT}/scripts/train_two_tasks.sh" "$@"
