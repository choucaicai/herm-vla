#!/usr/bin/env bash
set -euo pipefail
PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd -- "${PACKAGE_ROOT}/../.." && pwd)"
export PIPER_DATA_ROOT="${PIPER_DATA_ROOT:-${REPO_ROOT}/data}"
export PIPER_DATASET="${PIPER_DATASET:-${PIPER_DATA_ROOT}/Pick_block_hard}"
export CONFIG_YAML="${CONFIG_YAML:-${PACKAGE_ROOT}/config/train_hard.yaml}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-40000}"
export RUN_ID="${RUN_ID:-real_piper_hard_qwen35_40k_$(date +%Y%m%d_%H%M%S)}"
exec bash "${PACKAGE_ROOT}/scripts/train_real.sh" "$@"
