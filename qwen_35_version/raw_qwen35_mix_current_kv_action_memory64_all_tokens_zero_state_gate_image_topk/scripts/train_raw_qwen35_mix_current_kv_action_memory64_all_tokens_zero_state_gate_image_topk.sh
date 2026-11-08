#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../" && pwd)"
SOURCE_ROOT="${RAW_QWEN35_MIX_CURRENT_KV_ACTION_MEMORY64_ALL_TOKENS_ZERO_STATE_GATE_IMAGE_TOPK_SOURCE_ROOT:-${REPO_ROOT}}"
EVENTVLA_ROOT="${EVENTVLA_ROOT:-${REPO_ROOT}}"
STARVLA_ENV="${STARVLA_ENV:-${HOME}/miniconda3/envs/starvla}"
PACKAGE_ROOT="${SOURCE_ROOT}/qwen_35_version/raw_qwen35_mix_current_kv_action_memory64_all_tokens_zero_state_gate_image_topk"
CONFIG_YAML="${CONFIG_YAML:-${PACKAGE_ROOT}/config/train.yaml}"
NUM_PROCESSES=${NUM_PROCESSES:-8}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
ZERO_STAGE=${ZERO_STAGE:-1}
RUN_ROOT_DIR="${RUN_ROOT_DIR:-${SOURCE_ROOT}/results/qwen35-runs}"
RUN_ID=${RUN_ID:-rmbench9_raw_qwen35_mix_current_kv_action_memory64_all_tokens_zero_state_gate_image_topk_memory_queries64_zero${ZERO_STAGE}_b${PER_DEVICE_BATCH_SIZE}_$(date +%Y%m%d_%H%M%S)}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-80000}
SAVE_CHECKPOINTS=${SAVE_CHECKPOINTS:-true}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-38975}
if [[ "${ZERO_STAGE}" != 1 && "${ZERO_STAGE}" != 2 ]]; then
  printf 'ZERO_STAGE must be 1 or 2\n' >&2
  exit 1
fi
if [[ "${RUN_ID}" == */* ]]; then
  printf 'RUN_ID must be a directory name\n' >&2
  exit 1
fi
if [[ ! -f "${CONFIG_YAML}" ]]; then
  printf 'Training configuration not found: %s\n' "${CONFIG_YAML}" >&2
  exit 1
fi

export PYTHONPATH="${SOURCE_ROOT}:${EVENTVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${STARVLA_ENV}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 STARVLA_VIDEO_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 EVENTVLA_SINGLE_NODE_INIT=1
export RAW_QWEN35_MIX_CURRENT_KV_ACTION_MEMORY64_ALL_TOKENS_ZERO_STATE_GATE_IMAGE_TOPK_CACHE_BASE="${RAW_QWEN35_MIX_CURRENT_KV_ACTION_MEMORY64_ALL_TOKENS_ZERO_STATE_GATE_IMAGE_TOPK_CACHE_BASE:-/tmp/eventvla_raw_qwen35_mix_current_kv_action_memory64_all_tokens_zero_state_gate_image_topk/${RUN_ID}}"
export TRITON_CACHE_DIR="${RAW_QWEN35_MIX_CURRENT_KV_ACTION_MEMORY64_ALL_TOKENS_ZERO_STATE_GATE_IMAGE_TOPK_CACHE_BASE}/launcher/triton"
export TORCHINDUCTOR_CACHE_DIR="${RAW_QWEN35_MIX_CURRENT_KV_ACTION_MEMORY64_ALL_TOKENS_ZERO_STATE_GATE_IMAGE_TOPK_CACHE_BASE}/launcher/inductor"
cd "${SOURCE_ROOT}"

command=(
  "${STARVLA_ENV}/bin/python" -m accelerate.commands.launch
  --config_file "${PACKAGE_ROOT}/config/deepspeed_zero${ZERO_STAGE}.yaml"
  --main_process_ip 127.0.0.1
  --main_process_port "${MAIN_PROCESS_PORT}"
  --rdzv_backend static --same_network --num_processes "${NUM_PROCESSES}"
  --module qwen_35_version.raw_qwen35_mix_current_kv_action_memory64_all_tokens_zero_state_gate_image_topk.training.train
  --config_yaml "${CONFIG_YAML}"
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}"
  --trainer.enable_gradient_checkpointing false
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}"
  --trainer.save_checkpoints "${SAVE_CHECKPOINTS}"
  --run_root_dir "${RUN_ROOT_DIR}" --run_id "${RUN_ID}" "$@"
)

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
mkdir -p "${RUN_ROOT_DIR}"
mkdir -p "${RUN_ROOT_DIR}/${RUN_ID}"
printf 'run_id=%s, zero_stage=%s, batch=%s, slots=%s, decisions_per_update=%s, tbptt=2, state=memory_queries64+moba_top4_images, config=%s, stride=50, gradient_checkpointing=false, steps=%s\n' "${RUN_ID}" "${ZERO_STAGE}" "${PER_DEVICE_BATCH_SIZE}" "$((NUM_PROCESSES * PER_DEVICE_BATCH_SIZE))" "$((NUM_PROCESSES * PER_DEVICE_BATCH_SIZE * 2))" "${CONFIG_YAML}" "${MAX_TRAIN_STEPS}"
set +e
"${command[@]}" 2>&1 | tee "${RUN_ROOT_DIR}/${RUN_ID}/train.log"
status=${PIPESTATUS[0]}
set -e
printf '%s\n' "${status}" > "${RUN_ROOT_DIR}/${RUN_ID}/exit_code"
exit "${status}"
