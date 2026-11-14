#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PACKAGE_ROOT="${REPO_ROOT}/qwen_35_version/real_qwen35_final"
STARVLA_ENV="${STARVLA_ENV:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlm-hlsc/hadoop-mlm/zhoujiacong/envs/starvla}"
CONFIG_YAML="${CONFIG_YAML:-${PACKAGE_ROOT}/config/train.yaml}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
ZERO_STAGE="${ZERO_STAGE:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-38985}"
RUN_ID="${RUN_ID:-real_piper_qwen35_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-${REPO_ROOT}/results/real-qwen35-runs}"
[[ "${RUN_ID}" != */* && -n "${RUN_ID}" ]] || { echo 'Invalid RUN_ID' >&2; exit 1; }
[[ "${ZERO_STAGE}" == 1 || "${ZERO_STAGE}" == 2 ]] || { echo 'ZERO_STAGE must be 1 or 2' >&2; exit 1; }
export PIPER_DATASET="${PIPER_DATASET:-${REPO_ROOT}/data/Pick_block_easy}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${STARVLA_ENV}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 STARVLA_VIDEO_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 EVENTVLA_SINGLE_NODE_INIT=1
export REAL_PROCESS_GROUP_PORT="${REAL_PROCESS_GROUP_PORT:-$((MAIN_PROCESS_PORT + 1))}"
export REAL_QWEN35_FINAL_CACHE_BASE="${REAL_QWEN35_FINAL_CACHE_BASE:-/tmp/real_qwen35_final/${RUN_ID}}"
export TRITON_CACHE_DIR="${REAL_QWEN35_FINAL_CACHE_BASE}/launcher/triton"
export TORCHINDUCTOR_CACHE_DIR="${REAL_QWEN35_FINAL_CACHE_BASE}/launcher/inductor"
cd "${REPO_ROOT}"
command=("${STARVLA_ENV}/bin/python" -m accelerate.commands.launch
  --config_file "${PACKAGE_ROOT}/config/deepspeed_zero${ZERO_STAGE}.yaml"
  --main_process_ip 127.0.0.1 --main_process_port "${MAIN_PROCESS_PORT:-38985}"
  --rdzv_backend static --same_network --num_processes "${NUM_PROCESSES}"
  --module qwen_35_version.real_qwen35_final.training.train
  --config_yaml "${CONFIG_YAML}"
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE:-2}"
  --trainer.max_train_steps "${MAX_TRAIN_STEPS:-80000}"
  --run_root_dir "${RUN_ROOT_DIR}" --run_id "${RUN_ID}")
if [[ -n "${INIT_CHECKPOINT:-}" ]]; then
  command+=(--trainer.init_checkpoint "${INIT_CHECKPOINT}")
fi
command+=("$@")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${command[@]}"; printf '\n'
  exit 0
fi
[[ -f "${PIPER_DATASET}/meta/info.json" ]] || { echo "Missing Piper dataset: ${PIPER_DATASET}" >&2; exit 1; }
mkdir -p "${RUN_ROOT_DIR}"
mkdir "${RUN_ROOT_DIR}/${RUN_ID}"
set +e
"${command[@]}" 2>&1 | tee "${RUN_ROOT_DIR}/${RUN_ID}/train.log"
status=${PIPESTATUS[0]}
set -e
printf '%s\n' "${status}" > "${RUN_ROOT_DIR}/${RUN_ID}/exit_code"
exit "${status}"
