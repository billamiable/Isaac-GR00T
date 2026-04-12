#!/usr/bin/env bash
set -x -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ENV_SH="${TRAINING_ENV_SH:-${SCRIPT_DIR}/training_paths.sh}"
if [[ -f "${ENV_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_SH}"
fi

BASE_MODEL_PATH=${BASE_MODEL_PATH:-/ephemeral/models/GR00T-N1.6-3B}
PREPROCESSED_ROOT=${PREPROCESSED_ROOT:-/ephemeral/agibot/instruction_preprocessed}

NUM_GPUS=${NUM_GPUS:-8}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}
MASTER_PORT=${MASTER_PORT:-29500}
USE_COLOR_JITTER=${USE_COLOR_JITTER:-0}
COLOR_JITTER_PARAMS=${COLOR_JITTER_PARAMS:-"brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08"}

check_cuda_ready() {
  local expected_gpus="$1"
  local py_exec="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
  local probe

  if [[ ! -x "${py_exec}" ]]; then
    echo "[ERROR] Python executable not found: ${py_exec}" >&2
    echo "[HINT] Run 'uv sync' in ${REPO_ROOT} or set PYTHON_BIN explicitly." >&2
    exit 1
  fi

  probe=$("${py_exec}" -c "import torch; print(int(torch.cuda.is_available()), torch.cuda.device_count())")
  local is_available
  local device_count
  is_available="$(echo "${probe}" | awk '{print $1}')"
  device_count="$(echo "${probe}" | awk '{print $2}')"

  if [[ "${is_available}" != "1" || "${device_count}" -lt "${expected_gpus}" ]]; then
    echo "[ERROR] CUDA preflight failed: is_available=${is_available}, device_count=${device_count}, expected>=${expected_gpus}" >&2
    echo "[HINT] Try resetting NVIDIA UVM and rerun:" >&2
    echo "  sudo modprobe -r nvidia_uvm && sudo modprobe nvidia_uvm" >&2
    echo "  ./.venv/bin/python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.device_count())\"" >&2
    exit 1
  fi
}

if [[ "${USE_COLOR_JITTER}" == "1" ]]; then
  COLOR_JITTER_TAG="color_jitter_on"
else
  COLOR_JITTER_TAG="color_jitter_off"
fi

DATE_HOUR=$(date +%Y%m%d_%H)
OUTPUT_ROOT=${OUTPUT_ROOT:-/ephemeral/gr00t_models}
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/instruction_${COLOR_JITTER_TAG}_${DATE_HOUR}}

export NUM_GPUS
export CUDA_VISIBLE_DEVICES

if [[ ! -d "${BASE_MODEL_PATH}" ]]; then
  echo "[ERROR] Base model directory not found: ${BASE_MODEL_PATH}" >&2
  exit 1
fi

for task_name in \
  pick_block_color \
  pick_block_number \
  pick_block_shape \
  pick_block_size \
  pick_common_sense \
  pick_object_type \
  pick_specific_object \
  straighten_object \
  pick_follow_logic_or \
  pick_billards_color; do
  if [[ ! -d "${PREPROCESSED_ROOT}/${task_name}" ]]; then
    echo "[ERROR] Missing preprocessed task directory: ${PREPROCESSED_ROOT}/${task_name}" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "${OUTPUT_DIR}")"

check_cuda_ready "${NUM_GPUS}"

CMD=(
  uv run --python 3.10 torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" gr00t/experiment/launch_finetune.py
  --base_model_path "${BASE_MODEL_PATH}"
  --dataset_paths
  "${PREPROCESSED_ROOT}/pick_block_color"
  "${PREPROCESSED_ROOT}/pick_block_number"
  "${PREPROCESSED_ROOT}/pick_block_shape"
  "${PREPROCESSED_ROOT}/pick_block_size"
  "${PREPROCESSED_ROOT}/pick_common_sense"
  "${PREPROCESSED_ROOT}/pick_object_type"
  "${PREPROCESSED_ROOT}/pick_specific_object"
  "${PREPROCESSED_ROOT}/straighten_object"
  "${PREPROCESSED_ROOT}/pick_follow_logic_or"
  "${PREPROCESSED_ROOT}/pick_billards_color"
  --embodiment_tag AGIBOT_GENIE1
  --num_gpus "${NUM_GPUS}"
  --output_dir "${OUTPUT_DIR}"
  --save_steps 3000
  --save_total_limit 5
  --max_steps 30000
  --warmup_ratio 0.05
  --weight_decay 1e-5
  --learning_rate 1e-4
  --use_wandb
  --global_batch_size 512
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
)

if [[ "${USE_COLOR_JITTER}" == "1" ]]; then
  # shellcheck disable=SC2206
  CJ_ARGS=( ${COLOR_JITTER_PARAMS} )
  CMD+=(--color_jitter_params "${CJ_ARGS[@]}")
fi

"${CMD[@]}"

