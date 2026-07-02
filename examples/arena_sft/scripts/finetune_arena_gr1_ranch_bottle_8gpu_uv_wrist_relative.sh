#!/usr/bin/env bash
# GR1 ranch-bottle GR00T N1.6 SFT (ego + wrist views, relative arms and absolute hands).
#
# Required environment variables:
#   BASE_MODEL_PATH  Local GR00T-N1.6-3B checkpoint or Hugging Face model ID.
#   DATASET_PATH     GR00T-flavoured LeRobot v2 dataset root.
#   OUTPUT_DIR       Directory for checkpoints and experiment metadata.
#
# The training defaults reproduce the original 8-GPU recipe. Override runtime
# knobs (for example NUM_GPUS=1 MAX_STEPS=2) for a smoke test.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARENA_SFT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GR00T_ROOT="$(cd "${ARENA_SFT_ROOT}/../.." && pwd)"
cd "${GR00T_ROOT}"

: "${BASE_MODEL_PATH:?Set BASE_MODEL_PATH to GR00T-N1.6-3B or a Hugging Face model ID}"
: "${DATASET_PATH:?Set DATASET_PATH to the ranch-bottle LeRobot dataset root}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a writable checkpoint directory}"

MODALITY_CONFIG_PATH="${MODALITY_CONFIG_PATH:-${ARENA_SFT_ROOT}/configs/gr1_arms_only_wrist_relative_data_config.py}"
NUM_GPUS="${NUM_GPUS:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-96}"
MAX_STEPS="${MAX_STEPS:-20000}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-16}"
MASTER_PORT="${MASTER_PORT:-29500}"
USE_WANDB="${USE_WANDB:-1}"

# torchcodec needs FFmpeg shared libraries. A caller-managed conda environment
# may provide them; otherwise the uv environment uses the system libraries.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export PATH="${CONDA_PREFIX}/bin:${PATH}"
fi

WANDB_ARGS=()
if [[ "${USE_WANDB}" == "1" ]]; then
  WANDB_ARGS+=(--use-wandb)
fi

uv run python -m torch.distributed.run \
  --nproc_per_node="${NUM_GPUS}" \
  --master_port="${MASTER_PORT}" \
  --standalone \
  gr00t/experiment/launch_finetune.py \
  --base-model-path "${BASE_MODEL_PATH}" \
  --dataset-path "${DATASET_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --modality-config-path "${MODALITY_CONFIG_PATH}" \
  --embodiment-tag GR1 \
  --num-gpus "${NUM_GPUS}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --max-steps "${MAX_STEPS}" \
  --save-steps "${SAVE_STEPS}" \
  --save-total-limit "${SAVE_TOTAL_LIMIT}" \
  --no-tune-llm \
  --tune-visual \
  --tune-projector \
  --tune-diffusion-model \
  --dataloader-num-workers "${DATALOADER_NUM_WORKERS}" \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  "${WANDB_ARGS[@]}" \
  "$@"
