#!/usr/bin/env bash
set -x -e

cd /home/shadeform/iCode/VLA/Isaac-GR00T

BASE_MODEL_PATH=${BASE_MODEL_PATH:-/home/shadeform/iDataset/VLA/gr00t/GR00T-N1.6-3B}
PREPROCESSED_ROOT=${PREPROCESSED_ROOT:-/home/shadeform/iDataset/simulation/task_suite/instruction_preprocessed}

NUM_GPUS=${NUM_GPUS:-8}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
USE_COLOR_JITTER=${USE_COLOR_JITTER:-0}
COLOR_JITTER_PARAMS=${COLOR_JITTER_PARAMS:-"brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08"}

if [[ "${USE_COLOR_JITTER}" == "1" ]]; then
  COLOR_JITTER_TAG="color_jitter_on"
else
  COLOR_JITTER_TAG="color_jitter_off"
fi

DATE_HOUR=$(date +%Y%m%d_%H)
OUTPUT_DIR=${OUTPUT_DIR:-/ephemeral/gr00t_models/instruction_${COLOR_JITTER_TAG}_${DATE_HOUR}}

export NUM_GPUS
export CUDA_VISIBLE_DEVICES

CMD=(
  uv run --python 3.10 torchrun --nproc_per_node="${NUM_GPUS}" --master_port=29500 gr00t/experiment/launch_finetune.py
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
  --dataloader_num_workers 8
)

if [[ "${USE_COLOR_JITTER}" == "1" ]]; then
  # shellcheck disable=SC2206
  CJ_ARGS=( ${COLOR_JITTER_PARAMS} )
  CMD+=(--color_jitter_params "${CJ_ARGS[@]}")
fi

"${CMD[@]}"

