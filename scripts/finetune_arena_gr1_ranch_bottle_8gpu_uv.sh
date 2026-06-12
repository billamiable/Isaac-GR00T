#!/usr/bin/env bash
# GR1 ranch-bottle GR00T N1.6 post-training (8-GPU) via uv (no Arena Docker).
# Run from host after: cd Isaac-GR00T && uv sync --python 3.10 && uv pip install -e .
# Optional: uv run wandb login
set -e
GR00T_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${GR00T_ROOT}"

# torchcodec needs FFmpeg native libs on PATH/LD_LIBRARY_PATH; DataLoader workers often fail with
# "torchcodec is not available" without this (same pattern as ACoT-VLA scripts/run_compute_norm_stats.sh).
PREFIX="${CONDA_PREFIX:-/home/billyw/miniconda3}"
export LD_LIBRARY_PATH="${PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PATH="${PREFIX}/bin:${PATH}"

# IsaacLab-Arena repo root (this file: Arena/submodules/Isaac-GR00T/scripts/… → up 3 levels)
ARENA_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
MODALITY_CONFIG_PATH="${ARENA_ROOT}/isaaclab_arena_gr00t/embodiments/gr1/gr1_arms_only_data_config.py"

BASE_MODEL_PATH=/home/billyw/iDataset/VLA/gr00t/GR00T-N1.6-3B
DATASET_PATH=/home/billyw/iDataset/VLA/gr00t/arena_dataset/ranch_bottle_into_fridge_generated_100/lerobot
OUTPUT_DIR=/home/billyw/iDataset/VLA/gr00t/ranch_finetune_out

uv run python -m torch.distributed.run --nproc_per_node=8 --standalone \
  gr00t/experiment/launch_finetune.py \
  --base-model-path "${BASE_MODEL_PATH}" \
  --dataset-path "${DATASET_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --modality-config-path "${MODALITY_CONFIG_PATH}" \
  --embodiment-tag GR1 \
  --num-gpus 8 \
  --global-batch-size 96 \
  --max-steps 20000 \
  --save-steps 5000 \
  --save-total-limit 5 \
  --no-tune-llm \
  --tune-visual \
  --tune-projector \
  --tune-diffusion-model \
  --dataloader-num-workers 16 \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --use-wandb
