#!/usr/bin/env bash
# Raw GenieSim instruction data (ModelScope):
# https://modelscope.cn/datasets/agibot_world/GenieSim3.0-Dataset/tree/master/task_suite/instruction
#
# Needs: uv pip install modelscope   (and modelscope on PATH, or run from a shell where `which modelscope` works)
#
# Usage:
#   bash scripts/preprocess_agibot/download_geniesim_instruction_modelscope.sh
#   LOCAL_DIR=/ephemeral/agibot/instruction bash scripts/preprocess_agibot/download_geniesim_instruction_modelscope.sh
set -euo pipefail

LOCAL_DIR="${LOCAL_DIR:-/ephemeral/agibot/instruction}"
mkdir -p "${LOCAL_DIR}"

modelscope download \
  --dataset agibot_world/GenieSim3.0-Dataset \
  --local_dir "${LOCAL_DIR}" \
  --include 'task_suite/instruction/**' \
  --max-workers "${MODELSCOPE_MAX_WORKERS:-4}"

# Match training_paths RAW_ROOT: task folders at top of LOCAL_DIR
mv "${LOCAL_DIR}/task_suite/instruction/"* "${LOCAL_DIR}/"
rm -rf "${LOCAL_DIR}/task_suite"

echo "OK → ${LOCAL_DIR}  (use as RAW_ROOT)"
