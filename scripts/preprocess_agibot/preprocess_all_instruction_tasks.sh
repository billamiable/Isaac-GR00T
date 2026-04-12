#!/usr/bin/env bash
set -euo pipefail

# Batch preprocess all instruction tasks by calling preprocess_agibot_data.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PREPROCESS_SH="${SCRIPT_DIR}/preprocess_agibot_data.sh"
ENV_SH="${TRAINING_ENV_SH:-${SCRIPT_DIR}/training_paths.sh}"

if [[ -f "${ENV_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_SH}"
fi

SRC_ROOT="${SRC_ROOT:-/home/shadeform/iDataset/simulation/task_suite/instruction}"
DST_ROOT="${DST_ROOT:-/home/shadeform/iDataset/simulation/task_suite/instruction_preprocessed}"
MODE="${MODE:-run}"               # run | dry-run
SKIP_VIDEO="${SKIP_VIDEO:-0}"     # 1 to pass --skip_video

if [[ ! -x "${PREPROCESS_SH}" ]]; then
  echo "Error: preprocess script not executable: ${PREPROCESS_SH}" >&2
  exit 1
fi

if [[ ! -d "${SRC_ROOT}" ]]; then
  echo "Error: source root not found: ${SRC_ROOT}" >&2
  exit 1
fi

mkdir -p "${DST_ROOT}"
cd "${REPO_ROOT}"

echo "Source root: ${SRC_ROOT}"
echo "Output root: ${DST_ROOT}"
echo "Mode: ${MODE}"
echo "Skip video: ${SKIP_VIDEO}"
echo

for task_dir in "${SRC_ROOT}"/*; do
  [[ -d "${task_dir}" ]] || continue
  task_name="$(basename "${task_dir}")"
  out_dir="${DST_ROOT}/${task_name}"

  cmd=(bash "${PREPROCESS_SH}" "${task_dir}" "${out_dir}")
  if [[ "${SKIP_VIDEO}" == "1" ]]; then
    cmd+=(--skip_video)
  fi
  if [[ "${MODE}" == "dry-run" ]]; then
    cmd+=(-- --dry-run)
  elif [[ "${MODE}" != "run" ]]; then
    echo "Error: unsupported MODE=${MODE}, use run or dry-run" >&2
    exit 1
  fi

  echo "=== ${task_name} ==="
  "${cmd[@]}"
  echo
done

echo "All tasks finished."

