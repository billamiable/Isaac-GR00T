#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_SH="${TRAINING_ENV_SH:-${SCRIPT_DIR}/training_paths.sh}"

if [[ -f "${ENV_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_SH}"
fi

PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-/ephemeral/agibot/instruction_preprocessed}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/ephemeral/models/GR00T-N1.6-3B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/ephemeral/gr00t_models}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
NUM_GPUS="${NUM_GPUS:-8}"

EXPECTED_TASKS=(
  pick_block_color
  pick_block_number
  pick_block_shape
  pick_block_size
  pick_common_sense
  pick_object_type
  pick_specific_object
  straighten_object
  pick_follow_logic_or
  pick_billards_color
)

fail() {
  echo "[FAIL] $*" >&2
  exit 1
}

check_cmd() {
  local cmd="$1"
  command -v "${cmd}" >/dev/null 2>&1 || fail "Missing command: ${cmd}"
}

echo "== Repo =="
echo "REPO_ROOT=${REPO_ROOT}"
echo

echo "== Command Checks =="
check_cmd bash
check_cmd timeout
check_cmd nvidia-smi
check_cmd uv
echo "[OK] Required commands found"
echo

echo "== Path Checks =="
[[ -d "${REPO_ROOT}" ]] || fail "Repo root not found: ${REPO_ROOT}"
[[ -x "${PYTHON_BIN}" ]] || fail "Python executable not found: ${PYTHON_BIN}. Run 'uv sync' first."
[[ -d "${BASE_MODEL_PATH}" ]] || fail "Base model path not found: ${BASE_MODEL_PATH}"
[[ -d "${PREPROCESSED_ROOT}" ]] || fail "Preprocessed dataset root not found: ${PREPROCESSED_ROOT}"
mkdir -p "${OUTPUT_ROOT}"
echo "[OK] Core paths validated"
echo

echo "== GPU Driver Check =="
timeout 10s nvidia-smi >/dev/null || fail "nvidia-smi failed or hung"
echo "[OK] nvidia-smi responded"
echo

echo "== Python/CUDA Check =="
CUDA_PROBE="$(
  timeout 20s "${PYTHON_BIN}" -c \
    "import torch; print(int(torch.cuda.is_available()), torch.cuda.device_count(), torch.__version__, torch.version.cuda)" \
    2>/dev/null
)" || fail "Python CUDA probe failed or hung"

CUDA_OK="$(echo "${CUDA_PROBE}" | awk '{print $1}')"
CUDA_COUNT="$(echo "${CUDA_PROBE}" | awk '{print $2}')"
TORCH_VERSION="$(echo "${CUDA_PROBE}" | awk '{print $3}')"
CUDA_VERSION="$(echo "${CUDA_PROBE}" | awk '{print $4}')"

echo "torch=${TORCH_VERSION} cuda=${CUDA_VERSION} is_available=${CUDA_OK} device_count=${CUDA_COUNT}"
[[ "${CUDA_OK}" == "1" ]] || fail "torch.cuda.is_available() != 1"
[[ "${CUDA_COUNT}" -ge "${NUM_GPUS}" ]] || fail "Visible GPU count ${CUDA_COUNT} is less than NUM_GPUS=${NUM_GPUS}"
echo "[OK] CUDA runtime looks healthy"
echo

echo "== Dataset Check =="
missing_tasks=0
for task_name in "${EXPECTED_TASKS[@]}"; do
  if [[ -d "${PREPROCESSED_ROOT}/${task_name}" ]]; then
    echo "[OK] ${PREPROCESSED_ROOT}/${task_name}"
  else
    echo "[MISSING] ${PREPROCESSED_ROOT}/${task_name}"
    missing_tasks=$((missing_tasks + 1))
  fi
done
if [[ "${missing_tasks}" -gt 0 ]]; then
  echo "[WARN] ${missing_tasks} expected preprocessed task(s) are missing"
else
  echo "[OK] All expected preprocessed tasks exist"
fi
echo

echo "== Recommended Launch =="
echo "cd ${REPO_ROOT}"
echo "source scripts/preprocess_agibot/training_paths.sh"
echo "bash scripts/preprocess_agibot/finetune_geniesim_instruction.sh"
