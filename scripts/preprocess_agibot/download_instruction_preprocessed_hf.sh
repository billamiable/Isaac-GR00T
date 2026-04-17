#!/usr/bin/env bash
# Preprocessed dataset: https://huggingface.co/datasets/billyyujie/instruction_preprocessed
#
# Runs huggingface-cli via the repo venv: uv run huggingface-cli ...
# Setup: uv sync && uv pip install -e .   (needs uv on PATH)
# Login: uv run huggingface-cli login   or export HF_TOKEN / HUGGINGFACE_HUB_TOKEN
#
# If you still get HTTP 429 (e.g. "1000 api requests per 5 minutes" on xet-read-token):
#   wait ~5 minutes, then e.g. HF_HUB_DOWNLOAD_MAX_WORKERS=1 HF_HUB_DISABLE_XET=1 bash ...
#
# Usage:
#   bash scripts/preprocess_agibot/download_instruction_preprocessed_hf.sh
#   LOCAL_DIR=/ephemeral/agibot/instruction_preprocessed bash scripts/preprocess_agibot/download_instruction_preprocessed_hf.sh
set -euo pipefail

MAX_WORKERS="${HF_HUB_DOWNLOAD_MAX_WORKERS:-2}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

DATASET_REPO="${DATASET_REPO:-billyyujie/instruction_preprocessed}"
LOCAL_DIR="${LOCAL_DIR:-/ephemeral/agibot/instruction_preprocessed}"
mkdir -p "${LOCAL_DIR}"

TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
TOKEN_ARG=()
[[ -n "${TOKEN}" ]] && TOKEN_ARG=(--token "${TOKEN}")

uv run huggingface-cli download \
  --repo-type dataset \
  "${DATASET_REPO}" \
  --local-dir "${LOCAL_DIR}" \
  --max-workers "${MAX_WORKERS}" \
  "${TOKEN_ARG[@]}"

echo "OK → ${LOCAL_DIR}  (use as PREPROCESSED_ROOT)"
