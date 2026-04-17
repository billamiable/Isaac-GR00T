#!/usr/bin/env bash
# Base checkpoint: https://huggingface.co/nvidia/GR00T-N1.6-3B
#
# Runs huggingface-cli via the repo venv: uv run huggingface-cli ...
# Setup: uv sync && uv pip install -e .   (needs uv on PATH)
# Login: uv run huggingface-cli login   or export HF_TOKEN / HUGGINGFACE_HUB_TOKEN
#
# Usage:
#   bash scripts/preprocess_agibot/download_gr00t_n1d6_base.sh
#   LOCAL_DIR=/ephemeral/models/GR00T-N1.6-3B bash scripts/preprocess_agibot/download_gr00t_n1d6_base.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

MODEL_REPO="${MODEL_REPO:-nvidia/GR00T-N1.6-3B}"
LOCAL_DIR="${LOCAL_DIR:-/ephemeral/models/GR00T-N1.6-3B}"
mkdir -p "${LOCAL_DIR}"

TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
TOKEN_ARG=()
[[ -n "${TOKEN}" ]] && TOKEN_ARG=(--token "${TOKEN}")

uv run huggingface-cli download "${MODEL_REPO}" --local-dir "${LOCAL_DIR}" "${TOKEN_ARG[@]}"

echo "OK → ${LOCAL_DIR}  (use as BASE_MODEL_PATH)"
