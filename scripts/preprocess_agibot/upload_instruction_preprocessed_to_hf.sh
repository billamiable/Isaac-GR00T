#!/usr/bin/env bash
set -euo pipefail

# 登录（在 Isaac-GR00T 下）: uv run --group dev hf auth login
# 或 export HF_TOKEN=hf_...

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SRC="${SRC:-/home/yujie/workspace/yujie/iDataset/simulation/genie_sim/dataset/task_suite/instruction_preprocessed}"
REPO="${REPO:-billyyujie/instruction_preprocessed}"
NUM_WORKERS="${NUM_WORKERS:-}"

args=(hf upload-large-folder "$REPO" "$SRC" --repo-type dataset)
[[ -n "$NUM_WORKERS" ]] && args+=(--num-workers "$NUM_WORKERS")

uv run --group dev "${args[@]}"
