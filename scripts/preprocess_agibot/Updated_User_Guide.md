# Updated User Guide (GenieSim -> GR00T Post-Training)

This guide describes the current workflow we are using:
- Cloud: data extraction, preprocessing, and GR00T finetuning with `uv`.
- Local: simulation evaluation with trained checkpoints.

Docker is intentionally removed from this guide.

## 0) Paths Used in This Guide

- Repo root: `/home/shadeform/iCode/VLA/Isaac-GR00T`
- Raw dataset root: `/home/shadeform/iDataset/simulation/task_suite/instruction`
- Preprocessed dataset root: `/home/shadeform/iDataset/simulation/task_suite/instruction_preprocessed`
- Base model: `/home/shadeform/iDataset/VLA/gr00t/GR00T-N1.6-3B`
- Training outputs: `/ephemeral/gr00t_models/...`

## 1) Prepare uv Environment

Install/sync dependencies:

```bash
cd /home/shadeform/iCode/VLA/Isaac-GR00T
uv sync
uv run python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

Note:
- The preprocessing wrapper uses `uv run --with decord python`, so `decord` is injected automatically for preprocessing.

## 2) Verify and Extract Raw Data Archives

Run from repo root:

```bash
cd /home/shadeform/iCode/VLA/Isaac-GR00T

# Verify archives only
bash scripts/preprocess_agibot/verify_and_unpack_geniesim_archives.sh \
  /home/shadeform/iDataset/simulation/task_suite/instruction

# Verify + extract all tasks
bash scripts/preprocess_agibot/verify_and_unpack_geniesim_archives.sh \
  /home/shadeform/iDataset/simulation/task_suite/instruction --extract
```

Expected summary:
- `tasks=10, ok=10, fail=0`

## 3) Preprocess All 10 Tasks

Use the batch script:

```bash
cd /home/shadeform/iCode/VLA/Isaac-GR00T
bash scripts/preprocess_agibot/preprocess_all_instruction_tasks.sh
```

Optional modes:

```bash
# Trim dry-run only (preprocess still runs)
MODE=dry-run bash scripts/preprocess_agibot/preprocess_all_instruction_tasks.sh

# Skip video resizing if source videos are already suitable
SKIP_VIDEO=1 bash scripts/preprocess_agibot/preprocess_all_instruction_tasks.sh
```

Quick validation:

```bash
python3 - <<'PY'
from pathlib import Path
root = Path('/home/shadeform/iDataset/simulation/task_suite/instruction_preprocessed')
tasks = sorted([p for p in root.iterdir() if p.is_dir()])
print('task_count=', len(tasks))
for t in tasks:
    pq = sum(1 for _ in (t/'data').rglob('*.parquet'))
    print(t.name, 'info=', (t/'meta'/'info.json').exists(), 'parquet=', pq)
PY
```

## 4) Finetune on All Instruction Tasks

Training script:
- `scripts/preprocess_agibot/finetune_geniesim_instruction.sh`

Before training:

```bash
cd /home/shadeform/iCode/VLA/Isaac-GR00T
uv run wandb login
```

Run training:

```bash
bash scripts/preprocess_agibot/finetune_geniesim_instruction.sh
```

### Color jitter option

The script supports:
- `USE_COLOR_JITTER=0|1` (default `0`)
- `COLOR_JITTER_PARAMS` (when `USE_COLOR_JITTER=1`)

Examples:

```bash
# No color jitter (default)
bash scripts/preprocess_agibot/finetune_geniesim_instruction.sh

# Enable color jitter with default params
USE_COLOR_JITTER=1 bash scripts/preprocess_agibot/finetune_geniesim_instruction.sh

# Enable color jitter with custom params
USE_COLOR_JITTER=1 \
COLOR_JITTER_PARAMS="brightness 0.2 contrast 0.3 saturation 0.4 hue 0.05" \
bash scripts/preprocess_agibot/finetune_geniesim_instruction.sh
```

Output directory default pattern:
- `/ephemeral/gr00t_models/instruction_color_jitter_{on|off}_YYYYMMDD_HH`

## 5) Move Checkpoints for Local Simulation Evaluation

After training finishes, copy only the new checkpoint output directory to local machine:

```bash
rsync -avP /ephemeral/gr00t_models/<run_dir> <local_user>@<local_host>:/path/to/local/gr00t_models/
```

Keep embodiment consistent during evaluation:
- `AGIBOT_GENIE1` (current script default)

## 6) Notes

- `GenieSimAssets` is not required for cloud post-training itself.
- Keep large datasets/checkpoints under `/ephemeral` to avoid filling `/home`.
- Video stack summary (important):
  - Raw instruction videos are sampled as `hevc` (`H.265`) in `.mp4` containers.
  - Preprocessing rewrites videos to `.mp4` with `mp4v` codec (`cv2.VideoWriter_fourcc("mp4v")`).
  - Preprocessing scripts (`preprocess_dataset.py`, `trim_static_frames.py`) require `decord`.
  - Training data loading defaults to `torchcodec` backend and can fall back to `decord` if `torchcodec` is unavailable.
  - Frequent warning like "torchcodec is not available, falling back to decord" means training is currently decoding with `decord`.
- Why `gr00t/configs/finetune_config.py` was changed:
  - `dataset_paths` was updated to `= None` to fix Python dataclass initialization order.
  - Without this, launch fails early with:
    - `TypeError: non-default argument 'dataset_paths' follows default argument`
- Why `scripts/preprocess_agibot/finetune_geniesim_instruction.sh` uses `uv run --python 3.10 torchrun`:
  - The Eagle backbone requires FlashAttention2 (`flash_attn`), and the current environment setup expects a Python 3.10-compatible wheel.
  - Running with Python 3.12 causes startup failure before training due to missing/unsupported `flash_attn`.
  - Pinning `--python 3.10` ensures the training command resolves a compatible runtime.

