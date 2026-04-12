# Updated User Guide (Simple Training Flow)

This guide assumes one convention:

- code lives under `home`
- large data lives under `ephemeral`
- you already have a preprocessed dataset

Main paths:

- repo: `/home/$USER/iCode/VLA/Isaac-GR00T`
- preprocessed data: `/ephemeral/agibot/instruction_preprocessed`
- base model: `/ephemeral/models/GR00T-N1.6-3B`
- outputs: `/ephemeral/gr00t_models`

## 1) Configure Paths Once

Edit:

- `scripts/preprocess_agibot/training_paths.sh`

This file should only define the key paths:

- `RAW_ROOT`
- `PREPROCESSED_ROOT`
- `BASE_MODEL_PATH`
- `OUTPUT_ROOT`

Then load it:

```bash
cd /home/$USER/iCode/VLA/Isaac-GR00T
source scripts/preprocess_agibot/training_paths.sh
```

If you want these paths in every shell automatically, add this to `~/.bashrc`:

```bash
source /home/$USER/iCode/VLA/Isaac-GR00T/scripts/preprocess_agibot/training_paths.sh
```

## 2) Run Sanity Check

Run:

```bash
cd /home/$USER/iCode/VLA/Isaac-GR00T
bash scripts/preprocess_agibot/check_training_env.sh
```

This script checks:

- `uv`, `nvidia-smi`, and basic command availability
- repo and `.venv`
- CUDA runtime health
- base model path
- preprocessed dataset root
- the 10 expected preprocessed task directories

If this passes, the environment is ready for training.

## 3) Train

Run:

```bash
cd /home/$USER/iCode/VLA/Isaac-GR00T
bash scripts/preprocess_agibot/finetune_geniesim_instruction.sh
```

The training script reads path defaults from `training_paths.sh` automatically.

Training-related defaults such as these stay inside `finetune_geniesim_instruction.sh`:

- `NUM_GPUS`
- `CUDA_VISIBLE_DEVICES`
- `DATALOADER_NUM_WORKERS`
- `MASTER_PORT`
- `USE_COLOR_JITTER`

Only override them inline when you want a one-off run:

```bash
USE_COLOR_JITTER=1 bash scripts/preprocess_agibot/finetune_geniesim_instruction.sh
```

```bash
OUTPUT_DIR="${OUTPUT_ROOT}/instruction_run_custom" \
USE_COLOR_JITTER=1 \
COLOR_JITTER_PARAMS="brightness 0.2 contrast 0.3 saturation 0.4 hue 0.05" \
bash scripts/preprocess_agibot/finetune_geniesim_instruction.sh
```

## 4) Notes

- outputs go to `ephemeral`, so sync important checkpoints out regularly
- `decord` is still important in this workflow
- training prefers `torchcodec` and can fall back to `decord`
- if you see "torchcodec is not available, falling back to decord", training is decoding with `decord`

Example checkpoint backup:

```bash
rsync -avP /ephemeral/gr00t_models/<run_dir> <user>@<host>:/path/to/gr00t_models/
```

## 5) Optional: Build Preprocessed Data

This section is only needed if you do not already have preprocessed data.

Relevant scripts:

- `scripts/preprocess_agibot/verify_and_unpack_geniesim_archives.sh`
- `scripts/preprocess_agibot/preprocess_all_instruction_tasks.sh`
- `scripts/preprocess_agibot/preprocess_agibot_data.sh`
- `scripts/preprocess_agibot/preprocess_dataset.py`
- `scripts/preprocess_agibot/trim_static_frames.py`

Quick examples:

```bash
cd /home/$USER/iCode/VLA/Isaac-GR00T
bash scripts/preprocess_agibot/verify_and_unpack_geniesim_archives.sh --extract
```

```bash
cd /home/$USER/iCode/VLA/Isaac-GR00T
bash scripts/preprocess_agibot/preprocess_all_instruction_tasks.sh
```

Video/codec notes:

- raw instruction videos are commonly `hevc` (`H.265`) in `.mp4`
- preprocessing rewrites videos to `.mp4` using `mp4v`
- preprocessing scripts rely on `decord`

