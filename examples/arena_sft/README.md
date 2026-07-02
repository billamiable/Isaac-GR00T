# Arena SFT recipes for GR00T N1.6

This directory contains the portable Stage-1 SFT recipes used for the Arena GR1
ranch-bottle task and the custom LIBERO task-space dataset. It is intentionally
independent of Isaac Sim, Arena RL, Ray, and verl: an SFT smoke test only needs
Isaac-GR00T, a GR00T-flavoured LeRobot v2 dataset, a base checkpoint, and a CUDA
GPU.

The migrated scripts and configs retain their original filenames so that they
can be compared with the source recipes.

## Layout

```text
examples/arena_sft/
 README.md
 configs/
    gr1_arms_only_wrist_data_config.py
    gr1_arms_only_wrist_relative_data_config.py
    libero_franka_task_space_rel_n16_config.py
    libero_sft_rel_experiment_cfg.config.yaml
    libero_sft_rel_experiment_cfg.conf.yaml
 scripts/
     finetune_arena_gr1_ranch_bottle_8gpu_uv_wrist.sh
     finetune_arena_gr1_ranch_bottle_8gpu_uv_wrist_relative.sh
     train_libero_rel_uv.sh
```

The two ranch scripts consume both `ego_view` and `wrist_view`. The first
uses absolute actions for arms and hands. The second uses relative arm actions
and absolute hand actions. Run each config in a separate process because both
register the `GR1` embodiment tag.

## 1. Prerequisites

- Linux with a CUDA-capable NVIDIA GPU.
- NVIDIA driver compatible with the CUDA 12.8 PyTorch wheels.
- `uv`.
- FFmpeg shared libraries and the `ffmpeg`/`ffprobe` commands.
- A local GR00T-N1.6-3B checkpoint, or access to
  `nvidia/GR00T-N1.6-3B` on Hugging Face.
- Ranch data in GR00T-flavoured LeRobot v2 format.

Keep models, datasets, and outputs outside the Git checkout. Define portable
roots for the current machine:

```bash
export GR00T_MODEL_ROOT=<path-to-model-storage>
export ARENA_DATA_ROOT=<path-to-arena-datasets>
export SFT_OUTPUT_ROOT=<path-to-training-outputs>

export BASE_MODEL_PATH="${GR00T_MODEL_ROOT}/GR00T-N1.6-3B"
export DATASET_PATH="${ARENA_DATA_ROOT}/ranch_bottle_into_fridge_newcam_wrist/lerobot"
```

A Hugging Face model ID can be used instead:

```bash
export BASE_MODEL_PATH=nvidia/GR00T-N1.6-3B
```

## 2. Create the uv environment

Run these commands at the Isaac-GR00T repository root:

```bash
uv sync --python 3.10 --locked
uv pip install -e .
```

Python 3.10 is selected deliberately because this repository pins a matching
prebuilt FlashAttention wheel. Verify the environment before training:

```bash
uv run python - <<'PY'
import torch
import transformers
import gr00t

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda runtime:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

ffmpeg -version
ffprobe -version
```

Expected essentials are PyTorch 2.7.1, a visible CUDA GPU, and importable
`gr00t`. The exact package build suffix may differ by platform.

## 3. Validate the ranch dataset

The expected dataset contains `meta/`, `data/`, and `videos/`. Validate
its metadata without changing the dataset:

```bash
test -f "${DATASET_PATH}/meta/info.json"
test -f "${DATASET_PATH}/meta/modality.json"
test -f "${DATASET_PATH}/meta/stats.json"
test -d "${DATASET_PATH}/data"
test -d "${DATASET_PATH}/videos"

DATASET_PATH="${DATASET_PATH}" uv run python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["DATASET_PATH"])
info = json.loads((root / "meta/info.json").read_text())
modality = json.loads((root / "meta/modality.json").read_text())

assert info["robot_type"] == "gr1"
assert info["features"]["observation.state"]["shape"] == [26]
assert info["features"]["action"]["shape"] == [26]
assert set(modality["video"]) == {"ego_view", "wrist_view"}
assert list(modality["state"]) == [
    "left_arm", "right_arm", "left_hand", "right_hand"
]
assert list(modality["action"]) == [
    "left_arm", "right_arm", "left_hand", "right_hand"
]
print(
    "dataset OK:",
    info["total_episodes"],
    "episodes,",
    info["total_frames"],
    "frames,",
    info["total_videos"],
    "videos",
)
PY
```

For relative-arm training, `meta/relative_stats.json` must match the relative
config. Never delete the only copy of an existing stats file. If the file came
from a different modality config, move it to a backup name and let the training
pipeline recompute it.

## 4. Single-GPU SFT smoke tests

The smoke test is a real, minimal training run. It must load videos and
trajectory data, build the model, complete forward and backward passes, update
parameters, report a finite loss, and write `checkpoint-10`. It does not measure
policy quality.

Use a clean output directory for each attempt.

### 4.1 Absolute arms and hands (run first)

```bash
export OUTPUT_DIR="${SFT_OUTPUT_ROOT}/ranch_absolute_smoke"

CUDA_VISIBLE_DEVICES=0 \
NUM_GPUS=1 \
GLOBAL_BATCH_SIZE=1 \
MAX_STEPS=10 \
SAVE_STEPS=10 \
SAVE_TOTAL_LIMIT=1 \
DATALOADER_NUM_WORKERS=0 \
USE_WANDB=0 \
bash examples/arena_sft/scripts/finetune_arena_gr1_ranch_bottle_8gpu_uv_wrist.sh
```

### 4.2 Relative arms and absolute hands (run after absolute passes)

```bash
export OUTPUT_DIR="${SFT_OUTPUT_ROOT}/ranch_relative_smoke"

CUDA_VISIBLE_DEVICES=0 \
NUM_GPUS=1 \
GLOBAL_BATCH_SIZE=1 \
MAX_STEPS=10 \
SAVE_STEPS=10 \
SAVE_TOTAL_LIMIT=1 \
DATALOADER_NUM_WORKERS=0 \
USE_WANDB=0 \
bash examples/arena_sft/scripts/finetune_arena_gr1_ranch_bottle_8gpu_uv_wrist_relative.sh
```

A smoke run passes only when all of the following are true:

1. The process exits with status 0.
2. At least one optimizer step completes and the reported loss is finite.
3. `OUTPUT_DIR/checkpoint-10/config.json` exists.
4. `OUTPUT_DIR/checkpoint-10` contains model weights, `experiment_cfg/`,
   `embodiment_id.json`, and processor assets.
5. No dataset, decoder, CUDA, or distributed-worker error is hidden in the log.

## 5. Full ranch SFT

The scripts retain the original recipe defaults:

- 8 GPUs
- global batch size 96
- 20,000 steps
- save every 5,000 steps
- tune visual encoder, projector, and diffusion model
- keep the LLM frozen
- enable W&B

Absolute run:

```bash
export OUTPUT_DIR="${SFT_OUTPUT_ROOT}/ranch_finetune_newcam_wrist_out"
bash examples/arena_sft/scripts/finetune_arena_gr1_ranch_bottle_8gpu_uv_wrist.sh
```

Relative run:

```bash
export OUTPUT_DIR="${SFT_OUTPUT_ROOT}/ranch_finetune_newcam_wrist_relative_out"
bash examples/arena_sft/scripts/finetune_arena_gr1_ranch_bottle_8gpu_uv_wrist_relative.sh
```

Every runtime setting can be overridden with an environment variable:
`NUM_GPUS`, `GLOBAL_BATCH_SIZE`, `MAX_STEPS`, `SAVE_STEPS`,
`SAVE_TOTAL_LIMIT`, `DATALOADER_NUM_WORKERS`, `MASTER_PORT`, and
`USE_WANDB`. Extra `launch_finetune.py` arguments may be appended to either
script.

## 6. LIBERO relative xyz+rotvec SFT

`train_libero_rel_uv.sh` trains the custom `all_libero_suites_rel_rotvec`
dataset with `NEW_EMBODIMENT`. The state and action are seven-dimensional:
EEF xyz+rotvec `[0:6]` and gripper `[6:7]`. EEF actions are relative to the
current EEF state; the gripper remains absolute. The two camera inputs are
`agentview_cam` and `eye_in_hand_cam`.

Set the LIBERO dataset path:

```bash
export DATASET_PATH="${ARENA_DATA_ROOT}/all_libero_suites_rel_rotvec"
```

The expected dataset has 1,477 episodes, 243,763 frames, 41 tasks, and two
videos per episode. Its `meta/relative_stats.json` must contain statistics for
`franka_eef_pose`.

Single-GPU 10-step smoke test:

```bash
export OUTPUT_DIR="${SFT_OUTPUT_ROOT}/libero_relative_smoke"

CUDA_VISIBLE_DEVICES=0 \
NUM_GPUS=1 \
GLOBAL_BATCH_SIZE=1 \
MAX_STEPS=10 \
SAVE_STEPS=10 \
SAVE_TOTAL_LIMIT=1 \
DATALOADER_NUM_WORKERS=0 \
USE_WANDB=0 \
bash examples/arena_sft/scripts/train_libero_rel_uv.sh
```

Full SFT uses the script defaults: 8 GPUs, global batch size 160, 20,000
steps, and a checkpoint every 5,000 steps:

```bash
export OUTPUT_DIR="${SFT_OUTPUT_ROOT}/gr00t_n16_libero_all_suites_rel_rotvec_out"
bash examples/arena_sft/scripts/train_libero_rel_uv.sh
```

The two resolved LIBERO YAML files are provenance snapshots, not launch files.
Machine-specific source paths have been replaced by the symbolic values
`DATASET_PATH`, `OUTPUT_DIR`, and `BASE_MODEL_PATH`.

## 7. Troubleshooting

- **torchcodec is not available / FFmpeg library error:** confirm both
  `ffmpeg` and `ffprobe` work. If FFmpeg comes from conda, activate that
  environment before running the script; its `lib` and `bin` paths are
  prepended automatically.
- **CUDA out of memory:** ensure no other process occupies the GPU. The smoke
  recipe already uses batch size 1. Do not disable trainable components merely
  to make the smoke test pass, because that would no longer validate the target
  recipe.
- **global_batch_size assertion:** it must be divisible by `NUM_GPUS`.
- **Port already in use:** set a different `MASTER_PORT`.
- **Incorrect relative statistics:** back up the existing
  `meta/relative_stats.json`, then rerun only the relative recipe so rank 0
  computes stats for the configured relative action keys.
- **W&B authentication:** use `USE_WANDB=0` for smoke tests, or run
  `uv run wandb login` before a full run.

## 8. Local verification record

The ranch dataset used for the initial verification contains 70 episodes,
22,873 frames, and 140 videos at 512 x 512 and 50 FPS. The runs used
Python 3.10.19, PyTorch 2.7.1+cu128, and one 48 GB RTX 5880 Ada GPU;
peak training memory was approximately 44.8 GB. The saved resolved configs
confirm four absolute action groups for the absolute run, and relative arms
plus absolute hands for the relative run.

The LIBERO dataset contains 1,477 episodes, 243,763 frames, 41 tasks, and
1,477 videos for each of its two camera views. Its resolved config confirms
relative EEF xyz+rotvec actions and an absolute gripper. Results and any
required compatibility fixes are recorded here after each smoke run:

| Variant | Training steps | Result | Evidence |
| --- | ---: | --- | --- |
| absolute arms/hands | 10 | PASS | exit 0; checkpoint-10; loss 0.302; grad norm 1.6502 |
| relative arms, absolute hands | 10 | PASS | exit 0; checkpoint-10; loss 0.5264; grad norm 1.8091 |
| LIBERO relative EEF xyz+rotvec, absolute gripper | 10 | PASS | exit 0; checkpoint-10; loss 1.2794; grad norm 2.3682 |
