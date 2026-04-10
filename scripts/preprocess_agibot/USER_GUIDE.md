# Agibot Data Pipeline User Guide

This guide covers the end-to-end workflow for preparing Agibot datasets, fine-tuning the GR00T model, and deploying the inference service.

---

## 1. Data Preprocessing

The preprocessing pipeline converts a raw Agibot dataset into a format ready for GR00T fine-tuning. It is composed of two stages that can be run individually or via the unified shell wrapper.

### 1.1 Quick Start (Shell Wrapper)

The easiest way to run the full pipeline is via `preprocess_agibot_data.sh`:

```bash
bash scripts/preprocess_agibot/preprocess_agibot_data.sh <SOURCE> <OUTPUT>
```

This executes two steps sequentially:
1. `preprocess_dataset.py` — copy, modality generation, video normalization, and language processing.
2. `trim_static_frames.py` — remove static (idle) frames at the head/tail of each episode.

**Examples:**

```bash
# Basic usage
bash scripts/preprocess_agibot/preprocess_agibot_data.sh /data/raw_dataset /data/preprocessed_dataset

# Skip video re-encoding (useful if videos are already 256x256)
bash scripts/preprocess_agibot/preprocess_agibot_data.sh /data/raw_dataset /data/preprocessed_dataset --skip_video

# Customize worker counts for both stages (separated by --)
bash scripts/preprocess_agibot/preprocess_agibot_data.sh /data/raw_dataset /data/preprocessed_dataset --workers 8 -- --workers 16

# Dry-run trimming (preview what would be trimmed without modifying files)
bash scripts/preprocess_agibot/preprocess_agibot_data.sh /data/raw_dataset /data/preprocessed_dataset -- --dry-run
```

### 1.2 Stage 1: `preprocess_dataset.py`

This script performs four processing steps on the dataset:

| Step | Description |
|------|-------------|
| **Step 0** | Generate `meta/modality.json` from `info.json` field descriptions (or copy from a template) |
| **Step 1** | Normalize all video frames to 256×256 (aspect-ratio-preserving scale + center pad) |
| **Step 2** | Update language annotations: refresh `high_level_instruction`, rewrite `tasks.jsonl` and `episodes.jsonl` |
| **Step 3** | Write `episode_index` / `task_index` columns into each parquet file based on filename |

**Usage:**

```bash
python scripts/preprocess_agibot/preprocess_dataset.py \
  --source /path/to/raw_dataset \
  --output /path/to/output_dataset
```

**Options:**

| Flag | Description |
|------|-------------|
| `--source` | Source dataset directory (must contain `meta/info.json`) |
| `--output` | Output directory (will copy source here, then process in-place) |
| `--skip_copy` | Skip re-copying from source; process the existing output directory |
| `--skip_video` | Skip video resolution normalization |
| `--modality-template PATH` | Use a custom `modality.json` template instead of auto-generating |
| `--target-size H W` | Target video resolution (default: `256 256`) |
| `--workers N` | Number of worker processes for parallel video encoding (default: CPU count) |
| `--no-waist` | Omit `waist_position` from the generated `modality.json` |

### 1.3 Stage 2: `trim_static_frames.py`

Removes idle/static frames from the head and tail of each episode using the `instruction_segments` field in `info.json`. Both parquet data and video files are trimmed accordingly.

**Usage:**

```bash
python scripts/preprocess_agibot/trim_static_frames.py /path/to/output_dataset
```

**Options:**

| Flag | Description |
|------|-------------|
| `--dry-run` | Preview planned trim actions without modifying any files |
| `--frame-index-col NAME` | Frame index column name in parquet (default: `frame_index`) |
| `--workers N` | Number of concurrent worker threads (default: `4`) |

### 1.4 Expected Dataset Structure

After preprocessing, the dataset should have the following layout:

```
output_dataset/
├── meta/
│   ├── info.json              # Dataset metadata (features, fps, instruction_segments, etc.)
│   ├── modality.json          # Modality configuration (auto-generated or from template)
│   ├── tasks.jsonl            # One task description per episode
│   └── episodes.jsonl         # Episode metadata (index, tasks, length)
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
└── videos/
    └── chunk-000/
        ├── observation.images.top_head/
        │   ├── episode_000000.mp4
        │   └── ...
        ├── observation.images.hand_left/
        │   └── ...
        └── observation.images.hand_right/
            └── ...
```

---

## 2. Training (Fine-Tuning)

Once preprocessing is complete, launch distributed fine-tuning with `torchrun`:

```bash
export NUM_GPUS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node=8 --master_port=29500 gr00t/experiment/launch_finetune.py \
  --base-model-path /path/to/gr00t_16_models \
  --dataset-path /path/to/preprocessed_dataset \
  --embodiment-tag AGIBOT_GENIE1 \
  --output-dir /path/to/checkpoints/output_dir \
  --num-gpus 8 \
  --global-batch-size 512 \
  --max-steps 30000 \
  --save-steps 3000 \
  --save-total-limit 5 \
  --dataloader_num_workers 8
```

**Key arguments:**

| Argument | Description |
|----------|-------------|
| `--base-model-path` | Path to the pre-trained GR00T N1.6 model weights |
| `--dataset-path` | Path to the preprocessed dataset (output from Stage 1) |
| `--embodiment-tag` | Embodiment identifier; use `AGIBOT_GENIE1` (without waist) or `AGIBOT_GENIE1_WAIST` (with waist) |
| `--output-dir` | Directory where checkpoints and logs will be saved |
| `--num-gpus` | Number of GPUs (should match `nproc_per_node`) |
| `--global-batch-size` | Total batch size across all GPUs |
| `--max-steps` | Total number of training steps |
| `--save-steps` | Save a checkpoint every N steps |
| `--save-total-limit` | Maximum number of checkpoints to keep on disk |
| `--dataloader_num_workers` | Number of data-loading workers per process |

---

## 3. Deployment

After training completes, deploy the fine-tuned model as a WebSocket inference service.

### 3.1 Start the WebSocket Server

```bash
uv run --extra websocket python scripts/deployment/serve_gr00t_websocket.py \
  --model-path /path/to/checkpoints/output_dir \
  --port 8000
```

If `uv` is not available, install the dependency manually and run with `python` directly:

```bash
pip install websockets
python scripts/deployment/serve_gr00t_websocket.py \
  --model-path /path/to/checkpoints/output_dir \
  --port 8000
```

**Server arguments:**

| Argument | Description |
|----------|-------------|
| `--model-path` | Path to the fine-tuned checkpoint directory |
| `--device` | CUDA device (default: `cuda:0`) |
| `--host` | Bind address (default: `0.0.0.0`) |
| `--port` | WebSocket port (default: `8000`) |
| `--modality-config-path` | Optional custom modality config `.py` file (if used during training) |
| `--action-horizon` | If set, truncate returned action sequences to N timesteps |

### 3.2 Client Payload Format

The server expects a **msgpack**-encoded dictionary with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `images.top_head` | ndarray (H,W,3) | Top-head camera image |
| `images.hand_left` | ndarray (H,W,3) | Left hand camera image |
| `images.hand_right` | ndarray (H,W,3) | Right hand camera image |
| `state` | list / ndarray | Flattened robot state vector (see below) |
| `prompt` or `task_name` | str | Language instruction |

**State vector layout** (`AGIBOT_GENIE1`, 16-D):

| Index | Content |
|-------|---------|
| 0–6 | Left arm joint positions (7) |
| 7–13 | Right arm joint positions (7) |
| 14 | Left gripper position (1) |
| 15 | Right gripper position (1) |

For `AGIBOT_GENIE1_WAIST` (17-D), an additional waist dimension is read from index 20.

### 3.3 Server Response

The server returns a msgpack-encoded dictionary:

```json
{
  "actions": [[...], [...]],
  "actions_by_key": {
    "left_arm_joint_position": [[...]],
    "right_arm_joint_position": [[...]],
    "left_effector_position": [[...]],
    "right_effector_position": [[...]]
  }
}
```

- `actions`: Concatenated action vectors per timestep (16-D for `AGIBOT_GENIE1`, 17-D for `AGIBOT_GENIE1_WAIST`).
- `actions_by_key`: Actions split by joint group for easier downstream consumption.
