# Updated User Guide (Simple Training Flow)

Convention: code under `home`, large data under `ephemeral`.

Training needs **`PREPROCESSED_ROOT`** (and the base checkpoint). Either download preprocessed data or **run the preprocess pipeline** (§7)—that step is required unless you already have `PREPROCESSED_ROOT` populated. **Raw GenieSim instruction** under `RAW_ROOT` is **optional**: only if you preprocess locally from archives instead of using a pre-downloaded preprocessed dataset.

Main paths:

- repo: `/home/$USER/iCode/VLA/Isaac-GR00T`
- raw instruction (optional): `/ephemeral/agibot/instruction`
- preprocessed (required for training): `/ephemeral/agibot/instruction_preprocessed`
- base model: `/ephemeral/models/GR00T-N1.6-3B`
- outputs: `/ephemeral/gr00t_models`

## 1) Configure Paths Once

Edit `scripts/preprocess_agibot/training_paths.sh` (`RAW_ROOT`, `PREPROCESSED_ROOT`, `BASE_MODEL_PATH`, `OUTPUT_ROOT`), then:

```bash
cd /home/$USER/iCode/VLA/Isaac-GR00T
source scripts/preprocess_agibot/training_paths.sh
```

Optional `~/.bashrc`: `source /home/$USER/iCode/VLA/Isaac-GR00T/scripts/preprocess_agibot/training_paths.sh`

## 2) Environment (system + Python)

**Recommended on a fresh dGPU Linux box:** one shot from the repo root — installs system packages (`ffmpeg`, `libaio-dev`, CUDA toolkit via apt if `/usr/local/cuda` is missing), installs `uv` if needed, then runs **`uv sync`** and **`uv pip install -e .`** inside the script. You do **not** need to run those two commands again after a successful `install_deps.sh`.

```bash
cd /home/$USER/iCode/VLA/Isaac-GR00T
bash scripts/deployment/dgpu/install_deps.sh
```

Requires `sudo` for apt (unless root). See main `README.md` / [`scripts/deployment/dgpu/install_deps.sh`](../deployment/dgpu/install_deps.sh) for details.

**If you skip that script** (system deps and CUDA toolkit already match what DeepSpeed / training need), install the Python env only:

```bash
cd /home/$USER/iCode/VLA/Isaac-GR00T
uv sync
uv pip install -e .
```

After changing dependencies or lockfile, run `uv sync` again (with or without having used `install_deps.sh`).

## 3) Optional: Downloads

| Script | Default role |
| ------ | ------------ |
| [`download_gr00t_n1d6_base.sh`](download_gr00t_n1d6_base.sh) | [GR00T-N1.6-3B](https://huggingface.co/nvidia/GR00T-N1.6-3B) → `BASE_MODEL_PATH` |
| [`download_instruction_preprocessed_hf.sh`](download_instruction_preprocessed_hf.sh) | HF dataset → `PREPROCESSED_ROOT` (if you use a published dump) |
| [`download_geniesim_instruction_modelscope.sh`](download_geniesim_instruction_modelscope.sh) | Optional raw instruction ([ModelScope](https://modelscope.cn/datasets/agibot_world/GenieSim3.0-Dataset/tree/master/task_suite/instruction)) → `RAW_ROOT`; needs `uv pip install modelscope` |

```bash
cd /home/$USER/iCode/VLA/Isaac-GR00T
bash scripts/preprocess_agibot/download_gr00t_n1d6_base.sh
bash scripts/preprocess_agibot/download_instruction_preprocessed_hf.sh   # if applicable
uv pip install modelscope && bash scripts/preprocess_agibot/download_geniesim_instruction_modelscope.sh   # only if you need RAW_ROOT
```

HF: `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` or `uv run huggingface-cli login`. Heavy many-file repos: see [rate limits](https://huggingface.co/docs/hub/rate-limits); tune env vars in `download_instruction_preprocessed_hf.sh` if needed.

## 4) Run Sanity Check

```bash
cd /home/$USER/iCode/VLA/Isaac-GR00T
bash scripts/preprocess_agibot/check_training_env.sh
```

Checks commands, `.venv`, CUDA, `BASE_MODEL_PATH`, `PREPROCESSED_ROOT`, and expected task folders.

## 5) Train

`finetune_geniesim_instruction.sh` passes **`--use_wandb`** to the trainer. Before running, either:

- `export WANDB_API_KEY=...`, or
- `uv run wandb login` (or `wandb login` from an activated `.venv`)

Without credentials, runs may fail or hang in non-interactive shells. For local-only logging you can try `WANDB_MODE=offline`. To turn off W&B entirely, remove `--use_wandb` from that shell script (there is no env toggle today).

```bash
cd /home/$USER/iCode/VLA/Isaac-GR00T
bash scripts/preprocess_agibot/finetune_geniesim_instruction.sh
```

**Color jitter (optional):** by default `USE_COLOR_JITTER=0`. Set `USE_COLOR_JITTER=1` to enable on-the-fly image augmentation. Strength is controlled by `COLOR_JITTER_PARAMS` (space-separated `name value` pairs, passed through to the training script). Defaults are defined in `finetune_geniesim_instruction.sh` (`brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08`).

```bash
# enable with script defaults
USE_COLOR_JITTER=1 bash scripts/preprocess_agibot/finetune_geniesim_instruction.sh
```

```bash
# custom strengths (example)
USE_COLOR_JITTER=1 \
COLOR_JITTER_PARAMS="brightness 0.2 contrast 0.3 saturation 0.4 hue 0.05" \
bash scripts/preprocess_agibot/finetune_geniesim_instruction.sh
```

Other training defaults (`NUM_GPUS`, `CUDA_VISIBLE_DEVICES`, `DATALOADER_NUM_WORKERS`, etc.) live in `finetune_geniesim_instruction.sh`.

## 6) Notes

- Sync checkpoints out of `ephemeral` if needed.
- `decord` matters; training may use `torchcodec` with `decord` fallback.

```bash
rsync -avP /ephemeral/gr00t_models/<run_dir> <user>@<host>:/path/to/gr00t_models/
```

## 7) Preprocess (required unless preprocessed data is already in `PREPROCESSED_ROOT`)

Skip if §3 already filled `PREPROCESSED_ROOT`. Otherwise run unpack + batch preprocess (raw archives under `RAW_ROOT` only needed here).

Scripts: `verify_and_unpack_geniesim_archives.sh`, `preprocess_all_instruction_tasks.sh`, `preprocess_agibot_data.sh`, `preprocess_dataset.py`, `trim_static_frames.py`.

```bash
cd /home/$USER/iCode/VLA/Isaac-GR00T
bash scripts/preprocess_agibot/verify_and_unpack_geniesim_archives.sh --extract
source scripts/preprocess_agibot/training_paths.sh
SRC_ROOT="${RAW_ROOT}" DST_ROOT="${PREPROCESSED_ROOT}" bash scripts/preprocess_agibot/preprocess_all_instruction_tasks.sh
```

Raw video is often HEVC in `.mp4`; preprocessing typically rewrites to `mp4v` and uses `decord`.
