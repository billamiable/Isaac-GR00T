# Updated User Guide (Simple Training Flow)

Convention: code under `home`, large data under `ephemeral`.

Test environment used for this guide:

- Platform: NVIDIA Brev
- Node: 1x H100 node
- GPUs: 8 GPUs on the same node

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

## 3) Downloads

| Script | Default role |
| ------ | ------------ |
| [`download_gr00t_n1d6_base.sh`](download_gr00t_n1d6_base.sh) | [GR00T-N1.6-3B](https://huggingface.co/nvidia/GR00T-N1.6-3B) → `BASE_MODEL_PATH` |
| [`download_instruction_preprocessed_hf.sh`](download_instruction_preprocessed_hf.sh) | Download preprocessed dataset from HF into `PREPROCESSED_ROOT` |
| [`download_geniesim_instruction_modelscope.sh`](download_geniesim_instruction_modelscope.sh) | Download raw instruction from ModelScope into `RAW_ROOT`; needs `uv pip install modelscope` |

Pick one dataset path (A or B), plus base model download:

```bash
cd /home/$USER/iCode/VLA/Isaac-GR00T
bash scripts/preprocess_agibot/download_gr00t_n1d6_base.sh
```

```bash
# A) Use preprocessed dataset directly (skip local preprocess)
bash scripts/preprocess_agibot/download_instruction_preprocessed_hf.sh
```

```bash
# B) Use raw dataset and run preprocess locally later (§7)
uv pip install modelscope
bash scripts/preprocess_agibot/download_geniesim_instruction_modelscope.sh
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


## 8) Benchmark (Task Suite Batch Run)

Reference first: [Genie Sim User Guide - Batch Run Task Suite](https://agibot-world.com/sim-evaluation/docs/#/v3?id=_315-batch-run-task-suite).

Use this to run local task-suite benchmark against a GR00T websocket inference service.

1. Clone simulator repo and build docker image:

```bash
git clone https://github.com/AgibotTech/genie_sim.git
cd genie_sim
docker build -f ./scripts/dockerfile -t registry.agibot.com/genie-sim/open_source:latest .
```

2. Download GenieSim assets and place them under `source/geniesim/assets`:

```bash
git clone https://modelscope.cn/datasets/agibot_world/GenieSimAssets.git -b rolling
```

3. Start simulator GUI from repo root:

```bash
cd genie_sim
./scripts/start_gui.sh
```

4. Open a new terminal and enter the container:

```bash
cd genie_sim
./scripts/into.sh
```

5. In the GR00T repo/container, start websocket inference server and note the printed `ip:port`:

```bash
uv run --extra websocket python scripts/deployment/serve_gr00t_websocket.py \
  --model-path /path/to/checkpoints/output_dir \
  --port 8000
```

6. In benchmark container, run IF benchmark on local host.

```bash
./scripts/run_batch_tasks.sh --num-episode 3 --type if --infer-host {ip:port}
```

Local benchmark note:

- We run this on local machine.
- Final reported results use `--num-episode 3`; for quick smoke testing, `--num-episode 1` is acceptable.
- Runtime is long: about 2 hours per run, around 6 hours total for 3 episodes.

## Appendix A) Preprocess Comparison Workflow (raw vs preprocessed)

Use this when you want to verify where visual differences come from in the preprocess chain.

Pipeline reminder (video side):

1. decode raw frame(s)
2. resize + pad to `256x256`
3. write `mp4v` video
4. decode + trim by `instruction_segments`
5. write `mp4v` again
6. training reads (decodes) the final video

Because trim changes frame indices, comparison must align frames by trim mapping (`raw_frame = raw_trim_start + pre_step`), not by raw index directly.

### Three setups used in comparison

- **Setup 1**: `raw_preprocess_like vs pre_decoded`  
  Compare raw frame after direct `resize+pad` (no re-encode) against decoded frame from preprocessed video.
- **Setup 2**: `simulated_redecoded vs pre_decoded`  
  Compare raw frame after one `mp4v` save+decode against preprocessed decode.
- **Setup 3**: `simulated_full_pipeline_redecoded vs pre_decoded`  
  Compare raw frame after full pipeline replay (`resize/pad -> save -> trim -> save -> decode`) against preprocessed decode.

### Run the comparison script

```bash
cd /home/$USER/iCode/VLA/Isaac-GR00T
uv run python scripts/preprocess_agibot/compare_preprocessed_sample.py \
  --task pick_block_color \
  --episode 0 \
  --pre-step 0
```

Outputs:

- compact metrics: terminal JSON (`video_mae`, `simulated_video_mae`, `simulated_full_pipeline_video_mae`, etc.)
- detailed report: `debug_outputs/preprocess_compare/<task>/episode_<id>/step_<id>/summary.json`
- visual diffs: `.../images/_diff_viz/*_diff_grid.png`

Interpretation tip: if Setup 3 is near zero while Setup 1/2 are not, then differences are from incomplete replay of the preprocess chain (typically re-encode stages), not random drift.

Current visual examples (copied under `scripts/preprocess_agibot/preprocess_compare_figures/`):

![Top head diff grid](preprocess_compare_figures/top_head_diff_grid.png)
![Hand left diff grid](preprocess_compare_figures/hand_left_diff_grid.png)
![Hand right diff grid](preprocess_compare_figures/hand_right_diff_grid.png)

Current analysis:

- From these visualizations, differences are hard to perceive by eye for Setup 2/3.
- In our subset rerun check, Setup 3 reaches near-zero/zero frame and processor differences, indicating preprocess image transformation is likely not the dominant source of downstream mismatch.
