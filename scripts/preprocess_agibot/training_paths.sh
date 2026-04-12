#!/usr/bin/env bash
#
# Unified path config for GenieSim -> GR00T post-training.
# Edit these exports once for a new node, then either:
#   1) source this file manually before running commands, or
#   2) add the following line to ~/.bashrc:
#      source /home/$USER/iCode/VLA/Isaac-GR00T/scripts/preprocess_agibot/training_paths.sh
#
# Convention:
# - code lives under /home
# - large data and outputs live under /ephemeral

export RAW_ROOT="${RAW_ROOT:-/ephemeral/agibot/instruction}"
export PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-/ephemeral/agibot/instruction_preprocessed}"
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-/ephemeral/models/GR00T-N1.6-3B}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/ephemeral/gr00t_models}"

# Optional overrides:
# export PYTHON_BIN="/home/$USER/iCode/VLA/Isaac-GR00T/.venv/bin/python"
# export COLOR_JITTER_PARAMS="brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08"
