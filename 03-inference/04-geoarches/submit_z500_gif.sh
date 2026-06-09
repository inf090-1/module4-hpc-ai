#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=z500-vs-gt-%j.out

module load apptainer 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Defaults (can be overridden via: sbatch --export=ALL,VAR=...)
export SIF_NAME="${SIF_NAME:-pytorch_training_geoarches.sif}"
export RUN_DIR="${RUN_DIR:-$PWD/geoarches_real_run}"
export MODEL_NAME="${MODEL_NAME:-archesweathergen}"

# Enforce tiny download: by default we download only year=2020 and hour=0.
# This keeps the ERA5 download under ~5GB.
export RUN_YEARS_STR="${RUN_YEARS_STR:-2020}"
export RUN_HOURS_STR="${RUN_HOURS_STR:-0}"
export MAX_DATA_GB="${MAX_DATA_GB:-5}"

# Switches.
export DOWNLOAD_ASSETS="${DOWNLOAD_ASSETS:-0}"
export DOWNLOAD_DATA="${DOWNLOAD_DATA:-1}"
export DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-1}"
export RUN_INFER="${RUN_INFER:-1}"

./run_z500_gif.sh
