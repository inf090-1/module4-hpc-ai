#!/bin/bash
set -euo pipefail

# Real-data inference (ERA5 Z500) using GeoArches + pre-trained ArchesWeatherGen.
# This wrapper uses Apptainer instead of Docker and is optimized for tiny demo downloads.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SIF_NAME="${SIF_NAME:-pytorch_training_geoarches.sif}"

# Persist data/models/outputs on the host.
RUN_DIR="${RUN_DIR:-$PWD/geoarches_real_run}"

# Which pretrained model to use (HuggingFace artifact name).
MODEL_NAME="${MODEL_NAME:-archesweathergen}"

# Forecast settings.
ROLLOUT_ITERATIONS="${ROLLOUT_ITERATIONS:-10}"
N_MEMBERS="${N_MEMBERS:-25}"

# Download controls.
# dl_era.py downloads ERA5 as netcdf files of (year, hour) at fixed 240x121 resolution.
# Each (year,hour) file is ~4.6GB, so to stay under ~5GB we force downloading
# at most ONE (year,hour) file.
MAX_DATA_GB="${MAX_DATA_GB:-5}"

# dl_era.py can segfault when writing very large NetCDF files in some environments.
# To keep this classroom demo stable, we limit the number of ERA5 timesteps written
# per (year,hour) file.
#
# With ERA5 6-hourly cadence, 180 timesteps is enough for rollout GIFs while
# staying under the MAX_DATA_GB constraint.
MAX_TIME_STEPS="${MAX_TIME_STEPS:-180}"

# Space-separated lists, e.g. RUN_YEARS='2020' RUN_HOURS='0'
RUN_YEARS_STR="${RUN_YEARS_STR:-2020}"
RUN_HOURS_STR="${RUN_HOURS_STR:-0}"

# Download switches.
DOWNLOAD_DATA="${DOWNLOAD_DATA:-0}"      # 1 downloads ERA5 samples
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-0}"  # 1 downloads checkpoint/config
DOWNLOAD_ASSETS="${DOWNLOAD_ASSETS:-0}"  # 1 downloads normalization stats
RUN_INFER="${RUN_INFER:-1}"

# Parse lists into arrays.
read -r -a RUN_YEARS <<<"${RUN_YEARS_STR}"
read -r -a RUN_HOURS <<<"${RUN_HOURS_STR}"

if [ "${#RUN_YEARS[@]}" -eq 0 ] || [ "${#RUN_HOURS[@]}" -eq 0 ]; then
  echo "ERROR: RUN_YEARS_STR and RUN_HOURS_STR must be non-empty" >&2
  exit 1
fi

# Estimate size: ~4.58GB per (year,hour) file (dl_era.py uses a 4.58e9 bytes corruption threshold).
APPROX_GB_PER_FILE=4.58
requested_files=$(( ${#RUN_YEARS[@]} * ${#RUN_HOURS[@]} ))
estimated_gb=$(python - <<PY
import math
print(${APPROX_GB_PER_FILE} * ${requested_files})
PY
)

if (( requested_files > 1 )); then
  if (( estimated_gb > MAX_DATA_GB )); then
    echo "[limited] Requested ${requested_files} (year,hour) files (~${estimated_gb}GB), exceeding MAX_DATA_GB=${MAX_DATA_GB}GB." >&2
    echo "[limited] Forcing download to a single file to keep data small." >&2
    # Keep just first year and first hour.
    RUN_YEARS=("${RUN_YEARS[0]}")
    RUN_HOURS=("${RUN_HOURS[0]}")
  fi
fi

# Build download args.
YEARS_ARGS="${RUN_YEARS[*]}"
HOURS_ARGS="${RUN_HOURS[*]}"

mkdir -p "$RUN_DIR/data/era5_240/full" \
         "$RUN_DIR/modelstore" \
         "$RUN_DIR/gifs" \
         "$RUN_DIR/geoarches/stats"

if [ ! -f "$SIF_NAME" ]; then
  echo "ERROR: Apptainer image not found: $SIF_NAME" >&2
  echo "Build it with: $SCRIPT_DIR/build_rocm_container.sh" >&2
  exit 1
fi

HF_BASE_URL="https://huggingface.co/gcouairon/ArchesWeather/resolve/main"
GH_RAW_BASE="https://raw.githubusercontent.com/INRIA/geoarches/90cc5fe/geoarches/stats"

# Paths *inside the container* (no host bind-mounts required).
# Files won't persist outside the container unless this wrapper copies them out.
# Use a /tmp-backed path inside the container for large writes.
# Writing under /workspace can crash with some overlay configurations.
DATA_PATH_IN_CONT="/tmp/geoarches_data/era5_240/full/"
MODEL_PATH_IN_CONT="modelstore/${MODEL_NAME}"
OUTPUT_DIR_IN_CONT="/tmp/geoarches_data/gifs/"

echo "=== Apptainer real-data inference (GeoArches) ==="
echo "SIF: $SIF_NAME"
echo "RUN_DIR: $RUN_DIR"
echo "MODEL_NAME: $MODEL_NAME"
echo "ROLLOUT_ITERATIONS: $ROLLOUT_ITERATIONS"
echo "N_MEMBERS: $N_MEMBERS"
echo "Download data: ${DOWNLOAD_DATA} (MAX_DATA_GB=${MAX_DATA_GB}GB, MAX_TIME_STEPS=${MAX_TIME_STEPS})"
echo "  downloading years: ${RUN_YEARS[*]}"
echo "  downloading hours: ${RUN_HOURS[*]}"

# Execute inside container.
# Apptainer GPU passthrough uses ONLY `--rocm` (AMD) or `--nv` (NVIDIA).
# We do NOT manually map /dev/kfd or /dev/dri.
APPTAINER_ACCEL_ARGS=()

# Force modes explicitly if requested.
if [ "${APPTAINER_USE_ROCM:-0}" = "1" ]; then
  APPTAINER_ACCEL_ARGS+=(--rocm)
elif [ "${APPTAINER_USE_NV:-0}" = "1" ]; then
  APPTAINER_ACCEL_ARGS+=(--nv)
else
  # Auto-detect for AMD.
  if [ -e /dev/kfd ]; then
    APPTAINER_ACCEL_ARGS+=(--rocm)
  fi
fi

mkdir -p "${RUN_DIR}/gifs"

# Use a separate script file to avoid heredoc escaping issues
cat > /tmp/inner_cmd.sh <<'INNER_EOF'
    set -euo pipefail
    cd /workspace/geoarches

    if [ '${DOWNLOAD_ASSETS}' = '1' ]; then
      echo '== Downloading normalization stats assets =='
      mkdir -p geoarches/stats

      # Core stats files required by the model
      for f in \
        era5-quantiles-2016_2022.nc \
        pangu_norm_stats2_with_w.pt \
        pangu_norm_stats2.pt \
        archesweather_constant_masks.pt \
        climatology_metrics.pt \
        dcpp_spatial_norm_stats.pt \
        delta24_stats.pt \
        delta24_stats_with_w.pt \
        deltapred24_aws_denorm.pt \
        hres-quantiles-2016_2022.nc; do
        echo "  Downloading ${f}..."
        wget -q --show-progress -O "geoarches/stats/${f}" \
          "${GH_RAW_BASE}/${f}"
      done
    fi

    if [ '${DOWNLOAD_DATA}' = '1' ]; then
      echo '== Downloading ERA5 (limited to keep <=5GB) =='
      python - <<PY
import xarray as xr
from pathlib import Path

obs_path = 'gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr'

out_folder = Path('${DATA_PATH_IN_CONT}')
out_folder.mkdir(parents=True, exist_ok=True)

years = [int(x) for x in '${YEARS_ARGS}'.split() if x]
hours = [int(x) for x in '${HOURS_ARGS}'.split() if x]
max_steps = int('${MAX_TIME_STEPS}')

obs_xarr = xr.open_zarr(obs_path)
for year in years:
    ds_year = obs_xarr.sel(time=obs_xarr.time.dt.year.isin([year]))
    for hour in hours:
        ds2 = ds_year.sel(time=ds_year.time.dt.hour.isin([hour]))
        if ds2.time.size > max_steps:
            ds2 = ds2.isel(time=slice(0, max_steps))

        fname = out_folder / f'era5_240_{year}_{hour}h.nc'
        print(f'[dl_era_limited] writing {fname} (time={int(ds2.time.size)})')
        ds2.to_netcdf(fname)
PY
    fi

    if [ '${DOWNLOAD_MODELS}' = '1' ]; then
      echo '== Downloading pretrained model =='
      mkdir -p '${MODEL_PATH_IN_CONT}/checkpoints'
      wget -q --show-progress -O '${MODEL_PATH_IN_CONT}/checkpoints/checkpoint.ckpt' \
        '${HF_BASE_URL}/${MODEL_NAME}_checkpoint.ckpt'
      wget -q --show-progress -O '${MODEL_PATH_IN_CONT}/config.yaml' \
        '${HF_BASE_URL}/${MODEL_NAME}_config.yaml'

      # ArchesWeatherGen depends on 4 deterministic weather-model checkpoints
      # referenced from its config.yaml.
      if [[ '${MODEL_NAME}' == archesweathergen* ]]; then
        for MOD in archesweather-m-seed0 archesweather-m-seed1 \
                   archesweather-m-skip-seed0 archesweather-m-skip-seed1; do
          echo "== Downloading deterministic model: ${MOD} =="
          mkdir -p "modelstore/${MOD}/checkpoints"
          wget -q --show-progress -O "modelstore/${MOD}/checkpoints/checkpoint.ckpt" \
            "${HF_BASE_URL}/${MOD}_checkpoint.ckpt"
          wget -q --show-progress -O "modelstore/${MOD}/config.yaml" \
            "${HF_BASE_URL}/${MOD}_config.yaml"
        done
      fi
    fi

    if [ '${RUN_INFER}' = '1' ]; then
      echo '== Running Z500 vs GT GIF =='

      # Newer images embed run_inference.py; older images embed the legacy
      # z500_vs_gt_make_gif_limited.py.
      if [ -f run_inference.py ]; then
        python run_inference.py \\
          --model '${MODEL_PATH_IN_CONT}' \\
          --data-path '${DATA_PATH_IN_CONT}' \\
          --output-dir '${OUTPUT_DIR_IN_CONT}' \\
          --rollout-iterations '${ROLLOUT_ITERATIONS}' \\
          --n-members '${N_MEMBERS}' \\
          --cmap viridis
      else
        python z500_vs_gt_make_gif_limited.py \\
          --model '${MODEL_PATH_IN_CONT}' \\
          --data-path '${DATA_PATH_IN_CONT}' \\
          --output-dir '${OUTPUT_DIR_IN_CONT}' \\
          --rollout-iterations '${ROLLOUT_ITERATIONS}' \\
          --n-members '${N_MEMBERS}' \\
          --cmap viridis
      fi

      tar -c '${OUTPUT_DIR_IN_CONT}Z500.gif' >&3
    fi
INNER_EOF

# Substitute environment variables into the inner script
INNER_CMD=$(cat /tmp/inner_cmd.sh | \
  sed "s|\${DOWNLOAD_ASSETS}|${DOWNLOAD_ASSETS}|g" | \
  sed "s|\${GH_RAW_BASE}|${GH_RAW_BASE}|g" | \
  sed "s|\${DOWNLOAD_DATA}|${DOWNLOAD_DATA}|g" | \
  sed "s|\${DATA_PATH_IN_CONT}|${DATA_PATH_IN_CONT}|g" | \
  sed "s|\${YEARS_ARGS}|${YEARS_ARGS}|g" | \
  sed "s|\${HOURS_ARGS}|${HOURS_ARGS}|g" | \
  sed "s|\${MAX_TIME_STEPS}|${MAX_TIME_STEPS}|g" | \
  sed "s|\${DOWNLOAD_MODELS}|${DOWNLOAD_MODELS}|g" | \
  sed "s|\${MODEL_PATH_IN_CONT}|${MODEL_PATH_IN_CONT}|g" | \
  sed "s|\${HF_BASE_URL}|${HF_BASE_URL}|g" | \
  sed "s|\${MODEL_NAME}|${MODEL_NAME}|g" | \
  sed "s|\${RUN_INFER}|${RUN_INFER}|g" | \
  sed "s|\${OUTPUT_DIR_IN_CONT}|${OUTPUT_DIR_IN_CONT}|g" | \
  sed "s|\${ROLLOUT_ITERATIONS}|${ROLLOUT_ITERATIONS}|g" | \
  sed "s|\${N_MEMBERS}|${N_MEMBERS}|g")

rm /tmp/inner_cmd.sh

if [ "${RUN_INFER}" = "1" ]; then
  apptainer exec --writable-tmpfs \
    "${APPTAINER_ACCEL_ARGS[@]}" \
    "$SIF_NAME" \
    bash -lc "exec 3>&1 1>&2
${INNER_CMD}" | tar -x -C "${RUN_DIR}"
else
  # No GIF archive to extract, so do NOT pipe into tar.
  apptainer exec --writable-tmpfs \
    "${APPTAINER_ACCEL_ARGS[@]}" \
    "$SIF_NAME" \
    bash -lc "${INNER_CMD}"
fi
