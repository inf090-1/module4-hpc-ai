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

# Bind host RUN_DIR into the container so downloads persist across runs.
# Avoid writing into /workspace (not writable in some container setups).
WORKDIR_IN_CONT="/work"
DATA_PATH_IN_CONT="${WORKDIR_IN_CONT}/data/era5_240/full/"
MODELSTORE_BASE_IN_CONT="${WORKDIR_IN_CONT}/modelstore"
MODEL_PATH_IN_CONT="${MODELSTORE_BASE_IN_CONT}/${MODEL_NAME}"
OUTPUT_DIR_IN_CONT="${WORKDIR_IN_CONT}/gifs/"

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

    # The model requires a set of normalization/statistics assets.
    # Download them if they're missing (more robust than relying solely on DOWNLOAD_ASSETS).
    if [ ! -f "geoarches/stats/era5-quantiles-2016_2022.nc" ] || \
       [ ! -f "geoarches/stats/archesweather_constant_masks.pt" ] || \
       [ ! -f "geoarches/stats/pangu_norm_stats2_with_w.pt" ] || \
       [ ! -f "geoarches/stats/pangu_norm_stats2.pt" ] || \
       [ ! -f "geoarches/stats/climatology_metrics.pt" ] || \
       [ ! -f "geoarches/stats/dcpp_spatial_norm_stats.pt" ] || \
       [ ! -f "geoarches/stats/delta24_stats.pt" ] || \
       [ ! -f "geoarches/stats/delta24_stats_with_w.pt" ] || \
       [ ! -f "geoarches/stats/deltapred24_aws_denorm.pt" ] || \
       [ ! -f "geoarches/stats/hres-quantiles-2016_2022.nc" ]; then
      echo '== Downloading normalization stats assets (missing required files) =='
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
        out_file="geoarches/stats/${f}"
        if [ -f "${out_file}" ]; then
          echo "  Using cached: ${f}"
        else
          # Quantile NetCDF files are published from HuggingFace in the upstream guide.
          if [[ "${f}" == era5-quantiles-* || "${f}" == hres-quantiles-* ]]; then
            wget --show-progress -O "${out_file}" "${HF_BASE_URL}/${f}"
          else
            wget --show-progress -O "${out_file}" "${GH_RAW_BASE}/${f}"
          fi
          if [ ! -s "${out_file}" ]; then
            echo "ERROR: download produced empty file: ${out_file}" >&2
            exit 3
          fi
        fi
      done

      # Hard verification: fail fast with a clear message.
      missing=0
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
        if [ ! -f "geoarches/stats/${f}" ]; then
          echo "ERROR: missing required asset: geoarches/stats/${f}" >&2
          missing=1
        fi
      done
      if [ "$missing" = "1" ]; then
        exit 2
      fi
    fi

    # Always validate quantile NetCDF files if they exist.
    # This catches the case where a previous download created corrupted/HTML files
    # with the right name but not the right format.
    if [ -f "geoarches/stats/era5-quantiles-2016_2022.nc" ] && [ -f "geoarches/stats/hres-quantiles-2016_2022.nc" ]; then
      validation_rc=0
      python - <<'PY' || validation_rc=$?
import os
import xarray as xr

check_files = [
    'geoarches/stats/era5-quantiles-2016_2022.nc',
    'geoarches/stats/hres-quantiles-2016_2022.nc',
]

bad = []
for p in check_files:
    ok = False
    for engine in ('netcdf4', 'scipy'):
        try:
            ds = xr.open_dataset(p, engine=engine)
            _ = list(ds.variables.keys())
            ok = True
            break
        except Exception:
            pass
    if not ok:
        bad.append(p)

if bad:
    print('BAD_XARRAY_NETCDF_ASSETS:' , bad)
    raise SystemExit(42)
print('NetCDF stats assets validation: OK')
PY

      if [ "${validation_rc}" -ne 0 ]; then
        echo 'NetCDF validation failed; re-downloading quantile assets...' >&2
        for f in era5-quantiles-2016_2022.nc hres-quantiles-2016_2022.nc; do
          rm -f "geoarches/stats/${f}"
          wget --show-progress -O "geoarches/stats/${f}" "${HF_BASE_URL}/${f}"
        done
      fi
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
        if fname.exists():
            print(f'[dl_era_limited] using cached {fname.name}')
        else:
            print(f'[dl_era_limited] writing {fname} (time={int(ds2.time.size)})')
            ds2.to_netcdf(fname)
PY
    fi

    if [ '${DOWNLOAD_MODELS}' = '1' ]; then
      echo '== Downloading pretrained model =='
       mkdir -p '${MODEL_PATH_IN_CONT}/checkpoints'
       if [ ! -f '${MODEL_PATH_IN_CONT}/checkpoints/checkpoint.ckpt' ]; then
         wget -q --show-progress -O '${MODEL_PATH_IN_CONT}/checkpoints/checkpoint.ckpt' \
           '${HF_BASE_URL}/${MODEL_NAME}_checkpoint.ckpt'
       else
         echo 'Using cached main model checkpoint'
       fi
       if [ ! -f '${MODEL_PATH_IN_CONT}/config.yaml' ]; then
         wget -q --show-progress -O '${MODEL_PATH_IN_CONT}/config.yaml' \
           '${HF_BASE_URL}/${MODEL_NAME}_config.yaml'
       else
         echo 'Using cached main model config'
       fi

      # ArchesWeatherGen depends on 4 deterministic weather-model checkpoints
      # referenced from its config.yaml.
       if [[ '${MODEL_NAME}' == archesweathergen* ]]; then
         for MOD in archesweather-m-seed0 archesweather-m-seed1 \
                     archesweather-m-skip-seed0 archesweather-m-skip-seed1; do
            echo "== Downloading deterministic model: ${MOD} =="
            mkdir -p "${MODELSTORE_BASE_IN_CONT}/${MOD}/checkpoints"
            if [ ! -f "${MODELSTORE_BASE_IN_CONT}/${MOD}/checkpoints/checkpoint.ckpt" ]; then
              wget -q --show-progress -O "${MODELSTORE_BASE_IN_CONT}/${MOD}/checkpoints/checkpoint.ckpt" \
                "${HF_BASE_URL}/${MOD}_checkpoint.ckpt"
            else
              echo "Using cached deterministic checkpoint for ${MOD}"
            fi
            if [ ! -f "${MODELSTORE_BASE_IN_CONT}/${MOD}/config.yaml" ]; then
              wget -q --show-progress -O "${MODELSTORE_BASE_IN_CONT}/${MOD}/config.yaml" \
                "${HF_BASE_URL}/${MOD}_config.yaml"
            else
              echo "Using cached deterministic config for ${MOD}"
            fi
          done
        fi

       # Validate a couple of key NetCDF assets with xarray.
      # If a previous run cached a corrupted download (e.g., an HTML error page saved as .nc),
      # xarray may refuse to open it with any backend.
      validation_rc=0
      python - <<'PY' || validation_rc=$?
import os
import sys
import xarray as xr

check_files = [
    'geoarches/stats/era5-quantiles-2016_2022.nc',
    'geoarches/stats/hres-quantiles-2016_2022.nc',
]

bad = []
for p in check_files:
    if not os.path.exists(p):
        bad.append(p)
        continue
    ok = False
    for engine in ('netcdf4', 'scipy'):
        try:
            ds = xr.open_dataset(p, engine=engine)
            # Force actual reading of metadata/coords.
            _ = list(ds.variables.keys())
            ok = True
            break
        except Exception:
            pass
    if not ok:
        bad.append(p)

if bad:
    print('BAD_XARRAY_NETCDF_ASSETS:', bad)
    sys.exit(42)
print('NetCDF stats assets validation: OK')
PY

      # If validation failed, retry downloads of those files (best effort).
      # (We purposely re-download only the quantile NetCDF files.)
      if [ "${validation_rc}" -ne 0 ]; then
        echo 'NetCDF validation failed; re-downloading quantile assets...' >&2
        for f in era5-quantiles-2016_2022.nc hres-quantiles-2016_2022.nc; do
          rm -f "geoarches/stats/${f}"
          wget --show-progress -O "geoarches/stats/${f}" "${HF_BASE_URL}/${f}"
        done

        echo 'Re-validating NetCDF assets...' >&2
        validation_rc=0
        python - <<'PY' || validation_rc=$?
import os
import xarray as xr

check_files = [
    'geoarches/stats/era5-quantiles-2016_2022.nc',
    'geoarches/stats/hres-quantiles-2016_2022.nc',
]

bad = []
for p in check_files:
    if not os.path.exists(p):
        bad.append(p)
        continue
    ok = False
    for engine in ('netcdf4', 'scipy'):
        try:
            ds = xr.open_dataset(p, engine=engine)
            _ = list(ds.variables.keys())
            ok = True
            break
        except Exception:
            pass
    if not ok:
        bad.append(p)

if bad:
    print('BAD_XARRAY_NETCDF_ASSETS_AFTER_REDOWNLOAD:', bad)
    raise SystemExit(43)
print('NetCDF stats assets validation (after retry): OK')
PY

      if [ "${validation_rc}" -ne 0 ]; then
        echo 'ERROR: NetCDF assets still invalid after retry.' >&2
        exit 44
      fi
      fi

     fi


    if [ '${RUN_INFER}' = '1' ]; then
      echo '== Running Z500 vs GT GIF =='

      # Newer images embed run_inference.py; older images embed the legacy
      # z500_vs_gt_make_gif_limited.py.
      if [ -f run_inference.py ]; then
        echo "[inner] run_inference.py args: --model=${MODEL_PATH_IN_CONT} --data-path=${DATA_PATH_IN_CONT} --output-dir=${OUTPUT_DIR_IN_CONT} --rollout-iterations=${ROLLOUT_ITERATIONS} --n-members=${N_MEMBERS} --cmap viridis"
         python run_inference.py \
           --model "${MODEL_PATH_IN_CONT}" \
           --data-path "${DATA_PATH_IN_CONT}" \
           --output-dir "${OUTPUT_DIR_IN_CONT}" \
           --rollout-iterations "${ROLLOUT_ITERATIONS}" \
           --n-members "${N_MEMBERS}" \
           --cmap viridis
      else
        echo "[inner] z500_vs_gt_make_gif_limited.py args: --model=${MODEL_PATH_IN_CONT} --data-path=${DATA_PATH_IN_CONT} --output-dir=${OUTPUT_DIR_IN_CONT} --rollout-iterations=${ROLLOUT_ITERATIONS} --n-members=${N_MEMBERS} --cmap viridis"
         python z500_vs_gt_make_gif_limited.py \
           --model "${MODEL_PATH_IN_CONT}" \
           --data-path "${DATA_PATH_IN_CONT}" \
           --output-dir "${OUTPUT_DIR_IN_CONT}" \
           --rollout-iterations "${ROLLOUT_ITERATIONS}" \
           --n-members "${N_MEMBERS}" \
           --cmap viridis
      fi
      # GIF is written directly to OUTPUT_DIR_IN_CONT (which is bound to the host RUN_DIR).
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
  sed "s|\${MODELSTORE_BASE_IN_CONT}|${MODELSTORE_BASE_IN_CONT}|g" | \
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
    --bind "${RUN_DIR}:${WORKDIR_IN_CONT}" \
    --bind "${RUN_DIR}/modelstore:/workspace/geoarches/modelstore" \
    --bind "${RUN_DIR}/geoarches/stats:/workspace/geoarches/geoarches/stats" \
    "$SIF_NAME" \
    bash -lc "${INNER_CMD}"
else
  apptainer exec --writable-tmpfs \
    "${APPTAINER_ACCEL_ARGS[@]}" \
    --bind "${RUN_DIR}:${WORKDIR_IN_CONT}" \
    --bind "${RUN_DIR}/modelstore:/workspace/geoarches/modelstore" \
    --bind "${RUN_DIR}/geoarches/stats:/workspace/geoarches/geoarches/stats" \
    "$SIF_NAME" \
    bash -lc "${INNER_CMD}"
fi
