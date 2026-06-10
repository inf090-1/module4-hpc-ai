# 3) Ensemble Weather Forecast Inference — GIF of mean + uncertainty

This lesson adds an **ensemble inference** example: instead of a single forecast, we run **E ensemble members in batch** and visualize:
- the **ensemble-mean prediction** (cartesian heatmap)
- the **ensemble-mean target** (synthetic physics)
- the **ensemble standard deviation** (spread across different initial conditions)
- a **time-series** at a chosen grid point with an uncertainty band

All on **MI300X** with optional AMP (BF16).

## What `ensemble_gif.py` actually does
- Builds an input batch of size `--ensemble` so each batch element is a different synthetic initial state.
- Runs the same WeatherNet rollout for all batch elements in parallel.
- For each forecast time step, it computes:
  - **ensemble mean** (average across the batch)
  - **ensemble std** (uncertainty/spread across the batch)
- Renders a GIF with:
  - heatmaps for the ensemble-mean prediction and the physics target
  - a pointwise time-series with an uncertainty band (mean +/- std)

## Checkpoint file (`.pt`)
- The default checkpoint `weathernet_infer_demo.pt` (or `--checkpoint` if you pass one) is a **PyTorch state_dict** for `WeatherNet`.
- It stores learned convolution weights for the synthetic **one-step** operator.
- If the checkpoint is missing, this script can **auto-train a small demo** model so the ensemble GIF still works.

## AMP optimization (`--use-amp`)
- `--use-amp` enables autocast during the ensemble rollout.
- `--amp-dtype bf16` chooses **BF16** precision for supported ops, which is usually faster on MI300X.
- Expect similar rollout quality, but the MSE/std values can differ slightly vs FP32 because of reduced precision.

---

## Files

- `ensemble_gif.py`
  - runs ensemble rollout inference
  - writes `weather_forecast_ensemble.gif`

- `submit_ensemble.sh`

---

## Install visualization deps (once)

```bash
pip install matplotlib imageio
```

---

## Local run (example)

```bash
cd 03-inference/03-ensemble
python ensemble_gif.py \
  --ensemble 4 \
  --forecast-steps 8 \
  --var-idx 0 \
  --use-amp --amp-dtype bf16 \
  --out-gif weather_forecast_ensemble.gif
```

---

## SLURM run

```bash
cd 03-inference/03-ensemble
sbatch submit_ensemble.sh
```

---

## Questions

1. Does the ensemble uncertainty (std) grow with forecast lead time?
2. Is the ensemble-mean prediction closer to the target than a single-member forecast?
3. Try `--point-lat-idx/--point-lon-idx`: do errors vary spatially?

## Comparison to the single-run lesson
- `01-batch/infer_batch.py` visualizes one prediction trajectory for one initial state.
- `03-ensemble/ensemble_gif.py` visualizes a distribution over initial states (mean + uncertainty) using the batch dimension.

## Ensemble meaning (important)
- This script does **not** train multiple different models.
- The ensemble spread comes from running the **same** WeatherNet model on multiple **different initial states** (different samples in the synthetic batch), then taking mean/std across the batch.
- If you do not pass `--point-lat-idx/--point-lon-idx`, the script uses the grid center: `nx//2, ny//2`.
