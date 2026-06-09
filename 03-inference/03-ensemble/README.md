# 3) Ensemble Weather Forecast Inference — GIF of mean + uncertainty

This lesson adds an **ensemble inference** example: instead of a single forecast, we run **E ensemble members in batch** and visualize:
- the **ensemble-mean prediction** (cartesian heatmap)
- the **ensemble-mean target** (synthetic physics)
- the **ensemble standard deviation** (model uncertainty)
- a **time-series** at a chosen grid point with an uncertainty band

All on **MI300X** with optional AMP (BF16).

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
