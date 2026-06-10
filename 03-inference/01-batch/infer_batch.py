import argparse
import os
import sys
import time
from typing import Optional

import numpy as np
import torch


# Make it possible to run this script directly from this folder.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COURSE_ROOT = os.path.dirname(SCRIPT_DIR)  # 03-inference/
sys.path.insert(0, COURSE_ROOT)

from weather_model import (  # noqa: E402
    WeatherNet,
    make_synthetic_batch,
    rollout_model,
    rollout_physics,
    weather_physics_step,
)


def _device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _maybe_import_viz():
    # Local import so the rest of the lesson can run without viz deps.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: WPS433
        import imageio.v2 as imageio

        return plt, imageio
    except ImportError as e:
        raise SystemExit(
            "Missing visualization deps. Install with:\n"
            "  pip install matplotlib imageio\n"
            f"Import error: {e}"
        )


def train_quick(
    model: torch.nn.Module,
    *,
    seed: int,
    nx: int,
    ny: int,
    nvars: int,
    hidden: int,
    steps: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    alpha: float,
) -> None:
    """Quickly fit the model to the synthetic one-step operator.

    This keeps the inference lessons runnable even without a pre-trained checkpoint.
    """

    # Coordinate grids reused for both data generation and the physics target.
    i = torch.arange(nx, device=device, dtype=torch.float32).view(nx, 1)
    j = torch.arange(ny, device=device, dtype=torch.float32).view(1, ny)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for step in range(steps):
        # Deterministic but varied samples.
        sample_start = step * batch_size
        x0 = make_synthetic_batch(
            seed=seed,
            sample_start=sample_start,
            batch_size=batch_size,
            nx=nx,
            ny=ny,
            nvars=nvars,
            device=device,
            i=i,
            j=j,
        )
        y = weather_physics_step(
            x0,
            seed=seed,
            nx=nx,
            ny=ny,
            alpha=alpha,
            i=i,
            j=j,
        )

        pred = model(x0)
        loss = torch.mean((pred - y) ** 2)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (step + 1) % max(1, steps // 5) == 0:
            print(f"[train-demo] step {step+1:>5d}/{steps} loss={loss.item():.6f}")

    model.eval()


@torch.no_grad()
def benchmark_rollout(
    model: torch.nn.Module,
    x0: torch.Tensor,
    *,
    steps: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> tuple[float, float]:
    """Return (elapsed_seconds, samples_per_second)."""
    device = x0.device
    torch.cuda.synchronize() if device.type == "cuda" else None

    # Copy so both benchmarks start from the same x0.
    x = x0

    start = time.time()
    if use_amp and device.type == "cuda":
        autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        autocast_ctx = nullcontext()

    with autocast_ctx:
        with torch.inference_mode():
            for _ in range(steps):
                x = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.time() - start

    # "samples" here means: how many (batch items) times how many forecast steps
    # were processed.
    samples_processed = float(x0.shape[0]) * float(steps)
    samples_per_sec = samples_processed / elapsed if elapsed > 0 else float("inf")
    return elapsed, samples_per_sec


class nullcontext:  # noqa: N801
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def create_forecast_gif(
    *,
    states_pred: list[torch.Tensor],
    states_true: Optional[list[torch.Tensor]],
    out_gif: str,
    var_idx: int,
    nx: int,
    ny: int,
    lat_min: float = -90.0,
    lat_max: float = 90.0,
    lon_min: float = -180.0,
    lon_max: float = 180.0,
    point_lat_idx: Optional[int] = None,
    point_lon_idx: Optional[int] = None,
    duration_s: float = 0.35,
) -> None:
    """Render a GIF showing the model weather evolution over forecast steps."""

    plt, imageio = _maybe_import_viz()

    steps = len(states_pred) - 1
    device = states_pred[0].device
    _ = device  # for clarity

    if point_lat_idx is None:
        point_lat_idx = nx // 2
    if point_lon_idx is None:
        point_lon_idx = ny // 2

    # Precompute data on CPU for consistent rendering.
    pred_fields = []
    true_fields = [] if states_true is not None else None
    pred_point_series = []
    true_point_series = [] if states_true is not None else None

    for t in range(steps + 1):
        pred_field = (
            states_pred[t][0, var_idx].detach().float().cpu().numpy()
        )  # (nx, ny)
        pred_fields.append(pred_field)
        pred_point_series.append(float(pred_field[point_lat_idx, point_lon_idx]))

        if states_true is not None:
            true_field = (
                states_true[t][0, var_idx].detach().float().cpu().numpy()
            )
            true_fields.append(true_field)
            true_point_series.append(float(true_field[point_lat_idx, point_lon_idx]))

    # Fixed color scale across the whole GIF.
    all_fields = np.concatenate(
        [f.reshape(-1) for f in pred_fields]
        + ([] if states_true is None else [f.reshape(-1) for f in true_fields])
    )
    vmin = float(np.quantile(all_fields, 0.05))
    vmax = float(np.quantile(all_fields, 0.95))

    y_min = min(pred_point_series)
    y_max = max(pred_point_series)
    if states_true is not None:
        y_min = min(y_min, min(true_point_series))
        y_max = max(y_max, max(true_point_series))

    frames = []
    for t in range(steps + 1):
        fig = plt.figure(figsize=(12, 4.2), dpi=120)
        fig.suptitle(f"WeatherNet forecast evolution (t={t})", fontsize=14)

        # Layout: (heatmap[s]) + (line plot)
        if states_true is None:
            gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.1])
            ax0 = fig.add_subplot(gs[0, 0])
            ax1 = fig.add_subplot(gs[0, 1])

            im = ax0.imshow(
                pred_fields[t],
                origin="lower",
                extent=[lon_min, lon_max, lat_min, lat_max],
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
            )
            ax0.set_xlabel("Longitude")
            ax0.set_ylabel("Latitude")
            ax0.set_title(f"Predicted field (var {var_idx})")

        else:
            gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 2.2, 1.1])
            ax0 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax1 = fig.add_subplot(gs[0, 2])

            ax0.imshow(
                pred_fields[t],
                origin="lower",
                extent=[lon_min, lon_max, lat_min, lat_max],
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
            )
            ax0.set_xlabel("Longitude")
            ax0.set_ylabel("Latitude")
            ax0.set_title(f"Pred (var {var_idx})")

            ax2.imshow(
                true_fields[t],
                origin="lower",
                extent=[lon_min, lon_max, lat_min, lat_max],
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
            )
            ax2.set_xlabel("Longitude")
            ax2.set_ylabel("Latitude")
            ax2.set_title(f"Target (var {var_idx})")

        # Line plot of evolution at a single grid point.
        ax1.plot(range(t + 1), pred_point_series[: t + 1], color="tab:red", lw=2)
        if states_true is not None:
            ax1.plot(
                range(t + 1),
                true_point_series[: t + 1],
                color="tab:blue",
                lw=2,
                alpha=0.85,
            )
            ax1.legend(["Pred", "Target"], loc="best", fontsize=10)

        ax1.set_xlabel("Forecast step")
        ax1.set_ylabel(f"Value @ (lat={point_lat_idx}, lon={point_lon_idx})")
        ax1.set_ylim(y_min - 0.05 * (abs(y_max) + 1e-6), y_max + 0.05 * (abs(y_max) + 1e-6))
        ax1.grid(True, alpha=0.25)
        ax1.set_title("Point-time evolution")

        # Matplotlib compatibility:
        # - Some versions no longer expose `FigureCanvasAgg.tostring_rgb()`.
        # - `buffer_rgba()` is more robust; convert RGBA -> RGB.
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())  # (H, W, 4), uint8
        img = rgba[..., :3]
        frames.append(img)
        plt.close(fig)

    os.makedirs(os.path.dirname(out_gif) or ".", exist_ok=True)
    imageio.mimsave(out_gif, frames, duration=duration_s)
    print(f"[viz] Wrote GIF: {out_gif}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WeatherNet inference demo + GIF visualization (MI300X friendly)."
    )

    # Grid + model.
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--vars", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=64)

    # Synthetic data.
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-start", type=int, default=0)

    # Forecast horizon.
    parser.add_argument("--forecast-steps", type=int, default=8)
    parser.add_argument("--var-idx", type=int, default=0)
    parser.add_argument("--point-lat-idx", type=int, default=None)
    parser.add_argument("--point-lon-idx", type=int, default=None)

    # Checkpoint.
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="weathernet_infer_demo.pt",
        help="State_dict checkpoint path. If missing, optionally auto-trains a small demo model.",
    )
    parser.add_argument(
        "--auto-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If checkpoint is missing, quickly train on the synthetic one-step operator (default: on).",
    )
    parser.add_argument("--auto-train-steps", type=int, default=120)
    parser.add_argument("--auto-train-batch-size", type=int, default=4)
    parser.add_argument("--auto-train-lr", type=float, default=3e-3)

    # Mixed precision.
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Run inference/benchmark with autocast BF16/FP16 (faster on MI300X).",
    )
    parser.add_argument(
        "--amp-dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
    )

    # Performance benchmark.
    parser.add_argument("--benchmark", action="store_true", help="Benchmark FP32 vs AMP rollouts")
    parser.add_argument("--benchmark-steps", type=int, default=16)
    parser.add_argument("--no-benchmark-gif", action="store_true", help="Only run visualization (skip FP32/AMP benchmark).")

    # Visualization.
    parser.add_argument("--out-gif", type=str, default="weather_forecast.gif")
    parser.add_argument("--gif-duration", type=float, default=0.35)

    # Repro.
    parser.add_argument("--deterministic", action="store_true")

    # Optional inference optimization.
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Try torch.compile(model) for faster inference (if supported).",
    )

    args = parser.parse_args()

    if args.deterministic:
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)

    device = _device()
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    print(f"[infer] Device: {device}")

    model = WeatherNet(nvars=args.vars, hidden=args.hidden).to(device)

    # Load or (optionally) quick-train demo weights.
    if args.checkpoint and os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        print(f"[infer] Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"[infer] No checkpoint found at: {args.checkpoint}")
        if args.auto_train:
            print("[infer] Auto-training a small demo model (for runnable examples)...")
            train_quick(
                model,
                seed=args.seed,
                nx=args.nx,
                ny=args.ny,
                nvars=args.vars,
                hidden=args.hidden,
                steps=args.auto_train_steps,
                batch_size=args.auto_train_batch_size,
                lr=args.auto_train_lr,
                device=device,
                alpha=0.1,
            )
            if args.checkpoint:
                torch.save(model.state_dict(), args.checkpoint)
                print(f"[infer] Saved demo checkpoint: {args.checkpoint}")
        else:
            print("[infer] Proceeding with random weights (GIF will still be generated).")

    model.eval()

    if args.compile_model and hasattr(torch, "compile") and device.type == "cuda":
        try:
            model = torch.compile(model)
            print("[infer] torch.compile enabled")
        except Exception as e:  # pragma: no cover
            print(f"[infer] torch.compile failed; continuing without it. Error: {e}")

    # Create an initial batch for inference.
    # We will forecast for the first item and use the rest only for benchmarking.
    x0 = make_synthetic_batch(
        seed=args.seed,
        sample_start=args.sample_start,
        batch_size=1 if not args.benchmark else 8,
        nx=args.nx,
        ny=args.ny,
        nvars=args.vars,
        device=device,
    )

    # Rollout (model predictions) for GIF.
    use_amp_for_gif = bool(args.use_amp and device.type == "cuda")
    print(f"[infer] Rolling out model for {args.forecast_steps} steps (use_amp={use_amp_for_gif})...")

    if use_amp_for_gif:
        states_pred = rollout_model(
            model,
            x0,
            args.forecast_steps,
            use_amp=True,
            amp_dtype=amp_dtype,
        )
    else:
        states_pred = rollout_model(model, x0, args.forecast_steps, use_amp=False)

    # Rollout for target/accuracy reference (synthetic physics).
    states_true = rollout_physics(
        seed=args.seed,
        x0=x0,
        steps=args.forecast_steps,
        nx=args.nx,
        ny=args.ny,
        alpha=0.1,
    )

    # Simple error metric over the whole rollout.
    errors = []
    for t in range(args.forecast_steps + 1):
        diff = states_pred[t] - states_true[t]
        mse = torch.mean(diff * diff).item()
        errors.append(mse)
    print("[infer] Rollout MSE over time:")
    for t, mse in enumerate(errors):
        print(f"  t={t:>2d}: mse={mse:.6f}")

    # Render GIF.
    create_forecast_gif(
        states_pred=states_pred,
        states_true=states_true,
        out_gif=args.out_gif,
        var_idx=args.var_idx,
        nx=args.nx,
        ny=args.ny,
        point_lat_idx=args.point_lat_idx,
        point_lon_idx=args.point_lon_idx,
        duration_s=args.gif_duration,
    )

    if args.no_benchmark_gif:
        return

    if args.benchmark:
        # Benchmark FP32 vs AMP.
        x0_bench = make_synthetic_batch(
            seed=args.seed,
            sample_start=args.sample_start,
            batch_size=16,
            nx=args.nx,
            ny=args.ny,
            nvars=args.vars,
            device=device,
        )

        print("\n[bench] FP32 rollout benchmark...")
        e_fp32, s_fp32 = benchmark_rollout(
            model,
            x0_bench,
            steps=args.benchmark_steps,
            use_amp=False,
            amp_dtype=amp_dtype,
        )
        print(f"[bench] FP32: {e_fp32:.3f}s | {s_fp32:.0f} samples/s")

        print("\n[bench] AMP rollout benchmark (autocast)...")
        e_amp, s_amp = benchmark_rollout(
            model,
            x0_bench,
            steps=args.benchmark_steps,
            use_amp=(device.type == "cuda"),
            amp_dtype=amp_dtype,
        )
        print(f"[bench] AMP: {e_amp:.3f}s | {s_amp:.0f} samples/s")
        print(f"[bench] Speedup: {s_fp32 / s_amp:.2f}x")


if __name__ == "__main__":
    main()
