import argparse
import os
import sys
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
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import imageio.v2 as imageio

        return plt, imageio
    except ImportError as e:
        raise SystemExit(
            "Missing visualization deps. Install with:\n"
            "  pip install matplotlib imageio\n"
            f"Import error: {e}"
        )


def _train_quick(model: torch.nn.Module, *, seed: int, nx: int, ny: int, nvars: int, steps: int, batch_size: int, lr: float, device: torch.device, alpha: float) -> None:
    # Small demo fit to the synthetic one-step operator.
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    i = torch.arange(nx, device=device, dtype=torch.float32).view(nx, 1)
    j = torch.arange(ny, device=device, dtype=torch.float32).view(1, ny)

    for step in range(steps):
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
            print(f"[train-demo] step {step+1:>4d}/{steps} loss={loss.item():.6f}")

    model.eval()


def _make_frames(
    *,
    mean_pred: list[np.ndarray],
    mean_true: list[np.ndarray],
    std_pred: list[np.ndarray],
    point_lat_idx: int,
    point_lon_idx: int,
    out_gif: str,
    var_idx: int,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    duration_s: float,
) -> None:
    plt, imageio = _maybe_import_viz()

    T = len(mean_pred) - 1

    # Fixed color scale across frames.
    all_fields = np.concatenate(
        [f.reshape(-1) for f in mean_pred] + [f.reshape(-1) for f in mean_true]
    )
    vmin = float(np.quantile(all_fields, 0.05))
    vmax = float(np.quantile(all_fields, 0.95))

    # Std color scale can be separate.
    all_std = np.concatenate([f.reshape(-1) for f in std_pred])
    std_vmax = float(np.quantile(all_std, 0.98))
    std_vmin = 0.0

    pred_point_mean = [float(f[point_lat_idx, point_lon_idx]) for f in mean_pred]
    pred_point_std = [float(f[point_lat_idx, point_lon_idx]) for f in std_pred]
    true_point_mean = [float(f[point_lat_idx, point_lon_idx]) for f in mean_true]

    os.makedirs(os.path.dirname(out_gif) or ".", exist_ok=True)

    frames = []
    for t in range(T + 1):
        fig = plt.figure(figsize=(14, 4.6), dpi=120)
        fig.suptitle(f"Weather ensemble forecast (t={t}), var={var_idx}", fontsize=14)

        gs = fig.add_gridspec(1, 3, width_ratios=[2.1, 2.1, 1.1])
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 2])

        im0 = ax0.imshow(
            mean_pred[t],
            origin="lower",
            extent=[lon_min, lon_max, lat_min, lat_max],
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        ax0.set_xlabel("Longitude")
        ax0.set_ylabel("Latitude")
        ax0.set_title("Pred mean")

        im1 = ax1.imshow(
            mean_true[t],
            origin="lower",
            extent=[lon_min, lon_max, lat_min, lat_max],
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        ax1.set_xlabel("Longitude")
        ax1.set_ylabel("Latitude")
        ax1.set_title("Target mean")

        # Right panel: point evolution with uncertainty.
        xs = list(range(t + 1))
        pm = pred_point_mean[: t + 1]
        ps = pred_point_std[: t + 1]
        tm = true_point_mean[: t + 1]

        ax2.plot(xs, pm, color="tab:red", lw=2, label="Pred mean")
        ax2.fill_between(
            xs,
            [m - s for m, s in zip(pm, ps)],
            [m + s for m, s in zip(pm, ps)],
            color="tab:red",
            alpha=0.25,
            label="Pred ± std",
        )
        ax2.plot(xs, tm, color="tab:blue", lw=2, alpha=0.9, label="Target mean")

        ax2.set_xlabel("Forecast step")
        ax2.set_ylabel(f"Value @ ({point_lat_idx},{point_lon_idx})")
        ax2.grid(True, alpha=0.25)
        ax2.legend(loc="best", fontsize=9)

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(img)
        plt.close(fig)

    imageio.mimsave(out_gif, frames, duration=duration_s)
    print(f"[viz] Wrote GIF: {out_gif}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WeatherNet ensemble inference rollout + GIF visualization"
    )

    # Grid + model.
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--vars", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=64)

    # Ensemble + initial conditions.
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--ensemble", type=int, default=4)

    # Forecast horizon.
    parser.add_argument("--forecast-steps", type=int, default=8)
    parser.add_argument("--var-idx", type=int, default=0)
    parser.add_argument("--point-lat-idx", type=int, default=None)
    parser.add_argument("--point-lon-idx", type=int, default=None)

    # Checkpoint.
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="../01-batch/weathernet_infer_demo.pt",
        help="State_dict checkpoint path. If missing, can optionally auto-train.",
    )
    parser.add_argument(
        "--auto-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If checkpoint missing, auto-train a small demo model.",
    )
    parser.add_argument("--auto-train-steps", type=int, default=80)
    parser.add_argument("--auto-train-batch-size", type=int, default=4)
    parser.add_argument("--auto-train-lr", type=float, default=3e-3)
    parser.add_argument("--alpha", type=float, default=0.1)

    # Mixed precision.
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16"])

    # Model opt.
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Try torch.compile(model) for inference (if supported).",
    )

    # Visualization.
    parser.add_argument("--out-gif", type=str, default="weather_forecast_ensemble.gif")
    parser.add_argument("--gif-duration", type=float, default=0.35)

    args = parser.parse_args()

    device = _device()
    print(f"[infer-ens] Device: {device}")

    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    model = WeatherNet(nvars=args.vars, hidden=args.hidden).to(device)

    # Load checkpoint or auto-train.
    if args.checkpoint and os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        print(f"[infer-ens] Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"[infer-ens] Checkpoint not found: {args.checkpoint}")
        if args.auto_train:
            print("[infer-ens] Auto-training a small demo model (runnable classroom example)...")
            _train_quick(
                model,
                seed=args.seed,
                nx=args.nx,
                ny=args.ny,
                nvars=args.vars,
                steps=args.auto_train_steps,
                batch_size=args.auto_train_batch_size,
                lr=args.auto_train_lr,
                device=device,
                alpha=args.alpha,
            )
        else:
            print("[infer-ens] Proceeding with random weights (GIF will still be generated).")

    model.eval()

    if args.compile_model and hasattr(torch, "compile") and device.type == "cuda":
        try:
            model = torch.compile(model)
            print("[infer-ens] torch.compile enabled")
        except Exception as e:  # pragma: no cover
            print(f"[infer-ens] torch.compile failed; continuing. Error: {e}")

    # Initial ensemble batch: B=ensemble.
    x0 = make_synthetic_batch(
        seed=args.seed,
        sample_start=args.sample_start,
        batch_size=args.ensemble,
        nx=args.nx,
        ny=args.ny,
        nvars=args.vars,
        device=device,
    )

    # Rollout.
    use_amp = bool(args.use_amp and device.type == "cuda")

    if use_amp:
        states_pred = rollout_model(
            model,
            x0,
            args.forecast_steps,
            use_amp=True,
            amp_dtype=amp_dtype,
        )
    else:
        states_pred = rollout_model(model, x0, args.forecast_steps, use_amp=False)

    # Physics targets: same initial conditions, same seed; structured term uses seed.
    states_true = rollout_physics(
        seed=args.seed,
        x0=x0,
        steps=args.forecast_steps,
        nx=args.nx,
        ny=args.ny,
        alpha=args.alpha,
    )

    # Convert to numpy arrays for visualization.
    # Each list element corresponds to time t; arrays are (nx, ny).
    mean_pred_fields: list[np.ndarray] = []
    mean_true_fields: list[np.ndarray] = []
    std_pred_fields: list[np.ndarray] = []

    for t in range(args.forecast_steps + 1):
        pred_t = states_pred[t][:, args.var_idx]  # (E, nx, ny)
        true_t = states_true[t][:, args.var_idx]

        mean_pred_fields.append(pred_t.mean(dim=0).detach().float().cpu().numpy())
        mean_true_fields.append(true_t.mean(dim=0).detach().float().cpu().numpy())
        std_pred_fields.append(pred_t.std(dim=0).detach().float().cpu().numpy())

    if args.point_lat_idx is None:
        point_lat_idx = args.nx // 2
    else:
        point_lat_idx = args.point_lat_idx

    if args.point_lon_idx is None:
        point_lon_idx = args.ny // 2
    else:
        point_lon_idx = args.point_lon_idx

    _make_frames(
        mean_pred=mean_pred_fields,
        mean_true=mean_true_fields,
        std_pred=std_pred_fields,
        point_lat_idx=point_lat_idx,
        point_lon_idx=point_lon_idx,
        out_gif=args.out_gif,
        var_idx=args.var_idx,
        lat_min=-90.0,
        lat_max=90.0,
        lon_min=-180.0,
        lon_max=180.0,
        duration_s=args.gif_duration,
    )


if __name__ == "__main__":
    main()
