import argparse
import os
import sys
import time
from typing import List

import numpy as np
import torch


# Make it possible to run this script directly from this folder.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COURSE_ROOT = os.path.dirname(SCRIPT_DIR)  # 03-inference/
sys.path.insert(0, COURSE_ROOT)

from weather_model import WeatherNet, make_synthetic_batch  # noqa: E402


def _device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _maybe_setup_amp(use_amp: bool, device: torch.device, amp_dtype: torch.dtype):
    if use_amp and device.type == "cuda":
        return True, amp_dtype
    return False, amp_dtype


@torch.no_grad()
def _rollout_steps(model: torch.nn.Module, x: torch.Tensor, steps: int, *, use_amp: bool, amp_dtype: torch.dtype):
    """Run an autoregressive rollout x_{t+1} = model(x_t) for `steps` steps."""
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else _NullContext()
    )

    for _ in range(steps):
        with autocast_ctx:
            x = model(x)
    return x


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WeatherNet latency/throughput sweep vs batch size (MI300X friendly)"
    )

    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--vars", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=64)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-start", type=int, default=0)

    parser.add_argument("--forecast-steps", type=int, default=8)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=8)

    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2,4,8,16,32",
        help="Comma-separated list of batch sizes to benchmark.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="../01-batch/weathernet_infer_demo.pt",
        help="Optional checkpoint. If missing, uses random weights.",
    )

    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16"])

    parser.add_argument("--compile-model", action="store_true")

    parser.add_argument("--out-dir", type=str, default="./latency_sweep_out")
    parser.add_argument("--out-plot", type=str, default="latency_batch_sweep.png")
    parser.add_argument("--out-csv", type=str, default="latency_batch_sweep.csv")

    args = parser.parse_args()

    device = _device()
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    use_amp, amp_dtype = _maybe_setup_amp(args.use_amp, device, amp_dtype)

    batch_sizes: List[int] = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]
    batch_sizes = sorted(set(batch_sizes))

    print(f"[sweep] Device: {device} | use_amp={use_amp} ({amp_dtype})")
    print(f"[sweep] batch_sizes={batch_sizes}")

    model = WeatherNet(nvars=args.vars, hidden=args.hidden).to(device)
    if args.checkpoint and os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        print(f"[sweep] Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"[sweep] Checkpoint not found: {args.checkpoint} (using random weights)")
    model.eval()

    if args.compile_model and hasattr(torch, "compile") and device.type == "cuda":
        try:
            model = torch.compile(model)
            print("[sweep] torch.compile enabled")
        except Exception as e:  # pragma: no cover
            print(f"[sweep] torch.compile failed; continuing. Error: {e}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, args.out_csv)
    out_plot = os.path.join(args.out_dir, args.out_plot)

    rows = []

    # Warmup + measure.
    for bs in batch_sizes:
        x0 = make_synthetic_batch(
            seed=args.seed,
            sample_start=args.sample_start,
            batch_size=bs,
            nx=args.nx,
            ny=args.ny,
            nvars=args.vars,
            device=device,
        )

        # Warmup
        for _ in range(args.warmup_iters):
            x_w = x0
            if device.type == "cuda":
                torch.cuda.synchronize()
            _rollout_steps(
                model,
                x_w,
                args.forecast_steps,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()

        # Timed repeats
        times = []
        for _ in range(args.repeats):
            x_w = x0
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            _rollout_steps(
                model,
                x_w,
                args.forecast_steps,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            t = time.time() - t0
            times.append(t)

        total_time = float(np.mean(times))
        std_time = float(np.std(times))

        # Convert to per-sample per-step latency.
        # total_time covers bs * forecast_steps forward passes.
        ms_per_sample_per_step = (total_time * 1000.0) / (bs * args.forecast_steps)
        samples_per_sec = (bs / (total_time / args.forecast_steps))

        print(
            f"[sweep] bs={bs:<3d} avg={total_time:.3f}s std={std_time:.3f}s | "
            f"lat={ms_per_sample_per_step:.3f} ms/sample/step | thr~{samples_per_sec:.0f} sample(s)/s/step"
        )

        rows.append(
            {
                "batch_size": bs,
                "avg_rollout_time_s": total_time,
                "std_rollout_time_s": std_time,
                "ms_per_sample_per_step": ms_per_sample_per_step,
                "throughput_samples_per_s_per_step": samples_per_sec,
            }
        )

    # Save CSV
    import csv

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Plot
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(
            "Missing visualization deps for plotting. Install with: pip install matplotlib"
            f"\nImport error: {e}"
        )

    b = np.array([r["batch_size"] for r in rows], dtype=np.float64)
    ms = np.array([r["ms_per_sample_per_step"] for r in rows], dtype=np.float64)
    thr = np.array(
        [r["throughput_samples_per_s_per_step"] for r in rows], dtype=np.float64
    )

    fig, ax1 = plt.subplots(figsize=(10.5, 5.2), dpi=140)
    ax1.plot(b, ms, marker="o", color="tab:red", lw=2)
    ax1.set_xlabel("Batch size")
    ax1.set_ylabel("Latency (ms / sample / step)", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(b, thr, marker="s", color="tab:blue", lw=2)
    ax2.set_ylabel("Throughput (samples/s/step)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    fig.suptitle(
        f"WeatherNet latency/throughput sweep (forecast_steps={args.forecast_steps}, use_amp={use_amp})"
    )

    plt.tight_layout()
    fig.savefig(out_plot)
    plt.close(fig)

    print(f"[sweep] Wrote CSV: {out_csv}")
    print(f"[sweep] Wrote plot: {out_plot}")


if __name__ == "__main__":
    main()
