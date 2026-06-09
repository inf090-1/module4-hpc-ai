import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler


# Make it possible to run this script directly from this folder.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COURSE_ROOT = os.path.dirname(SCRIPT_DIR)  # 03-inference/
sys.path.insert(0, COURSE_ROOT)

from weather_model import (  # noqa: E402
    WeatherNet,
    make_synthetic_batch,
    rollout_physics,
    weather_physics_step,
)


def _device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_quick_demo(
    model: torch.nn.Module,
    *,
    seed: int,
    nx: int,
    ny: int,
    nvars: int,
    alpha: float,
    steps: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> None:
    """Very small training loop to get a usable checkpoint for profiling."""

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    # Reuse grids for synthetic generation + physics target.
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
        loss = F.mse_loss(pred, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (step + 1) % max(1, steps // 5) == 0:
            print(f"[train-demo] step {step+1:>4d}/{steps} loss={loss.item():.6f}")

    model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile WeatherNet inference rollout with PyTorch Profiler"
    )

    # Model/data.
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--vars", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.1)

    # Checkpoint.
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="../01-batch/weathernet_infer_demo.pt",
        help="State_dict checkpoint to load (produced by lesson 1).",
    )
    parser.add_argument(
        "--auto-train",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If checkpoint is missing, auto-train a small demo model for profiling.",
    )
    parser.add_argument("--auto-train-steps", type=int, default=80)
    parser.add_argument("--auto-train-batch-size", type=int, default=4)
    parser.add_argument("--auto-train-lr", type=float, default=3e-3)

    # Inference rollout.
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=24)

    # Profiling.
    parser.add_argument("--use-amp", action="store_true", help="Profile with AMP autocast")
    parser.add_argument(
        "--amp-dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./traces",
        help="Where to write Chrome trace files",
    )

    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Try torch.compile(model) for faster inference (if supported).",
    )

    args = parser.parse_args()

    device = _device()
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    print(f"[profile] Device: {device}")

    model = WeatherNet(nvars=args.vars, hidden=args.hidden).to(device)

    # Load or (optionally) demo-train.
    if args.checkpoint and os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        print(f"[profile] Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"[profile] Checkpoint not found: {args.checkpoint}")
        if args.auto_train:
            print("[profile] Auto-training a small demo model ...")
            train_quick_demo(
                model,
                seed=args.seed,
                nx=args.nx,
                ny=args.ny,
                nvars=args.vars,
                alpha=args.alpha,
                steps=args.auto_train_steps,
                batch_size=args.auto_train_batch_size,
                lr=args.auto_train_lr,
                device=device,
            )
        else:
            print("[profile] Proceeding with random weights (trace will still show kernels).")

    model.eval()

    if args.compile_model and hasattr(torch, "compile") and device.type == "cuda":
        try:
            model = torch.compile(model)
            print("[profile] torch.compile enabled")
        except Exception as e:  # pragma: no cover
            print(f"[profile] torch.compile failed; continuing without it. Error: {e}")

    # Input batch.
    i = torch.arange(args.nx, device=device, dtype=torch.float32).view(args.nx, 1)
    j = torch.arange(args.ny, device=device, dtype=torch.float32).view(1, args.ny)

    x = make_synthetic_batch(
        seed=args.seed,
        sample_start=args.sample_start,
        batch_size=args.batch_size,
        nx=args.nx,
        ny=args.ny,
        nvars=args.vars,
        device=device,
        i=i,
        j=j,
    )

    use_amp = bool(args.use_amp and device.type == "cuda")

    # Warmup: run a few steps so kernels/JIT settle before tracing.
    with torch.inference_mode():
        x_warm = x
        for _ in range(2):
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    x_warm = model(x_warm)
            else:
                x_warm = model(x_warm)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    os.makedirs(args.out_dir, exist_ok=True)
    trace_dir = args.out_dir

    print(
        f"[profile] Profiling inference rollout: batch={args.batch_size} steps={args.steps} use_amp={use_amp}"
    )

    with profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=args.steps, repeat=1),
        on_trace_ready=tensorboard_trace_handler(trace_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with torch.inference_mode():
            x_prof = x
            for _ in range(args.steps):
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        x_prof = model(x_prof)
                else:
                    x_prof = model(x_prof)
                prof.step()

    # Print a quick summary.
    if device.type == "cuda":
        print("\n[profile] Top 10 CUDA ops by self time:")
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

    print("\n[profile] Top 10 CPU ops by self time:")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

    print(f"\n[profile] Traces written to: {trace_dir}/")
    print("[profile] Open the Chrome trace in Chrome via: chrome://tracing")


if __name__ == "__main__":
    main()
