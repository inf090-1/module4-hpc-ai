import argparse
import io
import os
from datetime import datetime, timezone

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import torch

import cartopy.crs as ccrs
import cartopy.feature as cf

from geoarches.lightning_modules import load_module
from geoarches.dataloaders.era5 import Era5Forecast


def plot_frame(pred, gt, ts_str, cmap, vmin, vmax):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    fig.subplots_adjust(left=0.05, right=0.9, wspace=0.05)
    for i, (ax, field, title) in enumerate(
        zip(axes, [pred, gt], ["Prediction (Ensemble Mean)", "Ground Truth"])
    ):
        ax.set_global()
        ax.add_feature(cf.COASTLINE.with_scale("50m"), lw=0.5)
        ax.add_feature(cf.BORDERS.with_scale("50m"), lw=0.3)
        ax.add_feature(cf.LAND.with_scale("50m"), facecolor="lightgray", alpha=0.4)
        ax.gridlines(
            draw_labels=False,
            linewidth=0.5,
            color="black",
            alpha=0.5,
            linestyle="--",
        )

        im = ax.imshow(
            field,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            extent=[0, 360, -90, 90],
            origin="upper",
        )
        ax.set_title(title)

        if i == 0:
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            fig.colorbar(im, cax=cbar_ax, fraction=0.035, pad=0.03)

    fig.suptitle(ts_str, fontsize=14, y=0.90)
    return fig


def infer_gt_index_step(ds, lead_time_hours: int = 24) -> int:
    """Infer how many dataset indices correspond to +lead_time_hours.

    The original blog assumes ERA5 6-hourly cadence, so +24h == 4 indices.
    If we download a *subset* of hours (e.g., only hour=0 to keep data small),
    the dataset cadence changes; this function estimates the cadence from timestamps.
    """

    # ds[i]["timestamp"] is int32 seconds.
    t0 = int(ds[0]["timestamp"].item())
    t1 = int(ds[1]["timestamp"].item())
    delta_s = t1 - t0
    if delta_s <= 0:
        return 4

    lead_s = int(lead_time_hours * 3600)
    step = int(round(lead_s / float(delta_s)))
    return max(1, step)


def create_gif(args):
    os.makedirs(args.output_dir, exist_ok=True)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gen_model, _ = load_module(args.model)
    gen_model = gen_model.to(device)

    ds = Era5Forecast(
        path=args.data_path,
        load_prev=True,
        norm_scheme="pangu",
        domain="test",
    )

    # Batch initial conditions.
    batch = {k: v[None].to(device) for k, v in ds[0].items()}

    # Generate ensemble rollouts.
    rollouts = []
    for member in range(args.n_members):
        r = gen_model.sample_rollout(
            batch,
            batch_nb=0,
            member=member,
            iterations=args.rollout_iterations,
        ).cpu()
        r = ds.denormalize(r)
        rollouts.append(r["level"][0])

    ensemble_mean = torch.stack(rollouts, dim=0).mean(dim=0)

    # Determine colorbar limits from both prediction and ground truth.
    # Ground truth is denormalized directly from the dataset.
    lead_time_hours = 24
    idx_step = infer_gt_index_step(ds, lead_time_hours=lead_time_hours)
    print(f"[limited] Inferred gt index step for +{lead_time_hours}h: {idx_step}")

    gt_denormalized = []
    for step_i in range(args.rollout_iterations):
        idx = step_i * idx_step
        if idx >= len(ds):
            raise RuntimeError(
                f"Not enough ground-truth timesteps for rollout: idx={idx} >= len(ds)={len(ds)}"
            )

        gt_sample = ds[idx]
        gt_denorm = ds.denormalize(gt_sample["state"])
        gt_field = gt_denorm["level"][0, 7]
        ts_str = datetime.fromtimestamp(int(gt_sample["timestamp"]), timezone.utc).date().isoformat()
        gt_denormalized.append((gt_field, ts_str))

    model_min = float(ensemble_mean[:, 0, 7].min().item())
    model_max = float(ensemble_mean[:, 0, 7].max().item())

    gt_min = min(gt_field.min().item() for gt_field, _ in gt_denormalized)
    gt_max = max(gt_field.max().item() for gt_field, _ in gt_denormalized)

    colorbar_min = min(model_min, gt_min)
    colorbar_max = max(model_max, gt_max)

    # Create GIF frames.
    frames = []
    for day, (gt_field, ts_str) in enumerate(gt_denormalized):
        fig = plot_frame(
            ensemble_mean[day, 0, 7],
            gt_field,
            ts_str,
            args.cmap,
            colorbar_min,
            colorbar_max,
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        frames.append(imageio.imread(buf))
        plt.close(fig)

    out_path = os.path.join(args.output_dir, "Z500.gif")
    imageio.mimsave(out_path, frames, fps=1, loop=0)
    print(f"Saved GIF: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--rollout-iterations", type=int, default=10)
    parser.add_argument("--n-members", type=int, default=5)
    parser.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap name")
    args = parser.parse_args()

    create_gif(args)
