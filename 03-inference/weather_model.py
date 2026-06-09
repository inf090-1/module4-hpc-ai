import torch
import torch.nn as nn
import torch.nn.functional as F


class WeatherNet(nn.Module):
    """Tiny convolutional weather-like model.

    Input/Output shape contract:
    - input:  (B, C, H, W)
    - output: (B, C, H, W)

    This mirrors the classroom WeatherNet used in the training part of the course,
    but is self-contained so the inference lessons can run independently.
    """

    def __init__(self, nvars: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nvars, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, nvars, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_synthetic_x(
    seed: int,
    sample_idx: int,
    nx: int,
    ny: int,
    nvars: int,
    device: torch.device,
    i: torch.Tensor | None = None,
    j: torch.Tensor | None = None,
) -> torch.Tensor:
    """Create an initial gridded weather state x.

    Returns:
      x: (C, nx, ny)
    """
    if i is None or j is None:
        i = torch.arange(nx, device=device, dtype=torch.float32).view(nx, 1)
        j = torch.arange(ny, device=device, dtype=torch.float32).view(1, ny)

    x = torch.zeros((nvars, nx, ny), device=device, dtype=torch.float32)
    for v in range(nvars):
        phase = 0.01 * float(seed + sample_idx) + 0.05 * float(v)
        x[v] = torch.sin(0.12 * i + 0.09 * j + phase) + 0.5 * torch.cos(
            0.07 * i - 0.11 * j + 0.7 * phase
        )
    return x


def make_synthetic_batch(
    seed: int,
    sample_start: int,
    batch_size: int,
    nx: int,
    ny: int,
    nvars: int,
    device: torch.device,
    i: torch.Tensor | None = None,
    j: torch.Tensor | None = None,
) -> torch.Tensor:
    """Create a batch of initial states.

    Returns:
      x0: (B, C, nx, ny)
    """
    if i is None or j is None:
        i = torch.arange(nx, device=device, dtype=torch.float32).view(nx, 1)
        j = torch.arange(ny, device=device, dtype=torch.float32).view(1, ny)

    xs = []
    for b in range(batch_size):
        xs.append(
            make_synthetic_x(
                seed=seed,
                sample_idx=sample_start + b,
                nx=nx,
                ny=ny,
                nvars=nvars,
                device=device,
                i=i,
                j=j,
            )
        )
    return torch.stack(xs, dim=0)


def weather_physics_step(
    x: torch.Tensor,
    seed: int,
    nx: int,
    ny: int,
    alpha: float = 0.1,
    i: torch.Tensor | None = None,
    j: torch.Tensor | None = None,
) -> torch.Tensor:
    """One-step deterministic "ground-truth" dynamics used for the synthetic dataset.

    Args:
      x: (B, C, nx, ny)

    Returns:
      y: (B, C, nx, ny)
    """
    if i is None or j is None:
        device = x.device
        i = torch.arange(nx, device=device, dtype=torch.float32).view(nx, 1)
        j = torch.arange(ny, device=device, dtype=torch.float32).view(1, ny)

    # Reflective padding and 5-point Laplacian.
    x_pad = F.pad(x, pad=(1, 1, 1, 1), mode="reflect")

    up = x_pad[:, :, 0:nx, 1 : ny + 1]
    down = x_pad[:, :, 2 : nx + 2, 1 : ny + 1]
    left = x_pad[:, :, 1 : nx + 1, 0:ny]
    right = x_pad[:, :, 1 : nx + 1, 2 : ny + 2]

    center = x
    lap = up + down + left + right - 4.0 * center
    y = center + alpha * lap

    structured = 0.01 * torch.sin(0.03 * i + 0.02 * j + 0.2 * float(seed))
    y = y + structured  # broadcast over (B, C)
    return y


@torch.no_grad()
def rollout_model(
    model: nn.Module,
    x0: torch.Tensor,
    steps: int,
    *,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> list[torch.Tensor]:
    """Autoregressive rollout: x_{t+1} = model(x_t)."""
    states = [x0]
    x = x0

    # torch.autocast works for ROCm as well (device_type="cuda").
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
    )

    for _ in range(steps):
        with autocast_ctx:
            x = model(x)
        states.append(x)

    return states


@torch.no_grad()
def rollout_physics(
    seed: int,
    x0: torch.Tensor,
    steps: int,
    nx: int,
    ny: int,
    *,
    alpha: float = 0.1,
) -> list[torch.Tensor]:
    """Autoregressive rollout using the synthetic physics operator."""
    # Precompute grids once.
    device = x0.device
    i = torch.arange(nx, device=device, dtype=torch.float32).view(nx, 1)
    j = torch.arange(ny, device=device, dtype=torch.float32).view(1, ny)

    states = [x0]
    x = x0
    for _ in range(steps):
        x = weather_physics_step(x, seed=seed, nx=nx, ny=ny, alpha=alpha, i=i, j=j)
        states.append(x)
    return states


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


nullcontext = _NullContext
