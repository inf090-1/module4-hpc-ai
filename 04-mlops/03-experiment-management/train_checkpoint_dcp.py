import os
import time

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler, random_split
from torchvision import datasets, transforms


class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class AppState(Stateful):
    """DCP wrapper to checkpoint both model + optimizer.

    This uses DCP's get_state_dict/set_state_dict helpers so the checkpointing
    logic remains robust to distributed wrappers.
    """

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer | None = None):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        model_state_dict, optim_state_dict = get_state_dict(self.model, self.optimizer)
        return {
            "model": model_state_dict,
            "optim": optim_state_dict,
        }

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )


def setup():
    # Mirrors the original script's SLURM/RANK env handling.
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))
        world_size = int(os.environ["SLURM_NTASKS"])
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    elif "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = None
    else:
        # Single-process fallback.
        return 0, 1, torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), 0

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if world_size is None:
        dist.init_process_group(backend=backend)
        world_size = dist.get_world_size()
    else:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return rank, world_size, device, local_rank


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def dcp_save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, checkpoint_id: str, epoch: int, loss: float):
    """Save a distributed checkpoint.

    Unlike the rank0-only variant, this call must be executed on *all* ranks.
    """

    model_state_for_dcp = model.module if hasattr(model, "module") else model
    state_dict = {
        "app": AppState(model_state_for_dcp, optimizer),
        "epoch": epoch,
        "loss": loss,
    }
    dcp.save(state_dict, checkpoint_id=checkpoint_id)


def dcp_load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    checkpoint_id: str,
    device: torch.device,
):
    """Load a distributed checkpoint.

    Returns (start_epoch, best_loss). All ranks must call this.
    """

    if not os.path.exists(checkpoint_id):
        return 0, float("inf")

    model_state_for_dcp = model.module if hasattr(model, "module") else model
    state_dict = {
        "app": AppState(model_state_for_dcp, optimizer),
        "epoch": 0,
        "loss": float("inf"),
    }

    # Ensure any CPU tensors created by dcp end up on the right device.
    # (DCP will reuse pre-allocated tensors when possible, so this is mostly
    # relevant for scalar tensors.)
    dcp.load(state_dict, checkpoint_id=checkpoint_id)

    return int(state_dict["epoch"]), float(state_dict["loss"])


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for data, target in loader:
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
        correct += output.argmax(1).eq(target).sum().item()
        total += data.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for data, target in loader:
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        output = model(data)
        correct += output.argmax(1).eq(target).sum().item()
        total += data.size(0)
    return correct / total


def main():
    rank, world_size, device, local_rank = setup()

    checkpoint_root = "./checkpoints_dcp"
    os.makedirs(checkpoint_root, exist_ok=True)
    if rank == 0:
        print(f"[dcp] Checkpoint root: {os.path.abspath(checkpoint_root)}")
        print(f"[dcp] Device: {device}")
        print(f"[dcp] World size: {world_size}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    full_train = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test = datasets.MNIST("./data", train=False, download=True, transform=transform)

    if dist.is_initialized():
        dist.barrier()

    train_size = 50000
    val_size = len(full_train) - train_size
    train, val = random_split(full_train, [train_size, val_size])

    train_sampler = DistributedSampler(train) if dist.is_initialized() else None
    train_loader = DataLoader(
        train,
        batch_size=256,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(val, batch_size=512, shuffle=False, num_workers=2)
    test_loader = DataLoader(test, batch_size=512, shuffle=False, num_workers=2)

    model = MNISTNet().to(device)
    if dist.is_initialized():
        if device.type == "cuda":
            model = DDP(model, device_ids=[local_rank])
        else:
            model = DDP(model)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    start_epoch, _ = dcp_load_checkpoint(
        model,
        optimizer,
        checkpoint_id=os.path.join(checkpoint_root, "latest"),
        device=device,
    )

    num_epochs = 5
    total_start = time.time()

    for epoch in range(start_epoch + 1, start_epoch + num_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model, val_loader, device)
        elapsed = time.time() - start

        if rank == 0:
            print(
                f"[dcp] Epoch {epoch}/{start_epoch + num_epochs} - "
                f"loss: {train_loss:.4f} - acc: {train_acc:.4f} - "
                f"val_acc: {val_acc:.4f} - time: {elapsed:.1f}s"
            )

        # All ranks participate in saving.
        epoch_checkpoint_id = os.path.join(checkpoint_root, f"epoch_{epoch}")
        latest_checkpoint_id = os.path.join(checkpoint_root, "latest")

        dcp_save_checkpoint(model, optimizer, epoch_checkpoint_id, epoch, train_loss)
        dcp_save_checkpoint(model, optimizer, latest_checkpoint_id, epoch, train_loss)

        if dist.is_initialized():
            dist.barrier()

    test_acc = evaluate(model, test_loader, device)
    total_elapsed = time.time() - total_start

    if rank == 0:
        print(f"[dcp] Final test accuracy: {test_acc * 100:.2f}%")
        print(f"[dcp] Total training time: {total_elapsed:.1f}s")
        print(f"[dcp] Checkpoints saved in: {os.path.abspath(checkpoint_root)}")

    cleanup()


if __name__ == "__main__":
    main()
