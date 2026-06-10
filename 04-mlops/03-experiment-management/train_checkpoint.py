import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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
        x = self.features(x)
        x = self.classifier(x)
        return x


def setup():
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))
        world_size = int(os.environ["SLURM_NTASKS"])
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    elif "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
    else:
        return 0, 1, torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), 1

    torch.cuda.set_device(local_rank)
    return rank, world_size, torch.device(f"cuda:{local_rank}"), local_rank


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def save_checkpoint(model, optimizer, epoch, loss, path, rank):
    if rank != 0:
        return

    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save({
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }, path)
    print(f"[ckpt] Saved checkpoint to {path}")


def load_checkpoint(model, optimizer, path, device):
    if not os.path.exists(path):
        print(f"[ckpt] No checkpoint found at {path}")
        return 0, float("inf")

    checkpoint = torch.load(path, map_location=device)
    model_to_load = model.module if hasattr(model, "module") else model
    model_to_load.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"[ckpt] Loaded checkpoint from {path} (epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.4f})")
    return checkpoint["epoch"], checkpoint["loss"]


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

    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "model_epoch_{}.pt")

    if rank == 0:
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        print(f"[ckpt] Device: {device} ({device_name})")
        print(f"[ckpt] World size: {world_size}")
        print(f"[ckpt] Checkpoint dir: {os.path.abspath(checkpoint_dir)}")

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
        train, batch_size=256, sampler=train_sampler,
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(val, batch_size=512, shuffle=False, num_workers=2)
    test_loader = DataLoader(test, batch_size=512, shuffle=False, num_workers=2)

    model = MNISTNet().to(device)
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank])

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path.format("latest"), device)

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
                f"[ckpt] Epoch {epoch}/{start_epoch + num_epochs} - "
                f"loss: {train_loss:.4f} - acc: {train_acc:.4f} - "
                f"val_acc: {val_acc:.4f} - time: {elapsed:.1f}s"
            )

        save_checkpoint(model, optimizer, epoch, train_loss, checkpoint_path.format(epoch), rank)
        save_checkpoint(model, optimizer, epoch, train_loss, checkpoint_path.format("latest"), rank)

    test_acc = evaluate(model, test_loader, device)
    total_elapsed = time.time() - total_start

    if rank == 0:
        print(f"[ckpt] Final test accuracy: {test_acc * 100:.2f}%")
        print(f"[ckpt] Total training time: {total_elapsed:.1f}s")
        print(f"[ckpt] Checkpoints saved in: {os.path.abspath(checkpoint_dir)}")

    cleanup()


if __name__ == "__main__":
    main()
