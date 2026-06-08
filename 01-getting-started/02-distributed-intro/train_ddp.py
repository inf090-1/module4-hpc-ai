import argparse
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
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))
        world_size = int(os.environ["SLURM_NTASKS"])
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    else:
        print("[ddp] Running in single-GPU mode")
        return 0, 1, torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, dist.get_world_size(), device


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def train_epoch(model, loader, optimizer, criterion, device, rank):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
        correct += output.argmax(1).eq(target).sum().item()
        total += data.size(0)

    avg_loss = total_loss / total
    acc = correct / total

    if dist.is_initialized():
        tensors = torch.tensor([avg_loss, acc], device=device)
        dist.all_reduce(tensors, op=dist.ReduceOp.AVG)
        avg_loss, acc = tensors.tolist()

    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        correct += output.argmax(1).eq(target).sum().item()
        total += data.size(0)
    return correct / total


def main():
    parser = argparse.ArgumentParser(description="PyTorch MNIST DDP Training")
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=512, metavar='N',
                        help='input batch size for testing (default: 512)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    args = parser.parse_args()

    rank, world_size, device = setup()

    if rank == 0:
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        print(f"[ddp] World size: {world_size}, Device: {device} ({device_name})")

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
    val_sampler = DistributedSampler(val, shuffle=False) if dist.is_initialized() else None

    train_loader = DataLoader(
        train, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val, batch_size=args.test_batch_size, sampler=val_sampler,
        num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(test, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

    model = MNISTNet().to(device)
    if dist.is_initialized():
        model = DDP(model, device_ids=[device])

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    num_epochs = args.epochs
    total_start = time.time()

    for epoch in range(1, num_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, rank)
        val_acc = evaluate(model, val_loader, device)
        elapsed = time.time() - start

        if rank == 0:
            print(
                f"[ddp] Epoch {epoch}/{num_epochs} - "
                f"train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - "
                f"val_acc: {val_acc:.4f}"
            )

    test_acc = evaluate(model, test_loader, device)

    if rank == 0:
        print(f"[ddp] Final test accuracy: {test_acc * 100:.2f}%")

    cleanup()


if __name__ == "__main__":
    main()
