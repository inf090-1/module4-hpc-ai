import argparse
import os
import socket
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from dataset import ShakespeareDataset
from model import TineLLM

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def setup():
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))
        world_size = int(os.environ["SLURM_NTASKS"])

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")

        init_method = os.environ.get("TORCH_DISTRIBUTED_INIT_METHOD")

        kwargs = dict(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            device_id=local_rank,
        )
        if init_method:
            kwargs["init_method"] = init_method

        dist.init_process_group(**kwargs)

    else:
        world_size = torch.cuda.device_count()
        if world_size < 2:
            print("Need at least 2 GPUs for DDP.")
            return None, None, None
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(find_free_port()))
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            device_id=local_rank,
        )

    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def train(
    rank,
    local_rank,
    world_size,
    *,
    epochs=2,
    max_batches=None,
    seq_len=64,
    batch_size=64,
    lr=1e-3,
    num_workers=2,
    bf16=None,
):
    device = torch.device(f"cuda:{local_rank}")

    dataset = ShakespeareDataset(seq_len=seq_len, train=True)
    vocab_size = dataset.vocab_size

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    if bf16 is None:
        bf16 = torch.cuda.is_bf16_supported()

    model = TineLLM(vocab_size=vocab_size, max_seq_len=seq_len).to(device)
    ddp_model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=lr)

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        ddp_model.train()

        epoch_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        epoch_correct = torch.zeros((), device=device, dtype=torch.float32)
        epoch_tokens = torch.zeros((), device=device, dtype=torch.float32)

        start = time.time()

        for batch_idx, (inputs, labels) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
                outputs = ddp_model(inputs)  # [B, S, V]
                loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))

                pred = outputs.argmax(dim=-1)
                correct = (pred == labels).sum()
                tokens = labels.numel()

            loss.backward()
            optimizer.step()

            epoch_loss_sum += loss.detach() * tokens
            epoch_correct += correct.detach().to(torch.float32)
            epoch_tokens += tokens

        torch.cuda.synchronize(device)

        stats = torch.stack([epoch_loss_sum, epoch_correct, epoch_tokens])
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        epoch_time = time.time() - start

        epoch_loss = (stats[0] / stats[2]).item()
        epoch_acc = (stats[1] / stats[2]).item()

        if rank == 0:
            print(
                f"Epoch {epoch+1}/{epochs} | loss: {epoch_loss:.4f} | acc: {epoch_acc*100:.2f}% | time: {epoch_time:.1f}s"
            )

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 autocast")
    parser.add_argument("--no-bf16", action="store_true", help="Disable bf16 autocast")
    parser.add_argument("--max_batches", type=int, default=None)
    args = parser.parse_args()

    rank, world_size, local_rank = setup()
    if rank is not None:
        bf16 = None
        if args.bf16:
            bf16 = True
        elif args.no_bf16:
            bf16 = False

        train(
            rank,
            local_rank,
            world_size,
            epochs=args.epochs,
            max_batches=args.max_batches,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            lr=args.lr,
            num_workers=args.num_workers,
            bf16=bf16,
        )
