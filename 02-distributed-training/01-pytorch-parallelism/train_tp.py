import argparse
import os
import socket
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from torch.utils.data import DataLoader, DistributedSampler
from model import SimpleLLM
from dataset import ShakespeareDataset

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
        os.environ.setdefault("MASTER_PORT", "29501")
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    else:
        world_size = torch.cuda.device_count()
        if world_size < 2:
            print("Need at least 2 GPUs for TP.")
            return None, None
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(find_free_port()))
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=local_rank)

    torch.cuda.set_device(local_rank)
    return rank, world_size

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, max_batches=None):
    dataset = ShakespeareDataset(seq_len=64, train=True)
    vocab_size = dataset.vocab_size
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler)

    model = SimpleLLM(vocab_size=vocab_size).to(rank)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total model parameters: {total_params:,}")

    tp_applied = False
    try:
        device_mesh = init_device_mesh("cuda", (world_size,))
        parallelize_plan = {
            "embedding": RowwiseParallel(input_layouts=None),
            "linear": ColwiseParallel(output_layouts=None),
        }
        model = parallelize_module(model, device_mesh, parallelize_plan)
        tp_applied = True
        if rank == 0:
            print(f"Tensor Parallelism applied: embedding + linear sharded across {world_size} GPUs")
    except Exception as e:
        if rank == 0:
            print(f"TP parallelization note: {e}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(2):
        sampler.set_epoch(epoch)
        for batch_idx, (inputs, labels) in enumerate(loader):
            if max_batches and batch_idx >= max_batches:
                break
            inputs, labels = inputs.to(rank), labels.to(rank)

            optimizer.zero_grad()
            outputs = model(inputs)

            if tp_applied:
                flat = outputs.reshape(-1, outputs.shape[-1])
                max_vocab = (vocab_size + world_size - 1) // world_size
                padded = torch.zeros(flat.shape[0], max_vocab, device=flat.device)
                padded[:, :flat.shape[-1]] = flat
                gathered = [torch.zeros_like(padded) for _ in range(world_size)]
                dist.all_gather(gathered, padded)
                full_output = torch.cat(gathered, dim=-1)[:, :vocab_size]
                loss = criterion(full_output, labels.view(-1))
            else:
                loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))

            loss.backward()
            optimizer.step()

            if rank == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_batches", type=int, default=None)
    args = parser.parse_args()

    rank, world_size = setup()
    if rank is not None:
        train(rank, world_size, args.max_batches)
