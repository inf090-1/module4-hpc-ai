import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import ShakespeareDataset
from model import TineLLM


def train(
    *,
    epochs: int = 2,
    max_batches: int | None = None,
    seq_len: int = 64,
    batch_size: int = 64,
    lr: float = 1e-3,
    num_workers: int = 2,
    bf16: bool | None = None,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this lesson's examples.")

    device = torch.device("cuda")

    dataset = ShakespeareDataset(seq_len=seq_len, train=True)
    vocab_size = dataset.vocab_size

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    if bf16 is None:
        bf16 = torch.cuda.is_bf16_supported()

    model = TineLLM(vocab_size=vocab_size, max_seq_len=seq_len).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()

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
                outputs = model(inputs)  # [B, S, V]
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
        epoch_time = time.time() - start

        epoch_loss = (epoch_loss_sum / epoch_tokens).item()
        epoch_acc = (epoch_correct / epoch_tokens).item()

        print(
            f"[single] Epoch {epoch+1}/{epochs} | loss: {epoch_loss:.4f} | acc: {epoch_acc*100:.2f}% | time: {epoch_time:.1f}s"
        )


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

    bf16 = None
    if args.bf16:
        bf16 = True
    elif args.no_bf16:
        bf16 = False

    train(
        epochs=args.epochs,
        max_batches=args.max_batches,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        bf16=bf16,
    )
