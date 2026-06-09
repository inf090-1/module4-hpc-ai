import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import ShakespeareDataset
from model import PositionalEncoding


class PipelineParallelTineLLM(nn.Module):
    def __init__(
        self,
        vocab_size=65,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        max_seq_len=64,
    ):
        super().__init__()
        self.dev0 = torch.device("cuda:0")
        self.dev1 = torch.device("cuda:1")

        # Part 1 on GPU 0
        self.embedding = nn.Embedding(vocab_size, d_model).to(self.dev0)
        self.pos_encoder = PositionalEncoding(d_model).to(self.dev0)
        encoder_layer1 = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer_part1 = nn.TransformerEncoder(
            encoder_layer1, num_layers=num_layers // 2
        ).to(self.dev0)

        # Part 2 on GPU 1
        encoder_layer2 = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer_part2 = nn.TransformerEncoder(
            encoder_layer2, num_layers=num_layers // 2
        ).to(self.dev1)
        self.linear = nn.Linear(d_model, vocab_size, bias=False).to(self.dev1)

        # Cache causal masks (seq_len is fixed in the examples)
        mask = nn.Transformer.generate_square_subsequent_mask(max_seq_len)
        self.register_buffer("causal_mask_dev0", mask.to(self.dev0), persistent=False)
        self.register_buffer("causal_mask_dev1", mask.to(self.dev1), persistent=False)

    def forward(self, src_dev0):
        # src_dev0: [B, S] on cuda:0
        seq_len = src_dev0.size(1)
        mask_dev0 = self.causal_mask_dev0[:seq_len, :seq_len]

        x = self.embedding(src_dev0)
        x = self.pos_encoder(x)
        x = self.transformer_part1(x, mask=mask_dev0, is_causal=True)

        # Activation transfer GPU0 -> GPU1
        x = x.to(self.dev1)
        mask_dev1 = self.causal_mask_dev1[:seq_len, :seq_len]

        x = self.transformer_part2(x, mask=mask_dev1, is_causal=True)
        return self.linear(x)  # [B, S, vocab]


def train(
    *,
    epochs=2,
    max_batches=None,
    seq_len=64,
    batch_size=64,
    lr=1e-3,
    num_workers=2,
    bf16=None,
):
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Need at least 2 GPUs for Pipeline Parallelism.")

    dev0 = torch.device("cuda:0")
    dev1 = torch.device("cuda:1")

    if bf16 is None:
        bf16 = torch.cuda.is_bf16_supported()

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

    model = PipelineParallelTineLLM(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()

        epoch_loss_sum = torch.zeros((), device=dev1, dtype=torch.float32)
        epoch_correct = torch.zeros((), device=dev1, dtype=torch.float32)
        epoch_tokens = torch.zeros((), device=dev1, dtype=torch.float32)

        start = time.time()

        for batch_idx, (inputs, labels) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            inputs = inputs.to(dev0, non_blocking=True)
            labels = labels.to(dev1, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
                outputs = model(inputs)  # on dev1
                outputs_fp32 = outputs.float()
                loss = criterion(outputs_fp32.view(-1, vocab_size), labels.view(-1))

                pred = outputs_fp32.argmax(dim=-1)
                correct = (pred == labels).sum()
                tokens = labels.numel()

            loss.backward()
            optimizer.step()

            epoch_loss_sum += loss.detach() * tokens
            epoch_correct += correct.detach().to(torch.float32)
            epoch_tokens += tokens

        # Ensure both GPUs finish before timing
        torch.cuda.synchronize(dev0)
        torch.cuda.synchronize(dev1)
        epoch_time = time.time() - start

        epoch_loss = (epoch_loss_sum / epoch_tokens).item()
        epoch_acc = (epoch_correct / epoch_tokens).item()

        print(
            f"Epoch {epoch+1}/{epochs} | loss: {epoch_loss:.4f} | acc: {epoch_acc*100:.2f}% | time: {epoch_time:.1f}s"
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
