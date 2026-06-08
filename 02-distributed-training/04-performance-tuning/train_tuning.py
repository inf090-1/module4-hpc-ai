import argparse
import time
import math
import os
import urllib.request
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class ShakespeareDataset(Dataset):
    def __init__(self, seq_len=128, train=True):
        file_path = "shakespeare.txt"
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
        n = int(0.9 * len(data))
        self.data = data[:n] if train else data[n:]
        self.seq_len = seq_len
        
    def __len__(self):
        # For quick profiling, limit dataset size
        return min(len(self.data) - self.seq_len, 2000)
        
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+1:idx+self.seq_len+1]
        return x, y

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return x

class SimpleLLM(nn.Module):
    def __init__(self, vocab_size=65, d_model=512, nhead=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        torch.cuda.nvtx.range_push("Embedding")
        src = self.embedding(src)
        src = self.pos_encoder(src)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("Transformer")
        seq_len = src.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(src.device)
        output = self.transformer_encoder(src, mask=mask, is_causal=True)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("Linear")
        output = self.linear(output)
        torch.cuda.nvtx.range_pop()
        
        return output

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        torch.cuda.nvtx.range_push(f"Batch_{batch_idx}")
        
        torch.cuda.nvtx.range_push("DataTransfer")
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        torch.cuda.nvtx.range_pop()
        
        optimizer.zero_grad(set_to_none=True)
        
        torch.cuda.nvtx.range_push("ForwardPass")
        output = model(data)
        loss = criterion(output.view(-1, 65), target.view(-1))
        torch.cuda.nvtx.range_pop()
        
        torch.cuda.nvtx.range_push("BackwardPass")
        loss.backward()
        torch.cuda.nvtx.range_pop()
        
        torch.cuda.nvtx.range_push("OptimizerStep")
        optimizer.step()
        torch.cuda.nvtx.range_pop()
        
        total_loss += loss.item() * data.size(0)
        total += data.size(0)
        
        torch.cuda.nvtx.range_pop() # End Batch
        
    return total_loss / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    file_path = "shakespeare.txt"
    if not os.path.exists(file_path):
        print(f"Downloading {file_path}...")
        urllib.request.urlretrieve(url, file_path)

    print(f"[tuning] Device: {device}")
    print(f"[tuning] Batch size: {args.batch_size}")
    print(f"[tuning] Num workers: {args.num_workers}")

    train_data = ShakespeareDataset(train=True)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, 
        num_workers=args.num_workers, pin_memory=True
    )

    model = SimpleLLM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("[tuning] Warming up...")
    model.train()
    dummy_x = torch.randint(0, 65, (args.batch_size, 128)).to(device)
    dummy_y = torch.randint(0, 65, (args.batch_size, 128)).to(device)
    out = model(dummy_x)
    loss = criterion(out.view(-1, 65), dummy_y.view(-1))
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    print("[tuning] Warmup complete. Starting training...")

    total_start = time.time()

    for epoch in range(1, args.num_epochs + 1):
        torch.cuda.nvtx.range_push(f"Epoch_{epoch}")
        start = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        torch.cuda.synchronize()
        epoch_time = time.time() - start

        samples_per_sec = len(train_data) / epoch_time

        print(
            f"[tuning] Epoch {epoch}/{args.num_epochs} - "
            f"loss: {train_loss:.4f} - "
            f"time: {epoch_time:.1f}s - samples/s: {samples_per_sec:.0f}"
        )
        torch.cuda.nvtx.range_pop()

    total_elapsed = time.time() - total_start
    print(f"[tuning] Total time: {total_elapsed:.1f}s")

if __name__ == "__main__":
    main()
