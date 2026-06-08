import argparse
import math
import os
import urllib.request

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import lightning as L

class ShakespeareDataset(Dataset):
    def __init__(self, seq_len=64, train=True):
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
        return len(self.data) - self.seq_len
        
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
        src = self.embedding(src)
        src = self.pos_encoder(src)
        seq_len = src.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(src.device)
        output = self.transformer_encoder(src, mask=mask, is_causal=True)
        output = self.linear(output)
        return output

class LLMLightning(L.LightningModule):
    def __init__(self, vocab_size=65, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = SimpleLLM(vocab_size=vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()
        self.vocab_size = vocab_size

    def prepare_data(self):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        file_path = "shakespeare.txt"
        if not os.path.exists(file_path):
            print(f"Downloading {file_path}...")
            urllib.request.urlretrieve(url, file_path)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits.view(-1, self.vocab_size), y.view(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits.view(-1, self.vocab_size), y.view(-1))
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=int, default=2)
    parser.add_argument("--strategy", type=str, default="ddp")
    parser.add_argument("--max_epochs", type=int, default=2)
    args = parser.parse_args()

    strategy = args.strategy
    if args.devices == 1:
        strategy = "auto"

    print(f"[lightning] Training with {args.devices} GPUs ({strategy})")

    model = LLMLightning()
    model.prepare_data()

    train_data = ShakespeareDataset(train=True)
    val_data = ShakespeareDataset(train=False)

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=128, shuffle=False, num_workers=2)

    trainer = L.Trainer(
        accelerator="gpu",
        devices=args.devices,
        num_nodes=1,
        strategy=strategy,
        max_epochs=args.max_epochs,
        logger=L.CSVLogger("logs", name="llm_lightning"),
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
