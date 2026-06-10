import argparse
import os
import urllib.request
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import lightning as L
from lightning.pytorch.loggers import CSVLogger

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
        return len(self.data) - self.seq_len
        
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+1:idx+self.seq_len+1]
        return x, y

class MassiveModel(nn.Module):
    def __init__(
        self,
        d_model=4096,
        num_layers=12,
        vocab_size=65,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=16,
            dim_feedforward=d_model * ff_mult,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        seq_len = x.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        x = self.transformer(x, mask=mask, is_causal=True)
        return self.head(x)

class MassiveLitModel(L.LightningModule):
    def __init__(
        self,
        d_model=4096,
        num_layers=12,
        vocab_size=65,
        lr=1e-4,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = MassiveModel(
            d_model=d_model,
            num_layers=num_layers,
            vocab_size=vocab_size,
            ff_mult=ff_mult,
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.vocab_size = vocab_size

    def prepare_data(self):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        file_path = "shakespeare.txt"
        if not os.path.exists(file_path):
            print(f"Downloading {file_path}...")
            urllib.request.urlretrieve(url, file_path)

    def on_train_start(self):
        if self.global_rank == 0:
            device = self.device
            vmb = torch.cuda.memory_allocated(device) / (1024 ** 2)
            print(f"\n[massive] VRAM allocated at start: {vmb:.1f} MB on rank {self.global_rank}")
            print(f"[massive] Model structure:\n{self.model}\n")

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits.view(-1, self.vocab_size), y.view(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=4096)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--ff_mult", type=int, default=4)
    parser.add_argument("--devices", type=int, default=2)
    parser.add_argument("--strategy", type=str, default="ddp")
    parser.add_argument("--max_epochs", type=int, default=2)
    args = parser.parse_args()

    strategy = args.strategy
    if args.devices == 1:
        strategy = "auto"

    print(f"[fsdp] Strategy: {strategy}, Devices: {args.devices}")
    print(f"[fsdp] Model: d_model={args.d_model}, layers={args.num_layers}")

    model = MassiveLitModel(
        d_model=args.d_model,
        num_layers=args.num_layers,
        ff_mult=args.ff_mult,
    )
    model.prepare_data()

    dataset = ShakespeareDataset(seq_len=args.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    trainer = L.Trainer(
        accelerator="gpu",
        devices=args.devices,
        num_nodes=1,
        strategy=strategy,
        max_epochs=args.max_epochs,
        logger=CSVLogger("logs", name="massive_model"),
    )

    trainer.fit(model, loader)

if __name__ == "__main__":
    main()
