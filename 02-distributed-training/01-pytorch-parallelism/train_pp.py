import argparse
import torch
import torch.nn as nn
from model import PositionalEncoding
import math
from dataset import ShakespeareDataset
from torch.utils.data import DataLoader

class PipelineParallelLLM(nn.Module):
    def __init__(self, vocab_size=65, d_model=512, nhead=8, num_layers=4):
        super().__init__()
        self.dev0 = "cuda:0"
        self.dev1 = "cuda:1"
        
        # Part 1 on GPU 0
        self.embedding = nn.Embedding(vocab_size, d_model).to(self.dev0)
        self.pos_encoder = PositionalEncoding(d_model).to(self.dev0)
        encoder_layer1 = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_part1 = nn.TransformerEncoder(encoder_layer1, num_layers=num_layers//2).to(self.dev0)
        
        # Part 2 on GPU 1
        encoder_layer2 = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_part2 = nn.TransformerEncoder(encoder_layer2, num_layers=num_layers//2).to(self.dev1)
        self.linear = nn.Linear(d_model, vocab_size).to(self.dev1)

    def forward(self, src):
        # GPU 0 forward
        src_dev0 = src.to(self.dev0)
        seq_len = src.size(1)
        mask_dev0 = nn.Transformer.generate_square_subsequent_mask(seq_len).to(self.dev0)
        
        x = self.embedding(src_dev0)
        x = self.pos_encoder(x)
        x = self.transformer_part1(x, mask=mask_dev0, is_causal=True)
        
        # Transfer to GPU 1
        x_dev1 = x.to(self.dev1)
        mask_dev1 = mask_dev0.to(self.dev1)
        
        # GPU 1 forward
        x = self.transformer_part2(x_dev1, mask=mask_dev1, is_causal=True)
        output = self.linear(x)
        return output

def train(max_batches=None):
    if torch.cuda.device_count() < 2:
        print("Need at least 2 GPUs for Pipeline Parallelism.")
        return

    dataset = ShakespeareDataset(seq_len=64, train=True)
    vocab_size = dataset.vocab_size
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = PipelineParallelLLM(vocab_size=vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(2):
        for batch_idx, (inputs, labels) in enumerate(loader):
            if max_batches and batch_idx >= max_batches:
                break
            # Labels need to be on the device where the output is produced (GPU 1)
            labels = labels.to("cuda:1")
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))
            
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_batches", type=int, default=None)
    args = parser.parse_args()
    train(max_batches=args.max_batches)
