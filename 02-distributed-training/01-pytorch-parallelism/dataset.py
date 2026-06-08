import os
import urllib.request
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

class ShakespeareDataset(Dataset):
    def __init__(self, seq_len=32, train=True):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        file_path = "shakespeare.txt"
        
        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0
        
        if rank == 0:
            if not os.path.exists(file_path):
                print(f"Downloading {file_path}...")
                urllib.request.urlretrieve(url, file_path)
        
        if is_dist:
            dist.barrier()
            
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
