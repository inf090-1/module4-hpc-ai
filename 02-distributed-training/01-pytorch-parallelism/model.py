import math
import torch
import torch.nn as nn

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
        
        # Causal mask for autoregressive prediction
        seq_len = src.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(src.device)
        
        output = self.transformer_encoder(src, mask=mask, is_causal=True)
        output = self.linear(output)
        return output
