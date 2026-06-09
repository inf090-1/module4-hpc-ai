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

class TineLLM(nn.Module):
    """A tiny GPT-like decoder-only transformer.

    This variant is intentionally small to keep the weight footprint low.
    """

    def __init__(
        self,
        vocab_size: int = 65,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        max_seq_len: int = 32,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Head projects hidden states back to vocabulary logits.
        # Bias is omitted to slightly reduce parameter count.
        self.linear = nn.Linear(d_model, vocab_size, bias=False)

        # Cache the causal mask so we don't regenerate it every batch.
        mask = nn.Transformer.generate_square_subsequent_mask(max_seq_len)
        self.register_buffer("causal_mask", mask)

    def forward(self, src):
        # src: [B, S]
        src = self.embedding(src)  # [B, S, d_model]
        src = self.pos_encoder(src)

        seq_len = src.size(1)
        if seq_len > self.causal_mask.size(0):
            # Fallback for unexpected seq_len; should not happen in the examples.
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(src.device)
        else:
            mask = self.causal_mask[:seq_len, :seq_len]

        # Decoder-only behavior comes from the causal mask.
        output = self.transformer_encoder(src, mask=mask, is_causal=True)
        output = self.linear(output)  # [B, S, vocab]
        return output

