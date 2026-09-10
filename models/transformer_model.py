"""
transformer_model.py
--------------------
Temporal Fusion Transformer (TFT-lite) for stock price regression:
  • Positional Encoding (sinusoidal)
  • Multi-Head Self-Attention encoder blocks
  • Feed-Forward sublayers with GELU + residual connections
  • LayerNorm throughout
  • Regression head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Positional Encoding
# ─────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)          # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ─────────────────────────────────────────────
# Transformer Encoder Block
# ─────────────────────────────────────────────

class TransformerEncoderBlock(nn.Module):
    """Single Transformer encoder block with Pre-LN."""

    def __init__(self, d_model: int, n_heads: int,
                 dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads,
                                           dropout=dropout,
                                           batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor,
                key_padding_mask=None) -> torch.Tensor:
        # Pre-LN self-attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed,
                                key_padding_mask=key_padding_mask)
        x = x + attn_out
        # Pre-LN feed-forward
        x = x + self.ff(self.norm2(x))
        return x


# ─────────────────────────────────────────────
# Temporal Transformer (TFT-lite)
# ─────────────────────────────────────────────

class TransformerModel(nn.Module):
    """
    Lightweight Temporal Fusion Transformer for 1-step-ahead regression.

    Parameters
    ----------
    input_size  : feature dimension per time-step
    d_model     : internal embedding dimension (must be divisible by n_heads)
    n_heads     : number of attention heads
    n_layers    : number of encoder blocks
    dim_ff      : feed-forward inner dimension
    dropout     : dropout probability
    seq_len     : input sequence length (for CLS token)
    output_size : regression outputs (default 1)
    """

    def __init__(self,
                 input_size:  int   = 30,
                 d_model:     int   = 128,
                 n_heads:     int   = 8,
                 n_layers:    int   = 4,
                 dim_ff:      int   = 256,
                 dropout:     float = 0.1,
                 seq_len:     int   = 60,
                 output_size: int   = 1):
        super().__init__()
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model)
        )

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_enc = PositionalEncoding(d_model,
                                          max_len=seq_len + 1,
                                          dropout=dropout)

        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, dim_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Regression head on CLS token
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )

    def forward(self, x: torch.Tensor):
        """
        x : (B, T, input_size)
        returns : (B, output_size)
        """
        B = x.size(0)
        x = self.input_proj(x)                        # (B, T, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)         # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)               # (B, T+1, d_model)
        x   = self.pos_enc(x)

        for block in self.encoder_blocks:
            x = block(x)

        x = self.norm(x)
        cls_out = x[:, 0, :]                           # (B, d_model) – CLS token
        return self.head(cls_out), None                 # None = no separate attn


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

def build_transformer(config: dict) -> TransformerModel:
    return TransformerModel(
        input_size  = config["input_size"],
        d_model     = config.get("d_model",      128),
        n_heads     = config.get("n_heads",        8),
        n_layers    = config.get("n_layers",       4),
        dim_ff      = config.get("dim_ff",       256),
        dropout     = config.get("dropout",      0.1),
        seq_len     = config.get("seq_len",       60),
        output_size = config.get("output_size",    1),
    )
