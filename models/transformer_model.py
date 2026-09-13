"""
transformer_model.py
--------------------
Temporal Fusion Transformer (TFT-lite) for stock price regression.

Architecture
------------
  • Input projection: Linear → LayerNorm
  • Learnable positional encoding (trainable, adapts to financial data)
    + sinusoidal fallback added to the learnable embed
  • Learnable CLS token that aggregates sequence information
  • Multi-Head Self-Attention encoder blocks (Pre-LayerNorm)
  • Feed-Forward sublayers with GELU activation + residual connections
  • Regression head on CLS token output

Why learnable PE for financial data?
  Sinusoidal PE assumes fixed periodic structure. Financial time series
  has irregular seasonality, so a learned PE adapts more effectively.
"""

import math
import torch
import torch.nn as nn


# ─────────────────────────────────────────────
# Positional Encoding (Hybrid: learned + sinusoidal)
# ─────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Hybrid positional encoding combining:
      1. Learnable position embeddings (adapt to data distribution)
      2. Fixed sinusoidal base (provides initialisation structure)

    The learnable component is initialized to zeros so training
    starts from the sinusoidal baseline.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Fixed sinusoidal encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        self.register_buffer("pe_fixed", pe.unsqueeze(0))  # (1, max_len, d_model)

        # Learnable component (initialized to 0 → starts from sinusoidal)
        self.pe_learned = nn.Parameter(torch.zeros(1, max_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        pos = self.pe_fixed[:, :T, :] + self.pe_learned[:, :T, :]
        return self.dropout(x + pos)


# ─────────────────────────────────────────────
# Transformer Encoder Block (Pre-LN)
# ─────────────────────────────────────────────

class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer encoder block using Pre-LayerNorm.

    Pre-LN is more stable than Post-LN for training deep networks
    as it prevents gradient explosion in early training.
    """

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
        # Pre-LN Multi-Head Self-Attention
        normed   = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed,
                                key_padding_mask=key_padding_mask)
        x = x + attn_out

        # Pre-LN Feed-Forward
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
    input_size  : feature dimension per time-step (F)
    d_model     : internal embedding dimension (must be divisible by n_heads)
    n_heads     : number of attention heads
    n_layers    : number of encoder blocks
    dim_ff      : feed-forward inner dimension (typically 2×–4× d_model)
    dropout     : dropout probability
    seq_len     : input sequence length (for positional encoding)
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

        # Input projection + normalization
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model)
        )

        # Learnable CLS token (sequence-level aggregator)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Hybrid positional encoding (learned + sinusoidal)
        self.pos_enc = PositionalEncoding(d_model,
                                          max_len=seq_len + 1,
                                          dropout=dropout)

        # Stacked encoder blocks
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
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x: torch.Tensor):
        """
        x : (B, T, input_size)

        Returns
        -------
        pred : (B, output_size)
        None : (no separate attention weights returned)
        """
        B = x.size(0)
        # Residual anchor: last observed Close price (feature index 0)
        last_close = x[:, -1, 0:1]

        h = self.input_proj(x)                    # (B, T, d_model)

        # Prepend CLS token to sequence
        cls = self.cls_token.expand(B, -1, -1)    # (B, 1, d_model)
        h   = torch.cat([cls, h], dim=1)          # (B, T+1, d_model)
        h   = self.pos_enc(h)

        for block in self.encoder_blocks:
            h = block(h)

        h = self.norm(h)
        cls_out = h[:, 0, :]                       # (B, d_model) — CLS token
        delta = self.head(cls_out)
        return last_close + delta, None


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

def build_transformer(config: dict) -> TransformerModel:
    """Build a TransformerModel from a flat config dictionary."""
    return TransformerModel(
        input_size  = config["input_size"],
        d_model     = config.get("d_model",     128),
        n_heads     = config.get("n_heads",       8),
        n_layers    = config.get("n_layers",      4),
        dim_ff      = config.get("dim_ff",      256),
        dropout     = config.get("dropout",     0.1),
        seq_len     = config.get("seq_len",      60),
        output_size = config.get("output_size",   1),
    )
