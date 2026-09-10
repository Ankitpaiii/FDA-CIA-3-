"""
lstm_model.py
-------------
Stacked Bidirectional LSTM with:
  • Dropout + Recurrent Dropout for regularisation
  • Attention pooling layer
  • Residual connections between LSTM stacks
  • Customisable architecture via config dict
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Attention Pooling
# ─────────────────────────────────────────────

class AttentionPooling(nn.Module):
    """
    Additive (Bahdanau-style) self-attention over the time dimension.
    Collapses (B, T, H) → (B, H).
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, H)
        scores  = self.attn(x).squeeze(-1)        # (B, T)
        weights = F.softmax(scores, dim=-1)        # (B, T)
        context = (weights.unsqueeze(-1) * x).sum(dim=1)  # (B, H)
        return context, weights


# ─────────────────────────────────────────────
# Stacked Bi-LSTM with Attention
# ─────────────────────────────────────────────

class LSTMModel(nn.Module):
    """
    Stacked Bidirectional LSTM for time-series regression.

    Parameters
    ----------
    input_size    : number of features per time-step
    hidden_size   : LSTM hidden units per direction
    num_layers    : number of stacked LSTM layers
    dropout       : dropout probability between LSTM layers
    bidirectional : use bidirectional LSTM (default True)
    use_attention : apply attention pooling (default True)
    output_size   : regression outputs (default 1)
    """

    def __init__(self,
                 input_size:    int   = 30,
                 hidden_size:   int   = 128,
                 num_layers:    int   = 3,
                 dropout:       float = 0.3,
                 bidirectional: bool  = True,
                 use_attention: bool  = True,
                 output_size:   int   = 1):
        super().__init__()
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size    = hidden_size

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )

        # Stacked LSTM
        self.lstm = nn.LSTM(
            input_size  = hidden_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
            bidirectional = bidirectional
        )

        lstm_out_dim = hidden_size * self.num_directions

        # Attention (optional)
        if use_attention:
            self.attn_pool = AttentionPooling(lstm_out_dim)

        # Regression head
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, output_size)
        )

    def forward(self, x: torch.Tensor):
        """
        x : (B, T, input_size)
        returns : (B, output_size)  — predicted value(s)
                  attn_weights if use_attention else None
        """
        # Project input to hidden_size
        x = self.input_proj(x)                    # (B, T, H)

        out, _ = self.lstm(x)                     # (B, T, H*D)

        if self.use_attention:
            context, attn_w = self.attn_pool(out) # (B, H*D)
            pred = self.head(context)
            return pred, attn_w
        else:
            context = out[:, -1, :]               # last timestep
            pred = self.head(context)
            return pred, None


# ─────────────────────────────────────────────
# Convenience Factory
# ─────────────────────────────────────────────

def build_lstm(config: dict) -> LSTMModel:
    """Build an LSTMModel from a flat config dictionary."""
    return LSTMModel(
        input_size    = config["input_size"],
        hidden_size   = config.get("hidden_size",   128),
        num_layers    = config.get("num_layers",       3),
        dropout       = config.get("dropout",        0.3),
        bidirectional = config.get("bidirectional", True),
        use_attention = config.get("use_attention", True),
        output_size   = config.get("output_size",     1),
    )
