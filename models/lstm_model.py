"""
lstm_model.py
-------------
Stacked Bidirectional LSTM with:
  • Input projection (Linear → LayerNorm → GELU)
  • Dropout + Recurrent Dropout regularisation
  • Bahdanau-style additive Attention Pooling
  • Deeper regression head: (H*2) → 128 → 64 → 1
  • Returns attention weights for interpretability / visualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Attention Pooling (Bahdanau-style)
# ─────────────────────────────────────────────

class AttentionPooling(nn.Module):
    """
    Additive (Bahdanau-style) self-attention over the time dimension.

    Collapses (B, T, H) → (B, H) by computing a weighted sum
    where the weights reflect which time-steps are most relevant
    for predicting the next price.

    The returned attention weights (B, T) can be visualized as a
    heatmap over the input window to understand model focus.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        # Two-layer attention scorer for richer expressiveness
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor):
        """
        x       : (B, T, H)
        returns : context (B, H), weights (B, T)
        """
        scores  = self.attn(x).squeeze(-1)           # (B, T)
        weights = F.softmax(scores, dim=-1)           # (B, T) — sum to 1
        context = (weights.unsqueeze(-1) * x).sum(dim=1)  # (B, H)
        return context, weights


# ─────────────────────────────────────────────
# Stacked Bi-LSTM with Attention
# ─────────────────────────────────────────────

class LSTMModel(nn.Module):
    """
    Stacked Bidirectional LSTM for 1-step-ahead price regression.

    Architecture
    ------------
    Input (B, T, F)
      → Input Projection (Linear → LayerNorm → GELU)
      → Stacked BiLSTM (num_layers, hidden_size per direction)
      → Bahdanau Attention Pooling (B, T, H*2) → (B, H*2)
      → Regression Head: H*2 → 128 → 64 → 1

    Parameters
    ----------
    input_size    : number of features per time-step (F)
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
        self.bidirectional  = bidirectional
        self.use_attention  = use_attention
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size    = hidden_size

        # Input projection: maps raw features to LSTM hidden dimension
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )

        # Stacked Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size    = hidden_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = bidirectional
        )

        lstm_out_dim = hidden_size * self.num_directions

        # Attention pooling (optional)
        if use_attention:
            self.attn_pool = AttentionPooling(lstm_out_dim)

        # Deeper regression head for richer mapping
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout / 4),
            nn.Linear(64, output_size)
        )

    def forward(self, x: torch.Tensor):
        """
        x : (B, T, input_size)

        Returns
        -------
        pred         : (B, output_size) — predicted price(s)
        attn_weights : (B, T) — attention over time, or None
        """
        # Residual anchor: last observed Close price (feature index 0)
        last_close = x[:, -1, 0:1]

        # Project input to hidden dimension
        h = self.input_proj(x)         # (B, T, H)
        out, _ = self.lstm(h)          # (B, T, H * num_directions)

        if self.use_attention:
            context, attn_w = self.attn_pool(out)   # (B, H*D), (B, T)
            delta = self.head(context)
            pred = last_close + delta
            return pred, attn_w
        else:
            context = out[:, -1, :]    # fallback: use last timestep
            delta = self.head(context)
            pred = last_close + delta
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
