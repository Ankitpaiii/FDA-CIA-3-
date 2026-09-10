"""
ensemble_model.py
-----------------
Stacking Ensemble that combines:
  1. Bi-LSTM with attention
  2. Transformer (TFT-lite)
  3. GRU baseline

The meta-learner is a small MLP trained on the
concatenated hidden representations of all base models.
"""

import torch
import torch.nn as nn
from models.lstm_model        import LSTMModel
from models.transformer_model import TransformerModel


# ─────────────────────────────────────────────
# GRU Baseline
# ─────────────────────────────────────────────

class GRUModel(nn.Module):
    """Simple stacked GRU for use as the third ensemble member."""

    def __init__(self, input_size: int = 30,
                 hidden_size: int = 64,
                 num_layers:  int = 2,
                 dropout:     float = 0.2,
                 output_size: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.GELU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x: torch.Tensor):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]), None


# ─────────────────────────────────────────────
# Meta-Learner
# ─────────────────────────────────────────────

class MetaLearner(nn.Module):
    """MLP meta-learner that ingests stacked base-model predictions."""

    def __init__(self, n_base: int = 3, output_size: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_base, 16),
            nn.GELU(),
            nn.Linear(16, output_size)
        )

    def forward(self, preds: torch.Tensor) -> torch.Tensor:
        # preds : (B, n_base)
        return self.net(preds)


# ─────────────────────────────────────────────
# Stacking Ensemble
# ─────────────────────────────────────────────

class EnsembleModel(nn.Module):
    """
    End-to-end stacking ensemble.

    The forward pass:
      1. Runs each base model independently.
      2. Concatenates their scalar predictions.
      3. Feeds the concat through the meta-learner.

    Parameters
    ----------
    lstm_cfg        : dict for LSTMModel
    transformer_cfg : dict for TransformerModel
    gru_cfg         : dict for GRUModel
    freeze_base     : if True, freeze base model weights during fine-tuning
    """

    def __init__(self,
                 lstm_cfg:        dict,
                 transformer_cfg: dict,
                 gru_cfg:         dict,
                 freeze_base:     bool = False):
        super().__init__()

        self.lstm        = LSTMModel(**lstm_cfg)
        self.transformer = TransformerModel(**transformer_cfg)
        self.gru         = GRUModel(**gru_cfg)
        self.meta        = MetaLearner(n_base=3, output_size=1)

        if freeze_base:
            for m in [self.lstm, self.transformer, self.gru]:
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, x: torch.Tensor):
        """
        x : (B, T, input_size)
        returns : (B, 1) final prediction
        """
        p_lstm, _  = self.lstm(x)        # (B, 1)
        p_trans, _ = self.transformer(x) # (B, 1)
        p_gru, _   = self.gru(x)         # (B, 1)

        # Stack base predictions → (B, 3)
        stacked = torch.cat([p_lstm, p_trans, p_gru], dim=-1)
        return self.meta(stacked), None


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

def build_ensemble(config: dict) -> EnsembleModel:
    """
    config keys
    -----------
    input_size  : int
    seq_len     : int
    freeze_base : bool  (optional, default False)
    """
    inp  = config["input_size"]
    sl   = config.get("seq_len", 60)
    frz  = config.get("freeze_base", False)

    lstm_cfg = {
        "input_size":    inp,
        "hidden_size":   128,
        "num_layers":    3,
        "dropout":       0.3,
        "bidirectional": True,
        "use_attention": True,
        "output_size":   1,
    }
    transformer_cfg = {
        "input_size":  inp,
        "d_model":     128,
        "n_heads":     8,
        "n_layers":    4,
        "dim_ff":      256,
        "dropout":     0.1,
        "seq_len":     sl,
        "output_size": 1,
    }
    gru_cfg = {
        "input_size":  inp,
        "hidden_size": 64,
        "num_layers":  2,
        "dropout":     0.2,
        "output_size": 1,
    }
    return EnsembleModel(lstm_cfg, transformer_cfg, gru_cfg,
                         freeze_base=frz)
