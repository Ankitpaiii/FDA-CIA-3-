"""
ensemble_model.py
-----------------
Stacking Ensemble that combines three base learners:
  1. Bidirectional LSTM with Bahdanau Attention
  2. Temporal Fusion Transformer (TFT-lite)
  3. GRU (lightweight recurrent baseline)

The meta-learner is an MLP trained on the concatenated scalar
predictions of all base models, with a residual shortcut connection.

Architecture Improvements (v2)
-------------------------------
  • Deeper meta-learner: 3→8→4→1 with BatchNorm + Dropout
  • Residual shortcut in meta-learner (simple mean of base preds)
  • Separate forward modes: base_only() for per-model diagnostics
  • Base model configs are leaner (avoid overfitting in joint training)
"""

import torch
import torch.nn as nn
from models.lstm_model        import LSTMModel
from models.transformer_model import TransformerModel


# ─────────────────────────────────────────────
# GRU Baseline
# ─────────────────────────────────────────────

class GRUModel(nn.Module):
    """
    Lightweight stacked GRU — serves as the 3rd ensemble member.
    Provides diversity through a simpler recurrent architecture.
    """

    def __init__(self, input_size: int = 30,
                 hidden_size: int = 96,
                 num_layers:  int = 2,
                 dropout:     float = 0.25,
                 output_size: int = 1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )
        self.gru = nn.GRU(hidden_size, hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 48),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(48, output_size)
        )

    def forward(self, x: torch.Tensor):
        last_close = x[:, -1, 0:1]
        h   = self.input_proj(x)               # (B, T, H)
        out, _ = self.gru(h)
        delta = self.head(out[:, -1, :])
        return last_close + delta, None   # last timestep


# ─────────────────────────────────────────────
# Improved Meta-Learner
# ─────────────────────────────────────────────

class MetaLearner(nn.Module):
    """
    MLP meta-learner that combines base-model scalar predictions.

    Architecture: 3 → 8 → 4 → 1
    With:
      • BatchNorm after first layer (stabilizes training)
      • Dropout for regularisation
      • Residual shortcut: adds mean(base_preds) to final output
        so the meta-learner only needs to learn a correction term
    """

    def __init__(self, n_base: int = 3, output_size: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_base, 8),
            nn.BatchNorm1d(8),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(8, 4),
            nn.GELU(),
            nn.Linear(4, output_size)
        )

    def forward(self, preds: torch.Tensor) -> torch.Tensor:
        """
        preds : (B, n_base) — stacked base-model predictions
        Returns (B, 1) — final blended prediction
        """
        residual = preds.mean(dim=-1, keepdim=True)   # (B, 1)
        return self.net(preds) + residual              # residual shortcut


# ─────────────────────────────────────────────
# Stacking Ensemble
# ─────────────────────────────────────────────

class EnsembleModel(nn.Module):
    """
    End-to-end differentiable stacking ensemble.

    Forward pass:
      1. Run each base model independently on the same input.
      2. Concatenate their scalar predictions → (B, 3).
      3. Feed through the meta-learner with residual shortcut.

    Parameters
    ----------
    lstm_cfg        : dict for LSTMModel
    transformer_cfg : dict for TransformerModel
    gru_cfg         : dict for GRUModel
    freeze_base     : if True, freeze base model weights
                      (useful for pre-trained bases)
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

    def base_predictions(self, x: torch.Tensor):
        """
        Return individual base-model predictions (for diagnostics).
        Returns: (p_lstm, p_transformer, p_gru) — each shape (B, 1)
        """
        p_lstm,  _ = self.lstm(x)
        p_trans, _ = self.transformer(x)
        p_gru,   _ = self.gru(x)
        return p_lstm, p_trans, p_gru

    def forward(self, x: torch.Tensor):
        """
        x : (B, T, input_size)
        Returns : (B, 1) final blended prediction, None
        """
        p_lstm, p_trans, p_gru = self.base_predictions(x)

        # Stack base predictions → (B, 3)
        stacked = torch.cat([p_lstm, p_trans, p_gru], dim=-1)
        return self.meta(stacked), None


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

def build_ensemble(config: dict) -> EnsembleModel:
    """
    Build EnsembleModel from a flat config dictionary.

    Config keys
    -----------
    input_size  : int   — feature dimension per time-step
    seq_len     : int   — input sequence length (default 60)
    freeze_base : bool  — freeze base model weights (default False)
    """
    inp = config["input_size"]
    sl  = config.get("seq_len",     60)
    frz = config.get("freeze_base", False)

    lstm_cfg = {
        "input_size":    inp,
        "hidden_size":   96,       # leaner than standalone (128) to avoid overfit
        "num_layers":    2,        # 2 layers in ensemble; standalone uses 3
        "dropout":       0.25,
        "bidirectional": True,
        "use_attention": True,
        "output_size":   1,
    }
    transformer_cfg = {
        "input_size":  inp,
        "d_model":     96,
        "n_heads":     8,
        "n_layers":    3,
        "dim_ff":      192,
        "dropout":     0.1,
        "seq_len":     sl,
        "output_size": 1,
    }
    gru_cfg = {
        "input_size":  inp,
        "hidden_size": 96,
        "num_layers":  2,
        "dropout":     0.25,
        "output_size": 1,
    }
    return EnsembleModel(lstm_cfg, transformer_cfg, gru_cfg,
                         freeze_base=frz)
