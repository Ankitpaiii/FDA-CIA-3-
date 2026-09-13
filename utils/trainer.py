"""
trainer.py
----------
Universal training loop for all models (LSTM / Transformer / Ensemble).

Features
--------
  • Adam + CosineAnnealingWarmRestarts scheduler (fixed step call)
  • Early stopping on validation loss
  • Gradient clipping (max-norm = 1.0)
  • Best-model checkpointing (outputs/checkpoints/)
  • tqdm progress bars per epoch
  • Per-epoch CSV log to outputs/training_log.csv
  • Seed support for reproducibility
  • Supports any model that returns (prediction, attn_weights)
"""

import os
import csv
import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────
# Dataset Builder
# ─────────────────────────────────────────────

def build_loaders(X_train, y_train, X_val, y_val,
                  batch_size: int = 64):
    """Wrap numpy arrays into PyTorch DataLoaders."""
    train_ds = TensorDataset(torch.FloatTensor(X_train),
                             torch.FloatTensor(y_train))
    val_ds   = TensorDataset(torch.FloatTensor(X_val),
                             torch.FloatTensor(y_val))
    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True,  drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size,
                          shuffle=False, drop_last=False)
    return train_dl, val_dl


# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────

class Trainer:
    """
    Universal trainer for all PyTorch models.

    Parameters
    ----------
    model          : nn.Module  – model to train
    lr             : float      – initial learning rate
    weight_decay   : float      – L2 regularisation
    clip_grad      : float      – gradient clipping max-norm
    patience       : int        – early-stopping patience (epochs)
    checkpoint_dir : str        – directory to save best checkpoint
    log_dir        : str        – directory to save training_log.csv
    device         : str | torch.device
    print_every    : int        – print progress every N epochs
    """

    def __init__(self,
                 model,
                 lr:             float = 1e-3,
                 weight_decay:   float = 1e-4,
                 clip_grad:      float = 1.0,
                 patience:       int   = 20,
                 checkpoint_dir: str   = "outputs/checkpoints",
                 log_dir:        str   = "outputs",
                 device:         str   = "auto",
                 print_every:    int   = 5):
        self.model       = model
        self.clip_grad   = clip_grad
        self.patience    = patience
        self.ckpt_dir    = checkpoint_dir
        self.log_dir     = log_dir
        self.print_every = print_every
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available()
                                       else "cpu")
        else:
            self.device = torch.device(device)
        self.model.to(self.device)

        self.criterion  = nn.HuberLoss(delta=1.0)   # robust to outliers
        self.optimizer  = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        # Fixed: do NOT pass epoch arg to CosineAnnealingWarmRestarts.step()
        self.scheduler  = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )

        self.train_losses: list = []
        self.val_losses:   list = []
        print(f"[Trainer] Device: {self.device}")

    def _run_epoch(self, loader, train: bool) -> float:
        self.model.train(train)
        total_loss = 0.0
        n_samples  = 0
        with torch.set_grad_enabled(train):
            for batch in loader:
                xb, yb = batch
                xb = xb.to(self.device)
                yb = yb.to(self.device).unsqueeze(-1)

                pred, _ = self.model(xb)
                loss     = self.criterion(pred, yb)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(),
                                             self.clip_grad)
                    self.optimizer.step()

                total_loss += loss.item() * xb.size(0)
                n_samples  += xb.size(0)
        return total_loss / max(n_samples, 1)

    def fit(self,
            train_dl:    DataLoader,
            val_dl:      DataLoader,
            epochs:      int  = 100,
            model_name:  str  = "model") -> dict:
        """
        Train the model and return a history dict.

        Returns
        -------
        {
          "train_loss" : list[float],
          "val_loss"   : list[float],
          "best_epoch" : int,
          "best_val"   : float,
          "ckpt_path"  : str
        }
        """
        best_val   = float("inf")
        best_epoch = 0
        best_state = None
        no_improve = 0
        ckpt_path  = os.path.join(self.ckpt_dir, f"{model_name}_best.pt")
        log_path   = os.path.join(self.log_dir, "training_log.csv")

        # Prepare CSV log
        write_header = not os.path.exists(log_path)
        log_file = open(log_path, "a", newline="")
        log_writer = csv.writer(log_file)
        if write_header:
            log_writer.writerow(["model", "epoch", "train_loss", "val_loss", "lr"])

        print(f"\n[Trainer] Training '{model_name}' for up to {epochs} epochs ...")
        t0 = time.time()

        for epoch in range(1, epochs + 1):
            tr_loss  = self._run_epoch(train_dl, train=True)
            val_loss = self._run_epoch(val_dl,   train=False)

            # Fixed: call scheduler.step() without epoch argument
            self.scheduler.step()

            self.train_losses.append(tr_loss)
            self.val_losses.append(val_loss)

            lr_now = self.optimizer.param_groups[0]["lr"]

            # Log to CSV
            log_writer.writerow([model_name, epoch,
                                  f"{tr_loss:.6f}", f"{val_loss:.6f}",
                                  f"{lr_now:.2e}"])
            log_file.flush()

            if val_loss < best_val - 1e-6:
                best_val   = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                no_improve = 0
                torch.save(best_state, ckpt_path)
            else:
                no_improve += 1

            if epoch % self.print_every == 0 or epoch == 1:
                elapsed = time.time() - t0
                print(f"  Epoch {epoch:4d}/{epochs} | "
                      f"Train={tr_loss:.5f} | Val={val_loss:.5f} | "
                      f"LR={lr_now:.2e} | Elapsed={elapsed:.0f}s"
                      + (" *" if no_improve == 0 else ""))

            if no_improve >= self.patience:
                print(f"  [EarlyStopping] No improvement for {self.patience} "
                      f"epochs. Best epoch={best_epoch}. Stopping.")
                break

        log_file.close()

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)
        print(f"\n[Trainer] Best epoch={best_epoch}, Best val loss={best_val:.5f}")
        print(f"[Trainer] Checkpoint saved -> {ckpt_path}\n")

        return {
            "train_loss": self.train_losses,
            "val_loss":   self.val_losses,
            "best_epoch": best_epoch,
            "best_val":   best_val,
            "ckpt_path":  ckpt_path
        }

    @torch.no_grad()
    def predict(self, X: np.ndarray, batch_size: int = 128) -> np.ndarray:
        """Run inference on numpy array; returns numpy array of predictions."""
        self.model.eval()
        ds = TensorDataset(torch.FloatTensor(X))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        preds = []
        for (xb,) in dl:
            xb   = xb.to(self.device)
            out, _ = self.model(xb)
            preds.append(out.cpu().numpy())
        return np.concatenate(preds, axis=0).flatten()

    @torch.no_grad()
    def predict_with_attention(self, X: np.ndarray,
                               batch_size: int = 128):
        """
        Run inference and collect attention weights.

        Returns
        -------
        preds        : np.ndarray  shape (N,)
        attn_weights : np.ndarray  shape (N, T) or None
        """
        self.model.eval()
        ds = TensorDataset(torch.FloatTensor(X))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        preds, attns = [], []
        for (xb,) in dl:
            xb = xb.to(self.device)
            out, attn = self.model(xb)
            preds.append(out.cpu().numpy())
            if attn is not None:
                attns.append(attn.cpu().numpy())
        preds = np.concatenate(preds, axis=0).flatten()
        attn_weights = (np.concatenate(attns, axis=0)
                        if attns else None)
        return preds, attn_weights
