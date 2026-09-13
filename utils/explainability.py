"""
explainability.py
-----------------
Model-agnostic explainability tools for stock price forecasting.

Methods
-------
1. permutation_importance()
   Shuffles each feature column independently and measures how much
   the model's RMSE increases. A large increase = the feature is critical.
   No external dependencies (just numpy + the trained model).

2. extract_attention_weights()
   Extracts Bahdanau attention weights from the LSTM model over the test set.
   These show which timesteps in the lookback window the model focuses on.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def permutation_importance(model,
                           X_test: np.ndarray,
                           y_true: np.ndarray,
                           feature_names: list,
                           device: str = "cpu",
                           n_repeats: int = 3,
                           batch_size: int = 128) -> dict:
    """
    Compute permutation feature importance for any PyTorch model
    that returns (prediction, attn_weights).

    Algorithm
    ---------
    For each feature f:
      1. Shuffle the values of feature f across all test samples.
      2. Run the model on the perturbed data.
      3. Compute RMSE on the perturbed predictions.
      4. Importance score = (perturbed RMSE - baseline RMSE).
         → A larger score means the feature is more important.
    Repeat n_repeats times and average to reduce variance.

    Parameters
    ----------
    model         : nn.Module — trained PyTorch model
    X_test        : np.ndarray of shape (N, T, F)
    y_true        : np.ndarray of shape (N,) — true values (original scale)
    feature_names : list of length F — feature column names
    device        : 'cpu' or 'cuda'
    n_repeats     : number of shuffles per feature (default 3)
    batch_size    : inference batch size

    Returns
    -------
    importances : dict {feature_name: importance_score}
    """
    dev = torch.device(device)
    model.eval()
    model.to(dev)

    def _predict(X: np.ndarray) -> np.ndarray:
        """Run model inference and return predictions."""
        ds = TensorDataset(torch.FloatTensor(X))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for (xb,) in dl:
                xb = xb.to(dev)
                out, _ = model(xb)
                preds.append(out.cpu().numpy())
        return np.concatenate(preds, axis=0).flatten()

    def _rmse(y_true, y_pred):
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # Baseline RMSE (unperturbed)
    baseline_pred = _predict(X_test)
    baseline_rmse = _rmse(y_true, baseline_pred)
    print(f"[Explainability] Baseline RMSE: {baseline_rmse:.4f}")
    print(f"[Explainability] Computing permutation importance "
          f"for {len(feature_names)} features x {n_repeats} repeats...")

    importances = {}
    n_features  = X_test.shape[2]   # F dimension

    for f_idx, f_name in enumerate(feature_names):
        scores = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            # Shuffle this feature across all samples (preserves time structure)
            perm_order = np.random.permutation(len(X_perm))
            X_perm[:, :, f_idx] = X_perm[perm_order, :, f_idx]
            perm_pred = _predict(X_perm)
            perm_rmse = _rmse(y_true, perm_pred)
            scores.append(perm_rmse - baseline_rmse)

        importances[f_name] = float(np.mean(scores))
        if f_idx % 10 == 0:
            print(f"  Feature {f_idx + 1}/{n_features}: {f_name} "
                  f"-> importance={importances[f_name]:.5f}")

    # Sort by importance descending
    importances = dict(sorted(importances.items(),
                               key=lambda x: x[1], reverse=True))
    print("[Explainability] Permutation importance complete.")
    return importances


def extract_attention_weights(model,
                              X_test: np.ndarray,
                              device: str = "cpu",
                              batch_size: int = 128) -> np.ndarray:
    """
    Extract LSTM attention weights from the test set.

    Parameters
    ----------
    model      : LSTMModel — must have use_attention=True
    X_test     : np.ndarray of shape (N, T, F)
    device     : 'cpu' or 'cuda'
    batch_size : inference batch size

    Returns
    -------
    attn_weights : np.ndarray of shape (N, T), or None if model
                   does not return attention weights.
    """
    dev = torch.device(device)
    model.eval()
    model.to(dev)

    ds = TensorDataset(torch.FloatTensor(X_test))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_attns = []
    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(dev)
            _, attn = model(xb)
            if attn is None:
                return None
            all_attns.append(attn.cpu().numpy())

    if not all_attns:
        return None
    return np.concatenate(all_attns, axis=0)   # (N, T)
