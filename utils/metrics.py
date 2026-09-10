"""
metrics.py
----------
Evaluation metrics for regression-based stock price forecasting.

Metrics implemented
-------------------
  • RMSE  – Root Mean Squared Error
  • MAE   – Mean Absolute Error
  • MAPE  – Mean Absolute Percentage Error
  • R²    – Coefficient of Determination
  • DA    – Directional Accuracy (% correct direction of change)
  • Theil's U  – Relative prediction quality vs. naïve model
"""

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (returns %)."""
    mask = np.abs(y_true) > 1e-9   # avoid division by zero
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask])
                                 / y_true[mask])) * 100)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Determination R²."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-9))


def directional_accuracy(y_true: np.ndarray,
                          y_pred: np.ndarray) -> float:
    """
    Percentage of time-steps where the model correctly predicts
    the direction of price movement.
    """
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    correct  = np.sum(true_dir == pred_dir)
    return float(correct / len(true_dir) * 100)


def theils_u(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Theil's U statistic.
    U < 1  → model better than naïve random walk.
    U = 1  → model equals naïve random walk.
    U > 1  → model worse than naïve random walk.
    """
    naive   = y_true[:-1]              # naïve: previous value
    actual  = y_true[1:]
    forecast = y_pred[1:]

    mse_model = np.mean((actual - forecast) ** 2)
    mse_naive = np.mean((actual - naive)    ** 2)
    return float(np.sqrt(mse_model / (mse_naive + 1e-9)))


def evaluate_all(y_true: np.ndarray,
                 y_pred: np.ndarray,
                 label:  str = "Model") -> dict:
    """
    Compute all metrics and return a summary dict.

    Parameters
    ----------
    y_true : ground-truth prices (original scale)
    y_pred : predicted prices   (original scale)
    label  : display name for print output
    """
    metrics = {
        "RMSE":  rmse(y_true, y_pred),
        "MAE":   mae(y_true, y_pred),
        "MAPE":  mape(y_true, y_pred),
        "R2":    r2_score(y_true, y_pred),
        "DA":    directional_accuracy(y_true, y_pred),
        "TheilU": theils_u(y_true, y_pred),
    }

    print(f"\n{'='*45}")
    print(f"  {label} – Evaluation Metrics")
    print(f"{'='*45}")
    print(f"  RMSE           : {metrics['RMSE']:.4f}")
    print(f"  MAE            : {metrics['MAE']:.4f}")
    print(f"  MAPE           : {metrics['MAPE']:.2f}%")
    print(f"  R²             : {metrics['R2']:.4f}")
    print(f"  Directional Acc: {metrics['DA']:.2f}%")
    print(f"  Theil's U      : {metrics['TheilU']:.4f}")
    print(f"{'='*45}\n")

    return metrics
