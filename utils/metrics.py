"""
metrics.py
----------
Comprehensive evaluation metrics for stock price forecasting.

Metrics
-------
  • RMSE   – Root Mean Squared Error (lower is better)
  • MAE    – Mean Absolute Error (lower is better)
  • MAPE   – Mean Absolute Percentage Error in % (lower is better)
  • SMAPE  – Symmetric MAPE — less biased than MAPE (lower is better)
  • R²     – Coefficient of Determination (higher is better, max=1.0)
  • DA     – Directional Accuracy, % correct direction (>50% beats coin flip)
  • Theil's U  – U<1 beats naïve random walk, U>1 is worse
  • Max Error  – Maximum single-step absolute error
  • Hit Rate   – % times model signals correct buy/sell trade
"""

import numpy as np


# ─────────────────────────────────────────────
# Individual Metric Functions
# ─────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error. Penalizes large errors heavily."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error. Robust average error in original price units."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error (returns %).
    Note: biased toward low actual values.
    """
    mask = np.abs(y_true) > 1e-9
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask])
                                 / y_true[mask])) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error (returns %).
    Less biased than MAPE; treats over- and under-prediction equally.
    Range: 0% to 200%.
    """
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask  = denom > 1e-9
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])
                         / denom[mask]) * 100)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient of Determination R².
    R²=1.0 → perfect fit. R²=0 → model equals mean baseline. R²<0 → worse than mean.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-9))


def directional_accuracy(y_true: np.ndarray,
                          y_pred: np.ndarray) -> float:
    """
    Percentage of time-steps where the model correctly predicts
    the direction of price movement (up or down).
    >50% = better than random. >55% = commercially relevant.
    """
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    correct  = np.sum(true_dir == pred_dir)
    return float(correct / len(true_dir) * 100)


def theils_u(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Theil's U statistic vs. naïve random walk (persistence model).
    U < 1  → model is better than naïve 'no change' prediction.
    U = 1  → model equals naïve.
    U > 1  → naïve model wins; our model is not useful.
    """
    naive    = y_true[:-1]      # naïve: last observed value
    actual   = y_true[1:]
    forecast = y_pred[1:]
    mse_model = np.mean((actual - forecast) ** 2)
    mse_naive = np.mean((actual - naive)    ** 2)
    return float(np.sqrt(mse_model / (mse_naive + 1e-9)))


def max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Maximum single absolute prediction error (worst-case)."""
    return float(np.max(np.abs(y_true - y_pred)))


def hit_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Hit Rate for a long/short trading strategy.
    Buy when model predicts price increase; sell when decrease.
    Returns % of trades that were profitable.
    """
    actual_returns = np.diff(y_true)          # +ve = price went up
    pred_signals   = np.sign(np.diff(y_pred)) # +1 = predict up, -1 = predict down
    # A 'hit' is when our signal aligns with actual direction
    hits = np.sum((pred_signals > 0) & (actual_returns > 0)) + \
           np.sum((pred_signals < 0) & (actual_returns < 0))
    total_signals = np.sum(pred_signals != 0)
    return float(hits / max(total_signals, 1) * 100)


def simulate_pnl(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Simulate cumulative Profit & Loss for a simple long/short strategy.

    Logic
    -----
    - If model predicts price UP next day  → go long (buy)
    - If model predicts price DOWN next day → go short (sell)
    - PnL = daily actual return * position

    Returns
    -------
    cumulative_pnl : np.ndarray  (cumulative sum of daily strategy returns)
    """
    actual_returns = np.diff(y_true) / (y_true[:-1] + 1e-9)  # pct returns
    pred_signals   = np.sign(np.diff(y_pred))                  # +1 or -1
    daily_pnl      = pred_signals * actual_returns
    return np.cumsum(daily_pnl)


# ─────────────────────────────────────────────
# Master Evaluation Function
# ─────────────────────────────────────────────

def evaluate_all(y_true: np.ndarray,
                 y_pred: np.ndarray,
                 label:  str = "Model") -> dict:
    """
    Compute all metrics and return a summary dict.

    Parameters
    ----------
    y_true : ground-truth prices (original scale, USD)
    y_pred : predicted prices   (original scale, USD)
    label  : display name for console output

    Returns
    -------
    dict with keys: RMSE, MAE, MAPE, SMAPE, R2, DA, TheilU, MaxError, HitRate
    """
    metrics = {
        "RMSE":     rmse(y_true, y_pred),
        "MAE":      mae(y_true, y_pred),
        "MAPE":     mape(y_true, y_pred),
        "SMAPE":    smape(y_true, y_pred),
        "R2":       r2_score(y_true, y_pred),
        "DA":       directional_accuracy(y_true, y_pred),
        "TheilU":   theils_u(y_true, y_pred),
        "MaxError": max_error(y_true, y_pred),
        "HitRate":  hit_rate(y_true, y_pred),
    }

    # ── Interpretation Guide ─────────────────
    r2_interp    = "Excellent" if metrics["R2"] > 0.95 else \
                   "Good"      if metrics["R2"] > 0.90 else \
                   "Fair"      if metrics["R2"] > 0.80 else "Poor"
    u_interp     = "Beats naive" if metrics["TheilU"] < 1.0 else "Worse than naive"
    da_interp    = "Commercially relevant" if metrics["DA"] > 55 else \
                   "Better than random"    if metrics["DA"] > 50 else \
                   "Below random"

    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  {label} — Evaluation Metrics")
    print(sep)
    print(f"  {'Metric':<22} {'Value':>10}  {'Interpretation'}")
    print(f"  {'-'*50}")
    print(f"  {'RMSE ($)':<22} {metrics['RMSE']:>10.4f}  lower is better")
    print(f"  {'MAE ($)':<22} {metrics['MAE']:>10.4f}  lower is better")
    print(f"  {'MAPE (%)':<22} {metrics['MAPE']:>10.2f}%  lower is better")
    print(f"  {'SMAPE (%)':<22} {metrics['SMAPE']:>10.2f}%  symmetric, lower is better")
    print(f"  {'R2 Score':<22} {metrics['R2']:>10.4f}  {r2_interp}")
    print(f"  {'Directional Acc.':<22} {metrics['DA']:>10.2f}%  {da_interp}")
    print(f"  {'Theil U':<22} {metrics['TheilU']:>10.4f}  {u_interp}")
    print(f"  {'Max Error ($)':<22} {metrics['MaxError']:>10.4f}  worst-case error")
    print(f"  {'Hit Rate (%)':<22} {metrics['HitRate']:>10.2f}%  profitable trades")
    print(f"{sep}\n")

    return metrics
