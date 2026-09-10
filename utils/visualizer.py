"""
visualizer.py
-------------
Publication-quality plotting for stock price forecasting results.

Plots generated
---------------
  1. Training & Validation loss curves (all models)
  2. Actual vs. Predicted price overlay (test set)
  3. Residual / Error distribution (histogram)
  4. Metrics comparison bar chart (RMSE, MAE, MAPE, R², DA)
  5. Candlestick + MA chart of historical data
  6. Feature correlation heatmap
"""

import os
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# Style
plt.rcParams.update({
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#e6edf3",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#e6edf3",
    "grid.color":        "#21262d",
    "grid.linewidth":    0.8,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "lines.linewidth":   2.0,
    "figure.titlesize":  15,
})

COLORS = {
    "actual":      "#58a6ff",
    "baseline":    "#8b949e",
    "lstm":        "#3fb950",
    "transformer": "#d2a8ff",
    "ensemble":    "#ffa657",
    "loss_train":  "#58a6ff",
    "loss_val":    "#f85149",
}

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PLOT_DIR = os.path.join(BASE_DIR, "outputs", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1. Loss Curves
# ─────────────────────────────────────────────

def plot_loss_curves(histories: dict, save: bool = True) -> str:
    """
    histories : {model_name: {"train_loss": [...], "val_loss": [...]}}
    """
    fig, axes = plt.subplots(1, len(histories),
                             figsize=(6 * len(histories), 5),
                             squeeze=False)
    fig.suptitle("Training & Validation Loss Curves", fontweight="bold", y=1.02)

    for ax, (name, h) in zip(axes[0], histories.items()):
        epochs = range(1, len(h["train_loss"]) + 1)
        ax.plot(epochs, h["train_loss"], color=COLORS["loss_train"], label="Train")
        ax.plot(epochs, h["val_loss"],   color=COLORS["loss_val"],   label="Val",
                linestyle="--")
        best_e = int(np.argmin(h["val_loss"])) + 1
        ax.axvline(best_e, color="#e3b341", linestyle=":", linewidth=1.5,
                   label=f"Best epoch={best_e}")
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Huber Loss")
        ax.legend(fontsize=9)
        ax.grid(True)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "loss_curves.png")
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[Plot] Saved -> {path}")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────
# 2. Actual vs. Predicted
# ─────────────────────────────────────────────

def plot_predictions(y_true:   np.ndarray,
                     preds:    dict,
                     dates:    pd.DatetimeIndex = None,
                     save:     bool = True) -> str:
    """
    y_true : ground-truth prices
    preds  : {model_name: predicted_prices_array}
    dates  : optional DatetimeIndex for x-axis
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    x = dates if (dates is not None and len(dates) == len(y_true)) else np.arange(len(y_true))

    ax.plot(x, y_true, color=COLORS["actual"],
            label="Actual Price", linewidth=2.5, zorder=5)

    palette = [COLORS.get(name, "#79c0ff") for name in preds.keys()]
    for (name, pred), color in zip(preds.items(), palette):
        ax.plot(x, pred, color=color, label=name,
                linestyle="--", linewidth=1.8, alpha=0.85)

    if dates is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()

    ax.set_title("Stock Price – Actual vs. Predicted (Test Set)",
                 fontweight="bold", fontsize=14)
    ax.set_xlabel("Date" if dates is not None else "Time Step")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "predictions.png")
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[Plot] Saved -> {path}")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────
# 3. Residual Distribution
# ─────────────────────────────────────────────

def plot_residuals(y_true: np.ndarray,
                   preds:  dict,
                   save:   bool = True) -> str:
    n = len(preds)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    fig.suptitle("Residual Distribution (Test Set)", fontweight="bold")

    palette = [COLORS.get(name, "#79c0ff") for name in preds.keys()]
    for ax, (name, pred), color in zip(axes[0], preds.items(), palette):
        residuals = y_true - pred
        ax.hist(residuals, bins=40, color=color, alpha=0.75, edgecolor="white")
        ax.axvline(0, color="#e3b341", linewidth=1.5, linestyle="--")
        ax.set_title(f"{name}\nMean={residuals.mean():.2f}  Std={residuals.std():.2f}",
                     fontweight="bold")
        ax.set_xlabel("Residual ($)")
        ax.set_ylabel("Frequency")
        ax.grid(True)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "residuals.png")
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[Plot] Saved -> {path}")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────
# 4. Metrics Comparison Bar Chart
# ─────────────────────────────────────────────

def plot_metrics_comparison(all_metrics: dict,
                            save: bool = True) -> str:
    """
    all_metrics : {model_name: {metric_name: value}}
    """
    metric_keys = ["RMSE", "MAE", "MAPE", "R2", "DA"]
    models = list(all_metrics.keys())
    n_metrics = len(metric_keys)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5),
                             squeeze=False)
    fig.suptitle("Model Performance Comparison", fontweight="bold")

    palette = [COLORS.get(m, "#79c0ff") for m in models]

    for ax, metric in zip(axes[0], metric_keys):
        vals   = [all_metrics[m].get(metric, 0) for m in models]
        bars   = ax.bar(models, vals,
                        color=palette[:len(models)],
                        edgecolor="white", linewidth=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005 * max(vals),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(metric, fontweight="bold")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, axis="y")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "metrics_comparison.png")
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[Plot] Saved -> {path}")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────
# 5. Price + Moving Average Chart
# ─────────────────────────────────────────────

def plot_price_with_ma(df: pd.DataFrame,
                       ticker: str = "Stock",
                       save: bool = True) -> str:
    """df must contain columns: Close, SMA_20, SMA_50, Volume."""
    fig = plt.figure(figsize=(14, 8))
    gs  = GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.08)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    ax1.plot(df.index, df["Close"], color=COLORS["actual"],
             label="Close", linewidth=1.5)
    if "SMA_20"  in df: ax1.plot(df.index, df["SMA_20"],
                                  color=COLORS["lstm"],
                                  label="SMA-20", linewidth=1.2, alpha=0.8)
    if "SMA_50"  in df: ax1.plot(df.index, df["SMA_50"],
                                  color=COLORS["transformer"],
                                  label="SMA-50", linewidth=1.2, alpha=0.8)
    if "SMA_200" in df: ax1.plot(df.index, df["SMA_200"],
                                  color=COLORS["ensemble"],
                                  label="SMA-200", linewidth=1.2, alpha=0.8)
    ax1.set_ylabel("Price (USD)")
    ax1.set_title(f"{ticker} – Historical Price with Moving Averages",
                  fontweight="bold", fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True)

    # RSI
    if "RSI_14" in df:
        ax2.plot(df.index, df["RSI_14"], color="#d2a8ff", linewidth=1.3)
        ax2.axhline(70, color="#f85149", linewidth=1, linestyle="--")
        ax2.axhline(30, color="#3fb950", linewidth=1, linestyle="--")
        ax2.set_ylabel("RSI(14)")
        ax2.set_ylim(0, 100)
        ax2.grid(True)

    # Volume
    ax3.bar(df.index, df["Volume"],
            color=COLORS["loss_train"], alpha=0.55)
    if "Vol_SMA_10" in df:
        ax3.plot(df.index, df["Vol_SMA_10"],
                 color=COLORS["loss_val"], linewidth=1.2)
    ax3.set_ylabel("Volume")
    ax3.grid(True)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    fig.autofmt_xdate()

    fig.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.08, hspace=0.08)
    path = os.path.join(PLOT_DIR, "price_ma_chart.png")
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[Plot] Saved -> {path}")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────
# 6. Feature Correlation Heatmap
# ─────────────────────────────────────────────

def plot_feature_heatmap(df: pd.DataFrame,
                         max_features: int = 20,
                         save: bool = True) -> str:
    """Plot correlation heatmap of the top `max_features` features."""
    try:
        import seaborn as sns
    except ImportError:
        print("[Plot] seaborn not installed; skipping heatmap.")
        return ""

    numeric = df.select_dtypes(include=np.number)
    # Select features with highest variance
    selected = numeric.std().nlargest(max_features).index.tolist()
    corr = numeric[selected].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0,
                linewidths=0.4, annot=True, fmt=".2f",
                annot_kws={"size": 7},
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "feature_heatmap.png")
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[Plot] Saved -> {path}")
    plt.close(fig)
    return path
