"""
visualizer.py
-------------
Publication-quality plotting for ML stock price forecasting results.
Dark-theme GitHub-inspired colour palette. 300 DPI output.

Plots Generated
---------------
  1.  loss_curves.png        — Training & Validation loss curves (all models)
  2.  predictions.png        — Actual vs. Predicted price overlay (test set)
  3.  residuals.png          — Residual histogram + Q-Q normality test (2 rows)
  4.  metrics_comparison.png — Grouped bar chart across all metrics & models
  5.  price_ma_chart.png     — Price + Bollinger Bands + RSI + Volume chart
  6.  feature_heatmap.png    — Feature correlation heatmap (top features)
  7.  attention_heatmap.png  — LSTM attention weights over time (interpretability)
  8.  feature_importance.png — Permutation feature importance
  9.  error_over_time.png    — Rolling MAE over test period (temporal error analysis)
  10. profit_simulation.png  — Cumulative PnL from trading on model signals
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
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Global Style
# ─────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#e6edf3",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#e6edf3",
    "grid.color":        "#21262d",
    "grid.linewidth":    0.7,
    "grid.alpha":        0.8,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "legend.fontsize":   9,
    "lines.linewidth":   2.0,
    "figure.titlesize":  15,
    "figure.dpi":        100,
})

# Colour palette (GitHub dark + financial chart colours)
COLORS = {
    "actual":      "#58a6ff",   # blue
    "baseline":    "#8b949e",   # grey
    "lstm":        "#3fb950",   # green
    "transformer": "#d2a8ff",   # purple
    "ensemble":    "#ffa657",   # orange
    "loss_train":  "#58a6ff",
    "loss_val":    "#f85149",   # red
    "profit":      "#3fb950",
    "loss_pnl":    "#f85149",
    "neutral":     "#e3b341",   # yellow
    "band":        "#30363d",   # dark grey for fills
}

MODEL_LABELS = {
    "baseline":    "Ridge Baseline",
    "lstm":        "BiLSTM + Attention",
    "transformer": "TFT-lite Transformer",
    "ensemble":    "Stacking Ensemble",
}

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PLOT_DIR = os.path.join(BASE_DIR, "outputs", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

DPI = 150  # output resolution


def _save(fig, filename: str) -> str:
    """Save figure and close it. Returns path."""
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Plot] Saved -> {path}")
    return path


def _model_color(name: str) -> str:
    return COLORS.get(name, "#79c0ff")


def _model_label(name: str) -> str:
    return MODEL_LABELS.get(name, name)


# ─────────────────────────────────────────────
# 1. Training & Validation Loss Curves
# ─────────────────────────────────────────────

def plot_loss_curves(histories: dict, save: bool = True) -> str:
    """
    histories : {model_name: {"train_loss": [...], "val_loss": [...]}}
    """
    n = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    fig.suptitle("Training & Validation Loss Curves (Huber Loss)",
                 fontweight="bold", y=1.02)

    for ax, (name, h) in zip(axes[0], histories.items()):
        epochs = range(1, len(h["train_loss"]) + 1)
        tr  = h["train_loss"]
        val = h["val_loss"]

        ax.plot(epochs, tr,  color=COLORS["loss_train"], label="Train Loss",
                linewidth=2.0)
        ax.plot(epochs, val, color=COLORS["loss_val"],   label="Val Loss",
                linestyle="--", linewidth=2.0)

        # Shaded area between train and val
        ax.fill_between(epochs, tr, val,
                        alpha=0.08, color=COLORS["neutral"])

        best_e = int(np.argmin(val)) + 1
        ax.axvline(best_e, color=COLORS["neutral"], linestyle=":",
                   linewidth=1.8, label=f"Best epoch {best_e}")
        ax.annotate(f"Best: {val[best_e-1]:.5f}",
                    xy=(best_e, val[best_e-1]),
                    xytext=(best_e + max(len(epochs) // 10, 1), val[best_e-1]),
                    arrowprops=dict(arrowstyle="->", color=COLORS["neutral"]),
                    color=COLORS["neutral"], fontsize=8)

        ax.set_title(_model_label(name), fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Huber Loss")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    return _save(fig, "loss_curves.png") if save else ""


# ─────────────────────────────────────────────
# 2. Actual vs. Predicted (with shaded error band)
# ─────────────────────────────────────────────

def plot_predictions(y_true: np.ndarray,
                     preds:  dict,
                     dates:  pd.DatetimeIndex = None,
                     save:   bool = True) -> str:
    """
    Plot actual vs. predicted prices for all models.
    Includes a shaded ±RMSE error band around the best model's predictions.
    """
    fig, ax = plt.subplots(figsize=(15, 6))
    x = (dates if (dates is not None and len(dates) == len(y_true))
         else np.arange(len(y_true)))

    ax.plot(x, y_true, color=COLORS["actual"],
            label="Actual Price", linewidth=2.5, zorder=5)

    # Find best model by RMSE
    best_name = None
    best_rmse = float("inf")
    for name, pred in preds.items():
        _r = float(np.sqrt(np.mean((y_true - pred) ** 2)))
        if _r < best_rmse:
            best_rmse = _r
            best_name = name

    for name, pred in preds.items():
        color = _model_color(name)
        label = _model_label(name)
        ax.plot(x, pred, color=color, label=label,
                linestyle="--", linewidth=1.8, alpha=0.85)
        if name == best_name:
            # Shaded ±RMSE confidence band for the best model
            ax.fill_between(x,
                            pred - best_rmse, pred + best_rmse,
                            alpha=0.12, color=color,
                            label=f"{label} ±RMSE band")

    if dates is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()

    # Annotate best RMSE
    ax.text(0.02, 0.97,
            f"Best model: {_model_label(best_name)}  RMSE=${best_rmse:.2f}",
            transform=ax.transAxes, fontsize=9, va="top",
            color=_model_color(best_name),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d1117", alpha=0.8))

    ax.set_title("Stock Price — Actual vs. Predicted (Test Set)",
                 fontweight="bold", fontsize=14)
    ax.set_xlabel("Date" if dates is not None else "Time Step")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="upper left")
    ax.grid(True)
    plt.tight_layout()
    return _save(fig, "predictions.png") if save else ""


# ─────────────────────────────────────────────
# 3. Residuals: Histogram + Q-Q Plot
# ─────────────────────────────────────────────

def plot_residuals(y_true: np.ndarray,
                   preds:  dict,
                   save:   bool = True) -> str:
    """
    2-row grid: top row = residual histograms, bottom row = Q-Q plots.
    Q-Q plot tests normality of residuals (important for model diagnostics).
    """
    from scipy import stats as scipy_stats
    n   = len(preds)
    fig, axes = plt.subplots(2, n, figsize=(5.5 * n, 9), squeeze=False)
    fig.suptitle("Residual Analysis — Test Set",
                 fontweight="bold", fontsize=14, y=1.01)

    palette = [_model_color(name) for name in preds.keys()]

    for col, (name, pred), color in zip(range(n), preds.items(), palette):
        residuals = y_true - pred
        mean_r    = residuals.mean()
        std_r     = residuals.std()
        label     = _model_label(name)

        # ── Row 1: Histogram
        ax_hist = axes[0][col]
        ax_hist.hist(residuals, bins=40, color=color, alpha=0.75,
                     edgecolor="#30363d", linewidth=0.5)
        ax_hist.axvline(0,      color=COLORS["neutral"], linewidth=1.8,
                        linestyle="--", label="Zero error")
        ax_hist.axvline(mean_r, color=COLORS["loss_val"], linewidth=1.5,
                        linestyle=":", label=f"Mean={mean_r:.2f}")
        ax_hist.set_title(f"{label}\nMean={mean_r:.2f}  Std={std_r:.2f}",
                          fontweight="bold")
        ax_hist.set_xlabel("Residual (USD)")
        ax_hist.set_ylabel("Frequency")
        ax_hist.legend(fontsize=8)
        ax_hist.grid(True)

        # ── Row 2: Q-Q Plot
        ax_qq = axes[1][col]
        (osm, osr), (slope, intercept, r) = scipy_stats.probplot(
            residuals, dist="norm")
        ax_qq.scatter(osm, osr, color=color, alpha=0.4, s=8)
        qq_line = np.array([min(osm), max(osm)])
        ax_qq.plot(qq_line, slope * qq_line + intercept,
                   color=COLORS["neutral"], linewidth=1.8,
                   label=f"Normal line (r={r:.3f})")
        ax_qq.set_title(f"Q-Q Plot — {label}", fontweight="bold")
        ax_qq.set_xlabel("Theoretical Quantiles")
        ax_qq.set_ylabel("Sample Quantiles")
        ax_qq.legend(fontsize=8)
        ax_qq.grid(True)

    plt.tight_layout()
    return _save(fig, "residuals.png") if save else ""


# ─────────────────────────────────────────────
# 4. Metrics Comparison Bar Chart
# ─────────────────────────────────────────────

def plot_metrics_comparison(all_metrics: dict,
                            save: bool = True) -> str:
    """
    Grouped bar chart comparing all evaluation metrics across models.
    Highlights the winning model for each metric.
    """
    metric_keys   = ["RMSE", "MAE", "MAPE", "SMAPE", "R2", "DA", "TheilU"]
    metric_labels = ["RMSE ($)", "MAE ($)", "MAPE (%)", "SMAPE (%)",
                     "R² Score", "Dir. Accuracy (%)", "Theil's U"]
    # For R² and DA: higher is better; for others: lower is better
    higher_better = {"R2", "DA"}

    models  = list(all_metrics.keys())
    palette = [_model_color(m) for m in models]

    fig, axes = plt.subplots(1, len(metric_keys),
                             figsize=(4.2 * len(metric_keys), 6),
                             squeeze=False)
    fig.suptitle("Model Performance Comparison — All Metrics",
                 fontweight="bold", fontsize=14)

    for ax, key, label in zip(axes[0], metric_keys, metric_labels):
        vals = [all_metrics[m].get(key, 0) for m in models]

        # Find winner index
        if key in higher_better:
            winner_idx = int(np.argmax(vals))
        else:
            winner_idx = int(np.argmin(vals))

        bars = ax.bar(range(len(models)), vals,
                      color=palette[:len(models)],
                      edgecolor="#30363d", linewidth=0.6,
                      width=0.6)

        # Highlight winner bar with a gold border
        bars[winner_idx].set_edgecolor(COLORS["neutral"])
        bars[winner_idx].set_linewidth(2.5)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005 * max(abs(v) for v in vals),
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=8, color="#e6edf3")

        ax.set_title(label, fontweight="bold")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(
            [MODEL_LABELS.get(m, m) for m in models],
            rotation=20, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.5)
        ax.set_ylabel(label)

    plt.tight_layout()
    return _save(fig, "metrics_comparison.png") if save else ""


# ─────────────────────────────────────────────
# 5. Price + Bollinger Bands + RSI + Volume
# ─────────────────────────────────────────────

def plot_price_with_ma(df: pd.DataFrame,
                       ticker: str = "Stock",
                       save: bool = True) -> str:
    """
    4-panel chart: Price+BB+MAs | RSI | Volume | Volatility.
    df must contain columns: Close, SMA_20, SMA_50, RSI_14, Volume.
    """
    fig = plt.figure(figsize=(15, 10))
    gs  = GridSpec(4, 1, height_ratios=[4, 1.5, 1.5, 1.2], hspace=0.06)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)

    # ── Panel 1: Price + MAs + Bollinger Bands ─────
    ax1.plot(df.index, df["Close"],  color=COLORS["actual"],
             label="Close Price",   linewidth=1.8, zorder=5)
    if "SMA_20"  in df:
        ax1.plot(df.index, df["SMA_20"], color=COLORS["lstm"],
                 label="SMA-20", linewidth=1.1, alpha=0.85)
    if "SMA_50"  in df:
        ax1.plot(df.index, df["SMA_50"], color=COLORS["transformer"],
                 label="SMA-50", linewidth=1.1, alpha=0.85)
    if "SMA_200" in df:
        ax1.plot(df.index, df["SMA_200"], color=COLORS["ensemble"],
                 label="SMA-200", linewidth=1.1, alpha=0.85)
    # Bollinger Bands
    if "BB_Upper" in df and "BB_Lower" in df:
        ax1.plot(df.index, df["BB_Upper"], color="#8b949e",
                 linewidth=0.9, linestyle=":", label="BB Upper/Lower")
        ax1.plot(df.index, df["BB_Lower"], color="#8b949e",
                 linewidth=0.9, linestyle=":")
        ax1.fill_between(df.index, df["BB_Lower"], df["BB_Upper"],
                         alpha=0.06, color="#8b949e", label="BB Band")
    ax1.set_ylabel("Price (USD)")
    ax1.set_title(f"{ticker} — Historical Price Analysis (2015–2024)",
                  fontweight="bold", fontsize=14)
    ax1.legend(loc="upper left", ncol=3, fontsize=8)
    ax1.grid(True)

    # ── Panel 2: RSI ────────────────────────────
    if "RSI_14" in df:
        ax2.plot(df.index, df["RSI_14"], color="#d2a8ff", linewidth=1.3)
        ax2.fill_between(df.index, 70, 100,
                         where=(df["RSI_14"] > 70),
                         alpha=0.18, color=COLORS["loss_val"], label="Overbought")
        ax2.fill_between(df.index, 0, 30,
                         where=(df["RSI_14"] < 30),
                         alpha=0.18, color=COLORS["profit"], label="Oversold")
        ax2.axhline(70, color=COLORS["loss_val"],  linewidth=0.9, linestyle="--")
        ax2.axhline(30, color=COLORS["profit"],    linewidth=0.9, linestyle="--")
        ax2.axhline(50, color=COLORS["band"],      linewidth=0.6, linestyle="-")
        ax2.set_ylabel("RSI(14)")
        ax2.set_ylim(0, 100)
        ax2.legend(loc="upper left", fontsize=7)
        ax2.grid(True)

    # ── Panel 3: Volume ─────────────────────────
    ax3.bar(df.index, df["Volume"],
            color=COLORS["loss_train"], alpha=0.45, width=2)
    if "Vol_SMA_10" in df:
        ax3.plot(df.index, df["Vol_SMA_10"],
                 color=COLORS["loss_val"], linewidth=1.2, label="Vol SMA-10")
    ax3.set_ylabel("Volume")
    ax3.legend(loc="upper left", fontsize=7)
    ax3.grid(True)

    # ── Panel 4: 10-day Volatility ─────────────
    if "Volatility_10d" in df:
        ax4.plot(df.index, df["Volatility_10d"] * 100,
                 color=COLORS["neutral"], linewidth=1.2)
        ax4.fill_between(df.index, 0, df["Volatility_10d"] * 100,
                         alpha=0.15, color=COLORS["neutral"])
        ax4.set_ylabel("10-day Vol (%)")
        ax4.grid(True)

    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    fig.autofmt_xdate()
    fig.subplots_adjust(left=0.07, right=0.97, top=0.95, bottom=0.05)

    return _save(fig, "price_ma_chart.png") if save else ""


# ─────────────────────────────────────────────
# 6. Feature Correlation Heatmap
# ─────────────────────────────────────────────

def plot_feature_heatmap(df: pd.DataFrame,
                         max_features: int = 22,
                         save: bool = True) -> str:
    """Correlation heatmap of top `max_features` by variance."""
    try:
        import seaborn as sns
    except ImportError:
        print("[Plot] seaborn not installed; skipping heatmap.")
        return ""

    numeric  = df.select_dtypes(include=np.number)
    selected = numeric.std().nlargest(max_features).index.tolist()
    corr     = numeric[selected].corr()

    fig, ax = plt.subplots(figsize=(15, 12))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr, ax=ax, cmap=cmap, center=0,
                linewidths=0.3, annot=True, fmt=".2f",
                annot_kws={"size": 7},
                cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1)
    ax.set_title("Feature Correlation Heatmap (Top Features by Variance)",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    return _save(fig, "feature_heatmap.png") if save else ""


# ─────────────────────────────────────────────
# 7. Attention Weight Heatmap (LSTM Interpretability)
# ─────────────────────────────────────────────

def plot_attention_heatmap(attn_weights: np.ndarray,
                           dates: pd.DatetimeIndex = None,
                           n_samples: int = 30,
                           seq_len:   int = 60,
                           save: bool = True) -> str:
    """
    Visualize LSTM attention weights over the input window.

    attn_weights : (N, T) — attention scores for N test samples
    n_samples    : number of test samples to display (rows of heatmap)

    Interpretation: bright cells = timesteps the model focuses on
    when predicting the next price for that particular test day.
    """
    if attn_weights is None or len(attn_weights) == 0:
        print("[Plot] No attention weights available; skipping.")
        return ""

    # Select n_samples evenly spaced
    total  = len(attn_weights)
    idx    = np.linspace(0, total - 1, min(n_samples, total), dtype=int)
    data   = attn_weights[idx]        # (n_samples, T)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                             gridspec_kw={"width_ratios": [3, 1]})
    fig.suptitle("LSTM Attention Weight Heatmap — Test Set",
                 fontweight="bold", fontsize=14)

    # ── Left: heatmap ───────────────────────────
    ax = axes[0]
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest", vmin=0)
    fig.colorbar(im, ax=ax, label="Attention Weight")
    ax.set_xlabel(f"Lookback Window (t-{seq_len} to t-1)")
    ax.set_ylabel("Test Sample (evenly spaced)")

    # Label x-axis as relative days
    xticks = [0, seq_len // 4, seq_len // 2, 3 * seq_len // 4, seq_len - 1]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"t-{seq_len - x}" for x in xticks])
    ax.set_title("Per-Sample Attention Map", fontweight="bold")

    # ── Right: mean attention profile ───────────
    ax2 = axes[1]
    mean_attn = data.mean(axis=0)   # (T,)
    ax2.barh(range(seq_len), mean_attn,
             color=COLORS["transformer"], alpha=0.8)
    ax2.set_xlabel("Mean Attention Weight")
    ax2.set_ylabel("Lookback Day (0 = oldest)")
    ax2.set_title("Average Attention\nAcross Test Samples", fontweight="bold")
    ax2.axvline(1 / seq_len, color=COLORS["neutral"],
                linestyle="--", linewidth=1.2, label="Uniform baseline")
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="x")
    ax2.invert_yaxis()

    plt.tight_layout()
    return _save(fig, "attention_heatmap.png") if save else ""


# ─────────────────────────────────────────────
# 8. Feature Importance (Permutation-based)
# ─────────────────────────────────────────────

def plot_feature_importance(importances: dict,
                            top_n: int = 20,
                            save: bool = True) -> str:
    """
    importances : {feature_name: importance_score}
    Horizontal bar chart sorted by importance (descending).
    """
    if not importances:
        print("[Plot] No feature importances available; skipping.")
        return ""

    # Sort and take top_n
    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    sorted_items = sorted_items[:top_n]
    names  = [item[0] for item in sorted_items]
    scores = [item[1] for item in sorted_items]

    # Colour by score magnitude
    max_s  = max(scores) + 1e-9
    colors = [plt.cm.YlOrRd(s / max_s) for s in scores]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    bars = ax.barh(range(len(names)), scores, color=colors,
                   edgecolor="#30363d", linewidth=0.4)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.001 * max_s, bar.get_y() + bar.get_height() / 2,
                f"{score:.5f}", va="center", fontsize=8)

    ax.set_xlabel("Permutation Importance (RMSE increase when feature is shuffled)")
    ax.set_title(f"Feature Importance — Top {top_n} Features\n"
                 "(Higher = more important for model accuracy)",
                 fontweight="bold", fontsize=13)
    ax.grid(True, axis="x", alpha=0.5)
    ax.axvline(0, color="#30363d", linewidth=1.0)
    plt.tight_layout()
    return _save(fig, "feature_importance.png") if save else ""


# ─────────────────────────────────────────────
# 9. Prediction Error Over Time
# ─────────────────────────────────────────────

def plot_error_over_time(y_true: np.ndarray,
                         preds:  dict,
                         dates:  pd.DatetimeIndex = None,
                         window: int = 20,
                         save:   bool = True) -> str:
    """
    Rolling MAE (window=20 days) over the test period for each model.
    Shows where in time each model performs best / worst.
    Useful for identifying market regimes where models struggle.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    x = (dates[window - 1:] if (dates is not None and len(dates) >= window)
         else np.arange(len(y_true) - window + 1))

    for name, pred in preds.items():
        abs_errors    = np.abs(y_true - pred)
        rolling_mae   = np.convolve(abs_errors, np.ones(window) / window,
                                    mode="valid")
        color = _model_color(name)
        label = _model_label(name)
        ax.plot(x, rolling_mae, color=color, label=label, linewidth=1.8)
        ax.fill_between(x, 0, rolling_mae, alpha=0.06, color=color)

    if dates is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()

    ax.set_title(f"Rolling {window}-Day MAE Over Test Period\n"
                 "(Lower = better prediction in that period)",
                 fontweight="bold", fontsize=13)
    ax.set_xlabel("Date" if dates is not None else "Test Step")
    ax.set_ylabel(f"{window}-Day Rolling MAE (USD)")
    ax.legend(loc="upper left")
    ax.grid(True)
    plt.tight_layout()
    return _save(fig, "error_over_time.png") if save else ""


# ─────────────────────────────────────────────
# 10. Profit Simulation (Cumulative PnL)
# ─────────────────────────────────────────────

def plot_profit_simulation(y_true: np.ndarray,
                           preds:  dict,
                           dates:  pd.DatetimeIndex = None,
                           save:   bool = True) -> str:
    """
    Simulate a simple long/short trading strategy for each model.

    Strategy:
      - If model predicts price UP tomorrow → go long (buy)
      - If model predicts price DOWN tomorrow → go short (sell)
      - PnL = daily actual price change * our position

    Returns cumulative PnL expressed as total return (%).
    A buy-and-hold benchmark is also plotted for comparison.
    """
    from utils.metrics import simulate_pnl

    fig, ax = plt.subplots(figsize=(14, 6))
    x = (dates[1:] if (dates is not None and len(dates) > 1)
         else np.arange(len(y_true) - 1))

    for name, pred in preds.items():
        cum_pnl = simulate_pnl(y_true, pred) * 100   # in %
        color   = _model_color(name)
        label   = _model_label(name)
        final   = cum_pnl[-1]
        ax.plot(x, cum_pnl, color=color, linewidth=2.0,
                label=f"{label}  (final: {final:+.1f}%)")

    # Buy-and-hold benchmark (just hold the stock)
    bah_returns = np.diff(y_true) / (y_true[:-1] + 1e-9)
    bah_cum     = np.cumsum(bah_returns) * 100
    ax.plot(x, bah_cum, color=COLORS["neutral"], linewidth=1.8,
            linestyle="--", label=f"Buy & Hold  (final: {bah_cum[-1]:+.1f}%)")

    ax.axhline(0, color="#30363d", linewidth=1.0)
    ax.fill_between(x, 0, bah_cum,
                    where=(bah_cum > 0), alpha=0.07, color=COLORS["profit"])
    ax.fill_between(x, 0, bah_cum,
                    where=(bah_cum < 0), alpha=0.07, color=COLORS["loss_pnl"])

    if dates is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()

    ax.set_title("Simulated Cumulative Trading PnL (%)\n"
                 "Long/Short strategy based on model direction signals",
                 fontweight="bold", fontsize=13)
    ax.set_xlabel("Date" if dates is not None else "Test Day")
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True)
    plt.tight_layout()
    return _save(fig, "profit_simulation.png") if save else ""
