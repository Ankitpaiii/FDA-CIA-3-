"""
main.py
-------
Orchestration pipeline for the ML Stock Price Forecasting project.
Financial Data Analytics — CIA-3 Component 2

Pipeline
--------
  1. Download & feature-engineer stock data (yfinance / local cache)
  2. Build sliding-window sequences (RobustScaler; no data leakage)
  3. Train models: Ridge Baseline | BiLSTM+Attention | TFT-lite | Ensemble
  4. Evaluate: RMSE, MAE, MAPE, SMAPE, R², DA, Theil-U, MaxError, HitRate
  5. Generate 10 publication-quality plots in outputs/plots/
  6. Compute permutation feature importance
  7. Extract & visualize LSTM attention weights
  8. Write structured training report to outputs/training_report.txt

Usage
-----
  python main.py                         # defaults: AAPL, 2015-2024, 60-day window
  python main.py --ticker MSFT           # change ticker
  python main.py --model lstm            # train only LSTM
  python main.py --model all --seed 42   # fully reproducible run
  python main.py --epochs 80             # more training epochs
"""

import os
import sys
import json
import time
import argparse
import random
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(__file__))

import torch

# Utilize full CPU cores for accelerated training
num_cores = os.cpu_count() or 4
torch.set_num_threads(min(20, num_cores))

from data.data_loader import StockDataLoader
from models.baseline_model    import RidgeBaseline
from models.lstm_model        import build_lstm
from models.transformer_model import build_transformer
from models.ensemble_model    import build_ensemble
from utils.trainer    import Trainer, build_loaders
from utils.metrics    import evaluate_all
from utils.visualizer import (plot_loss_curves, plot_predictions,
                               plot_residuals, plot_metrics_comparison,
                               plot_price_with_ma, plot_feature_heatmap,
                               plot_attention_heatmap, plot_feature_importance,
                               plot_error_over_time, plot_profit_simulation)
from utils.explainability import permutation_importance, extract_attention_weights


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────

def set_seed(seed: int):
    """Set all random seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"[Main] Seed set to {seed} (reproducible run)")


# ─────────────────────────────────────────────
# CLI Arguments
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="ML Stock Price Forecasting -- BiLSTM | Transformer | Ensemble",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ticker",     type=str,   default="AAPL")
    parser.add_argument("--start",      type=str,   default="2015-01-01")
    parser.add_argument("--end",        type=str,   default="2024-01-01")
    parser.add_argument("--seq_len",    type=int,   default=60,
                        help="Lookback window in trading days (default: 60)")
    parser.add_argument("--epochs",     type=int,   default=50,
                        help="Max training epochs (default: 50)")
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--model",      type=str,   default="all",
                        choices=["baseline", "lstm", "transformer", "ensemble", "all"])
    parser.add_argument("--device",     type=str,   default="auto")
    parser.add_argument("--seed",       type=int,   default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--no_explain", action="store_true",
                        help="Skip explainability (faster runs)")
    parser.add_argument("--verbose",    action="store_true",
                        help="Print every epoch instead of every 5")
    return parser.parse_args()


# ─────────────────────────────────────────────
# Helper: train + evaluate one model
# ─────────────────────────────────────────────

def run_model(name, model, args, loaders, data_obj):
    """Train one deep-learning model, evaluate, return (history, metrics, y_true, y_pred)."""
    train_dl, val_dl, X_test, y_test_scaled = loaders

    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    trainer = Trainer(
        model          = model,
        lr             = args.lr,
        checkpoint_dir = os.path.join(PROJECT_ROOT, "outputs", "checkpoints"),
        log_dir        = os.path.join(PROJECT_ROOT, "outputs"),
        patience       = 12,
        device         = args.device,
        print_every    = 1 if args.verbose else 5
    )

    history = trainer.fit(train_dl, val_dl,
                          epochs     = args.epochs,
                          model_name = name)

    y_pred_scaled = trainer.predict(X_test)
    y_true = data_obj.inverse_transform_target(y_test_scaled)
    y_pred = data_obj.inverse_transform_target(y_pred_scaled)

    metrics = evaluate_all(y_true, y_pred, label=name.upper())
    return history, metrics, y_true, y_pred, trainer


# ─────────────────────────────────────────────
# Report Writer
# ─────────────────────────────────────────────

def write_report(args, all_metrics, feature_cols, n_features,
                 start_time: float, report_path: str):
    """Write a formatted text report of the run results."""
    elapsed = time.time() - start_time
    lines = []
    lines.append("=" * 65)
    lines.append("  ML STOCK PRICE FORECASTING — CIA-3 PROJECT REPORT")
    lines.append("  Financial Data Analytics | SRM Institute of Science & Technology")
    lines.append("=" * 65)
    lines.append(f"\n  Run Configuration")
    lines.append(f"  {'Ticker':<24}: {args.ticker}")
    lines.append(f"  {'Date Range':<24}: {args.start} to {args.end}")
    lines.append(f"  {'Lookback Window':<24}: {args.seq_len} trading days")
    lines.append(f"  {'Training Epochs':<24}: {args.epochs}")
    lines.append(f"  {'Batch Size':<24}: {args.batch_size}")
    lines.append(f"  {'Learning Rate':<24}: {args.lr}")
    lines.append(f"  {'Random Seed':<24}: {args.seed}")
    lines.append(f"  {'Total Runtime':<24}: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    lines.append(f"\n  Feature Engineering")
    lines.append(f"  {'Total Features':<24}: {n_features}")
    lines.append(f"  {'Feature Names':<24}: {', '.join(feature_cols[:10])}...")

    lines.append("\n" + "=" * 65)
    lines.append("  EVALUATION RESULTS SUMMARY")
    lines.append("=" * 65)

    header = (f"  {'Model':<22} {'RMSE':>7} {'MAE':>7} {'MAPE%':>7} "
              f"{'SMAPE%':>7} {'R2':>6} {'DA%':>6} {'TheilU':>7} {'Hit%':>7}")
    lines.append(header)
    lines.append("  " + "-" * 63)

    for mname, m in all_metrics.items():
        lines.append(
            f"  {mname:<22} "
            f"{m.get('RMSE', 0):>7.3f} "
            f"{m.get('MAE', 0):>7.3f} "
            f"{m.get('MAPE', 0):>6.2f}% "
            f"{m.get('SMAPE', 0):>6.2f}% "
            f"{m.get('R2', 0):>6.4f} "
            f"{m.get('DA', 0):>5.2f}% "
            f"{m.get('TheilU', 0):>7.4f} "
            f"{m.get('HitRate', 0):>6.2f}%"
        )

    lines.append("\n  Metric Interpretation Guide")
    lines.append("  RMSE / MAE   : prediction error in USD (lower = better)")
    lines.append("  MAPE / SMAPE : percentage error (lower = better)")
    lines.append("  R2           : 1.0 = perfect fit; 0.95+ = excellent")
    lines.append("  DA           : >55% = commercially relevant direction signal")
    lines.append("  Theil's U    : <1.0 = beats naive 'no change' prediction")
    lines.append("  Hit Rate     : % of long/short trades that were profitable")

    lines.append("\n  Output Files")
    lines.append("  outputs/plots/loss_curves.png        — Training loss history")
    lines.append("  outputs/plots/predictions.png        — Price forecast overlay")
    lines.append("  outputs/plots/residuals.png          — Error distribution + Q-Q")
    lines.append("  outputs/plots/metrics_comparison.png — All metrics bar chart")
    lines.append("  outputs/plots/price_ma_chart.png     — Technical analysis chart")
    lines.append("  outputs/plots/feature_heatmap.png    — Feature correlations")
    lines.append("  outputs/plots/attention_heatmap.png  — LSTM attention weights")
    lines.append("  outputs/plots/feature_importance.png — Permutation importance")
    lines.append("  outputs/plots/error_over_time.png    — Rolling MAE timeline")
    lines.append("  outputs/plots/profit_simulation.png  — Trading PnL simulation")
    lines.append("  outputs/metrics_summary.json         — All metrics (JSON)")
    lines.append("  outputs/training_log.csv             — Per-epoch loss history")
    lines.append("\n" + "=" * 65)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Main] Training report saved -> {report_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args       = parse_args()
    start_time = time.time()
    set_seed(args.seed)

    print("\n" + "=" * 60)
    print("  ML STOCK PRICE FORECASTING  — CIA-3")
    print(f"  Ticker : {args.ticker}  |  {args.start} -> {args.end}")
    print(f"  Seq len: {args.seq_len}  |  Max epochs: {args.epochs}")
    print(f"  Model  : {args.model}  |  Seed: {args.seed}")
    print("=" * 60 + "\n")

    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    # ── 1. Data ──────────────────────────────────────
    loader = StockDataLoader(
        ticker     = args.ticker,
        start_date = args.start,
        end_date   = args.end,
        seq_len    = args.seq_len
    )
    df = loader.build_features()
    (X_train, y_train,
     X_val,   y_val,
     X_test,  y_test,
     feature_cols) = loader.build_sequences()

    n_features = X_train.shape[2]
    print(f"[Main] Input features: {n_features}")
    print(f"[Main] First 10: {feature_cols[:10]}")

    train_dl, val_dl = build_loaders(X_train, y_train,
                                     X_val,   y_val,
                                     batch_size=args.batch_size)
    loaders = (train_dl, val_dl, X_test, y_test)

    # ── 2. Model Configs ─────────────────────────────
    lstm_cfg = {
        "input_size":    n_features,
        "hidden_size":   128,
        "num_layers":    3,
        "dropout":       0.3,
        "bidirectional": True,
        "use_attention": True,
        "output_size":   1,
    }
    transformer_cfg = {
        "input_size":  n_features,
        "d_model":     128,
        "n_heads":     8,
        "n_layers":    4,
        "dim_ff":      256,
        "dropout":     0.1,
        "seq_len":     args.seq_len,
        "output_size": 1,
    }
    ensemble_cfg = {
        "input_size":   n_features,
        "seq_len":      args.seq_len,
        "freeze_base":  False,
    }

    # ── 3. Train & Evaluate ──────────────────────────
    histories    = {}
    all_metrics  = {}
    all_preds    = {}
    trained_models = {}
    y_true_ref   = None

    models_to_run = (["baseline", "lstm", "transformer", "ensemble"]
                     if args.model == "all"
                     else [args.model])

    for mname in models_to_run:
        print(f"\n{'─'*60}")
        print(f"  TRAINING: {mname.upper()}")
        print(f"{'─'*60}")

        if mname == "baseline":
            b_model = RidgeBaseline(alpha=1.0)
            b_model.fit(X_train, y_train)
            y_pred_scaled = b_model.predict(X_test)
            y_true = loader.inverse_transform_target(y_test)
            y_pred = loader.inverse_transform_target(y_pred_scaled)
            metrics = evaluate_all(y_true, y_pred, label="Ridge Baseline")
            all_metrics[mname] = metrics
            all_preds[mname]   = y_pred
            y_true_ref         = y_true
            continue

        if   mname == "lstm":        model = build_lstm(lstm_cfg)
        elif mname == "transformer": model = build_transformer(transformer_cfg)
        else:                        model = build_ensemble(ensemble_cfg)

        history, metrics, y_true, y_pred, trainer = run_model(
            mname, model, args, loaders, loader
        )

        histories[mname]      = history
        all_metrics[mname]    = metrics
        all_preds[mname]      = y_pred
        trained_models[mname] = (model, trainer)
        y_true_ref            = y_true

    # ── 4. Save Metrics JSON ─────────────────────────
    out_dir      = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "metrics_summary.json")

    # Round for cleaner JSON output
    def _round_metrics(d):
        return {k: round(float(v), 4) for k, v in d.items()}
    metrics_export = {k: _round_metrics(v) for k, v in all_metrics.items()}

    with open(metrics_path, "w") as f:
        json.dump(metrics_export, f, indent=2)
    print(f"\n[Main] Metrics saved -> {metrics_path}")

    # ── 5. Plots ─────────────────────────────────────
    print("\n[Main] Generating plots ...")

    # Align date index with test set
    feature_df  = loader.feature_df
    seq_len     = args.seq_len
    n_total     = len(feature_df) - seq_len
    n_test      = int(n_total * 0.2)
    test_start  = n_total - n_test
    test_dates  = feature_df.index[seq_len + test_start:
                                   seq_len + test_start + len(y_true_ref)]

    dl_histories = {k: v for k, v in histories.items() if k != "baseline"}
    if dl_histories:
        plot_loss_curves(dl_histories)

    plot_predictions(y_true_ref, all_preds, dates=test_dates)
    plot_residuals(y_true_ref, all_preds)
    plot_metrics_comparison(all_metrics)
    plot_price_with_ma(df, ticker=args.ticker)
    plot_feature_heatmap(df)
    plot_error_over_time(y_true_ref, all_preds, dates=test_dates)
    plot_profit_simulation(y_true_ref, all_preds, dates=test_dates)

    # ── 6. Explainability ────────────────────────────
    if not args.no_explain and "lstm" in trained_models:
        print("\n[Main] Computing explainability ...")
        lstm_model, lstm_trainer = trained_models["lstm"]

        # a) Attention weights from LSTM
        attn_weights = extract_attention_weights(
            lstm_model, X_test,
            device=str(lstm_trainer.device)
        )
        if attn_weights is not None:
            plot_attention_heatmap(attn_weights,
                                   dates=test_dates,
                                   seq_len=args.seq_len)

        # b) Permutation feature importance (on LSTM — fastest DL model)
        y_true_scale = loader.target_scaler.transform(
            y_true_ref.reshape(-1, 1)).flatten()
        importances = permutation_importance(
            lstm_model, X_test,
            y_true        = y_true_scale,
            feature_names = feature_cols,
            device        = str(lstm_trainer.device),
            n_repeats     = 3
        )

        # Save importances JSON
        imp_path = os.path.join(out_dir, "feature_importances.json")
        with open(imp_path, "w") as f:
            json.dump({k: round(v, 6) for k, v in importances.items()}, f, indent=2)
        print(f"[Main] Feature importances saved -> {imp_path}")

        plot_feature_importance(importances, top_n=20)

    # ── 7. Final Summary ─────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<22} {'RMSE':>7} {'MAE':>7} {'MAPE%':>7} "
          f"{'R2':>6} {'DA%':>6} {'Hit%':>6}")
    print("  " + "-" * 58)
    for mname, m in all_metrics.items():
        print(f"  {mname:<22} "
              f"{m.get('RMSE', 0):>7.4f} "
              f"{m.get('MAE', 0):>7.4f} "
              f"{m.get('MAPE', 0):>6.2f}% "
              f"{m.get('R2', 0):>6.4f} "
              f"{m.get('DA', 0):>5.2f}% "
              f"{m.get('HitRate', 0):>5.2f}%")
    print("=" * 60)

    # ── 8. Write Text Report ─────────────────────────
    report_path = os.path.join(out_dir, "training_report.txt")
    write_report(args, all_metrics, feature_cols, n_features,
                 start_time, report_path)

    print(f"\n[Main] All outputs saved to {out_dir}/")
    print(f"[Main] Total runtime: {(time.time() - start_time)/60:.1f} min")
    print("[Main] Done.\n")


if __name__ == "__main__":
    main()
