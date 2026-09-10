"""
main.py
-------
Orchestration script for the Stock Price Forecasting project.

Pipeline
--------
  1. Download & feature-engineer stock data (yfinance)
  2. Build sliding-window sequences
  3. Train three models: BiLSTM-Attention, TFT-lite Transformer, Ensemble
  4. Evaluate on held-out test set (RMSE, MAE, MAPE, R², DA, Theil-U)
  5. Generate and save all plots to outputs/plots/

Usage
-----
  python main.py                        (defaults: AAPL, 2015-2024, 60-day window)
  python main.py --ticker MSFT
  python main.py --ticker TSLA --epochs 80 --seq_len 90
  python main.py --model lstm           (train only LSTM)
  python main.py --model transformer
  python main.py --model ensemble
  python main.py --model all            (train all three — default)
"""

import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import json
import argparse
import numpy as np
import pandas as pd

# -- Ensure project root is in path -------------
sys.path.insert(0, os.path.dirname(__file__))

from data.data_loader import StockDataLoader
from models.baseline_model    import RidgeBaseline
from models.lstm_model        import build_lstm
from models.transformer_model import build_transformer
from models.ensemble_model    import build_ensemble
from utils.trainer    import Trainer, build_loaders
from utils.metrics    import evaluate_all
from utils.visualizer import (plot_loss_curves, plot_predictions,
                               plot_residuals, plot_metrics_comparison,
                               plot_price_with_ma, plot_feature_heatmap)


# ─────────────────────────────────────────────
# CLI Arguments
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="ML Stock Price Forecasting – LSTM | Transformer | Ensemble"
    )
    parser.add_argument("--ticker",     type=str,   default="AAPL",
                        help="Yahoo Finance ticker (default: AAPL)")
    parser.add_argument("--start",      type=str,   default="2015-01-01",
                        help="Start date YYYY-MM-DD")
    parser.add_argument("--end",        type=str,   default="2024-01-01",
                        help="End date YYYY-MM-DD")
    parser.add_argument("--seq_len",    type=int,   default=60,
                        help="Lookback window length (default: 60)")
    parser.add_argument("--epochs",     type=int,   default=25,
                        help="Maximum training epochs (default: 25)")
    parser.add_argument("--batch_size", type=int,   default=128,
                        help="Batch size (default: 128)")
    parser.add_argument("--lr",         type=float, default=1e-3,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--model",      type=str,   default="all",
                        choices=["baseline", "lstm", "transformer", "ensemble", "all"],
                        help="Which model(s) to train")
    parser.add_argument("--device",     type=str,   default="auto",
                        help="Device: 'cuda', 'cpu', or 'auto'")
    return parser.parse_args()


# ─────────────────────────────────────────────
# Helper: train + evaluate one model
# ─────────────────────────────────────────────

def run_model(name, model, args, loaders, data_obj, n_features):
    """Train one model, evaluate it, and return (history, metrics, preds)."""
    train_dl, val_dl, X_test, y_test_scaled = loaders

    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    trainer = Trainer(
        model          = model,
        lr             = args.lr,
        checkpoint_dir = os.path.join(PROJECT_ROOT, "outputs", "checkpoints"),
        device         = args.device
    )

    history = trainer.fit(train_dl, val_dl,
                          epochs     = args.epochs,
                          model_name = name)

    # Predictions on test set (scaled)
    y_pred_scaled = trainer.predict(X_test)

    # Inverse-transform to real price
    y_true = data_obj.inverse_transform_target(y_test_scaled)
    y_pred = data_obj.inverse_transform_target(y_pred_scaled)

    metrics = evaluate_all(y_true, y_pred, label=name)
    return history, metrics, y_true, y_pred


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    print("\n" + "=" * 55)
    print("  ML STOCK PRICE FORECASTING")
    print(f"  Ticker : {args.ticker}  |  {args.start} -> {args.end}")
    print(f"  Seq len: {args.seq_len}  |  Max epochs: {args.epochs}")
    print(f"  Model  : {args.model}")
    print("=" * 55 + "\n")

    # ── 1. Data ───────────────────────────────
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
    print(f"[Main] Input feature count: {n_features}")
    print(f"[Main] Feature columns ({n_features}): {feature_cols[:5]}...")

    train_dl, val_dl = build_loaders(X_train, y_train,
                                     X_val,   y_val,
                                     batch_size=args.batch_size)

    loaders = (train_dl, val_dl, X_test, y_test)

    # ── 2. Base configs ───────────────────────
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

    # ── 3. Train ──────────────────────────────
    histories   = {}
    all_metrics = {}
    all_preds   = {}
    y_true_ref  = None

    models_to_run = (["baseline", "lstm", "transformer", "ensemble"]
                     if args.model == "all"
                     else [args.model])

    for mname in models_to_run:
        print(f"\n{'-'*55}")
        print(f"  Training: {mname.upper()}")
        print(f"{'-'*55}")

        if mname == "baseline":
            b_model = RidgeBaseline(alpha=1.0)
            b_model.fit(X_train, y_train)
            y_pred_scaled = b_model.predict(X_test)
            y_true = loader.inverse_transform_target(y_test)
            y_pred = loader.inverse_transform_target(y_pred_scaled)
            metrics = evaluate_all(y_true, y_pred, label="Baseline (Ridge)")
            all_metrics[mname] = metrics
            all_preds[mname]   = y_pred
            y_true_ref         = y_true
            continue

        if mname == "lstm":
            model = build_lstm(lstm_cfg)
        elif mname == "transformer":
            model = build_transformer(transformer_cfg)
        else:
            model = build_ensemble(ensemble_cfg)

        history, metrics, y_true, y_pred = run_model(
            mname, model, args, loaders, loader, n_features
        )

        histories[mname]   = history
        all_metrics[mname] = metrics
        all_preds[mname]   = y_pred
        y_true_ref         = y_true

    # ── 4. Save metrics JSON ──────────────────
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "metrics_summary.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[Main] Metrics saved -> {metrics_path}")

    # ── 5. Plots ──────────────────────────────
    print("\n[Main] Generating plots ...")

    dl_histories = {k: v for k, v in histories.items() if k != "baseline"}
    if dl_histories:
        plot_loss_curves(dl_histories)

    # Align date index with test set
    feature_df = loader.feature_df
    seq_len    = args.seq_len
    n_total    = len(feature_df) - seq_len
    n_test     = int(n_total * 0.2)
    n_val      = int(n_total * 0.1)
    test_start = n_total - n_test
    test_dates = feature_df.index[seq_len + test_start: seq_len + test_start + len(y_true_ref)]

    plot_predictions(y_true_ref, all_preds, dates=test_dates)
    plot_residuals(y_true_ref, all_preds)
    plot_metrics_comparison(all_metrics)
    plot_price_with_ma(df, ticker=args.ticker)
    plot_feature_heatmap(df)

    # ── 6. Summary ────────────────────────────
    print("\n" + "=" * 55)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 55)
    print(f"  {'Model':<16} {'RMSE':>8} {'MAE':>8} {'MAPE':>8} {'R2':>7} {'DA%':>7}")
    print("  " + "-" * 52)
    for mname, m in all_metrics.items():
        print(f"  {mname:<16} "
              f"{m['RMSE']:>8.4f} "
              f"{m['MAE']:>8.4f} "
              f"{m['MAPE']:>7.2f}% "
              f"{m['R2']:>7.4f} "
              f"{m['DA']:>6.2f}%")
    print("=" * 55)
    print("\n[Main] All outputs saved to ./outputs/")
    print("[Main] Done.\n")


if __name__ == "__main__":
    main()
