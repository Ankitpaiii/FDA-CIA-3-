"""
hyperparameter_tuning.py
------------------------
Grid-search / random-search style hyperparameter tuning
for the LSTM and Transformer models using k-fold cross-validation
on the training split.

Usage
-----
  python hyperparameter_tuning.py --ticker AAPL --model lstm
  python hyperparameter_tuning.py --ticker AAPL --model transformer
"""

import os
import sys
import json
import argparse
import numpy as np
from itertools import product

sys.path.insert(0, os.path.dirname(__file__))

from data.data_loader         import StockDataLoader
from models.lstm_model        import build_lstm
from models.transformer_model import build_transformer
from utils.trainer            import Trainer, build_loaders
from utils.metrics            import rmse


# ─────────────────────────────────────────────
# Parameter Grids
# ─────────────────────────────────────────────

LSTM_GRID = {
    "hidden_size":   [64, 128],
    "num_layers":    [2, 3],
    "dropout":       [0.2, 0.3],
    "bidirectional": [True],
    "use_attention": [True],
}

TRANSFORMER_GRID = {
    "d_model":  [64, 128],
    "n_heads":  [4, 8],
    "n_layers": [2, 4],
    "dropout":  [0.1, 0.2],
    "dim_ff":   [128, 256],
}


def grid_configs(grid: dict) -> list:
    keys   = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in product(*values)]


# ─────────────────────────────────────────────
# Single-run Evaluator
# ─────────────────────────────────────────────

def evaluate_config(cfg_override: dict,
                    model_type:   str,
                    X_train, y_train,
                    X_val,   y_val,
                    data_obj,
                    base_cfg:     dict,
                    epochs:       int = 30) -> float:
    """Train for `epochs` epochs and return validation RMSE."""
    cfg = {**base_cfg, **cfg_override}

    try:
        if model_type == "lstm":
            model = build_lstm(cfg)
        else:
            model = build_transformer(cfg)

        trainer = Trainer(
            model          = model,
            lr             = 1e-3,
            patience       = 8,
            checkpoint_dir = os.path.join("outputs", "tune_checkpoints"),
            device         = "auto"
        )
        train_dl, val_dl = build_loaders(X_train, y_train,
                                         X_val,   y_val,
                                         batch_size=64)
        history = trainer.fit(train_dl, val_dl,
                              epochs=epochs, model_name="tune_tmp")
        best_val = history["best_val"]
        return best_val

    except Exception as e:
        print(f"  [Tuning] Config failed: {e}")
        return float("inf")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker",  type=str, default="AAPL")
    parser.add_argument("--model",   type=str, default="lstm",
                        choices=["lstm", "transformer"])
    parser.add_argument("--start",   type=str, default="2015-01-01")
    parser.add_argument("--end",     type=str, default="2024-01-01")
    parser.add_argument("--seq_len", type=int, default=60)
    parser.add_argument("--epochs",  type=int, default=30,
                        help="Epochs per config (keep low for speed)")
    parser.add_argument("--max_configs", type=int, default=12,
                        help="Max configs to evaluate (random subset)")
    args = parser.parse_args()

    # ── Data ─────────────────────────────────
    loader = StockDataLoader(args.ticker, args.start, args.end, args.seq_len)
    loader.build_features()
    (X_train, y_train, X_val, y_val, X_test, y_test, _) = loader.build_sequences()
    n_features = X_train.shape[2]

    base_cfg = {
        "input_size":  n_features,
        "output_size": 1,
        "seq_len":     args.seq_len,
    }

    grid = LSTM_GRID if args.model == "lstm" else TRANSFORMER_GRID
    configs = grid_configs(grid)

    # Random subsample for speed
    np.random.shuffle(configs)
    configs = configs[: args.max_configs]

    print(f"\n[Tuning] Evaluating {len(configs)} configs for '{args.model.upper()}' …\n")

    results = []
    for i, cfg in enumerate(configs, 1):
        print(f"  [{i}/{len(configs)}] Config: {cfg}")
        score = evaluate_config(cfg, args.model,
                                X_train, y_train,
                                X_val,   y_val,
                                loader,  base_cfg,
                                epochs=args.epochs)
        results.append({"config": cfg, "val_loss": score})
        print(f"         Val Loss = {score:.5f}\n")

    # Sort and display
    results.sort(key=lambda r: r["val_loss"])
    print("\n" + "=" * 60)
    print(f"  TOP 3 CONFIGS for {args.model.upper()}")
    print("=" * 60)
    for rank, r in enumerate(results[:3], 1):
        print(f"  #{rank}  Val Loss={r['val_loss']:.5f}")
        for k, v in r["config"].items():
            print(f"       {k}: {v}")
        print()

    # Save results
    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", f"tuning_{args.model}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Tuning] Results saved → {out_path}")


if __name__ == "__main__":
    main()
