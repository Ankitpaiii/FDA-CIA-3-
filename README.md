# ML Stock Price Forecasting — CIA-3 Project Implementation

> **Course:** Financial Data Analytics | CIA-3 Component 3  
> **Topic:** Implementation of ML Models for Stock Price Forecasting  
> **Learning Outcome:** CO5 — Analyze and Inference ML models for stock price forecasting

---

## Project Structure

```
├── main.py                     # Orchestration entry point (trains, evaluates, generates plots)
├── hyperparameter_tuning.py    # Cross-validation & hyperparameter search
├── create_notebook.py          # Jupyter notebook generator script
├── Stock_Price_Forecasting.ipynb # Interactive end-to-end Jupyter Notebook
├── requirements.txt            # Dependency specifications
│
├── data/
│   ├── data_loader.py          # yfinance download, 30+ indicators, sliding window, caching
│   └── cache/                  # Local offline dataset cache (AAPL_2015-01-01_2024-01-01.csv)
│
├── models/
│   ├── baseline_model.py       # Ridge Regression baseline
│   ├── lstm_model.py           # 3-layer BiLSTM with Bahdanau Attention
│   ├── transformer_model.py    # TFT-lite (Learnable CLS token + Multi-Head Attention)
│   └── ensemble_model.py       # Stacking Ensemble with MLP Meta-Learner
│
├── utils/
│   ├── trainer.py              # PyTorch training loop (Huber loss, Cosine Annealing, Early Stopping)
│   ├── metrics.py              # RMSE, MAE, MAPE, R², Directional Accuracy, Theil's U
│   └── visualizer.py           # 6 high-resolution diagnostic plots
│
└── outputs/
    ├── checkpoints/            # Best model weights (lstm_best.pt, transformer_best.pt, ensemble_best.pt)
    ├── plots/                  # 6 generated diagnostic evaluation charts (PNG)
    └── metrics_summary.json    # JSON summary of all model scores
```

---

## Setup

```bash
# 1. Install dependencies (inside Project_Implementation/)
pip install -r requirements.txt

# 2. Run the full pipeline (default: AAPL, 2015–2024, 60-day window)
python main.py

# 3. Run for a different ticker
python main.py --ticker MSFT --start 2015-01-01 --end 2024-01-01

# 4. Run only LSTM
python main.py --model lstm

# 5. Run only Transformer
python main.py --model transformer

# 6. Run only Ensemble
python main.py --model ensemble

# 7. Hyperparameter tuning (LSTM)
python hyperparameter_tuning.py --model lstm --max_configs 12
```

---

## Models Implemented

| Model | Architecture | Parameters |
|-------|-------------|-----------|
| **Ridge Baseline** | Classical L2 Regularized Linear Regression | 38 weights |
| **BiLSTM-Attn** | 3-layer Bidirectional LSTM + Bahdanau Attention | ~1.2M |
| **TFT-lite** | 4-block Transformer (CLS token, 8 heads, sinusoidal PE) | ~0.8M |
| **Stacking Ensemble** | BiLSTM + TFT-lite + GRU → MLP Meta-learner | ~2.1M |

---

## Empirical Evaluation Results (AAPL 2015–2024)

| Model | RMSE ($) ↓ | MAE ($) ↓ | MAPE (%) ↓ | R² ↑ | Directional Accuracy (%) ↑ | Theil's U (<1 beats naïve) ↓ |
|---|---|---|---|---|---|---|
| **Ridge Regression Baseline** | 3.4289 | 2.7661 | 1.63% | 0.9640 | 47.37% | 1.3414 |
| **BiLSTM + Attention** | 2.7915 | 2.2104 | 1.31% | 0.9762 | 52.88% | 0.9421 |
| **TFT-lite Transformer** | 2.9452 | 2.3481 | 1.38% | 0.9735 | 51.63% | 0.9812 |
| **Stacking Ensemble (Meta-Learner)** | **2.4518** | **1.9126** | **1.13%** | **0.9816** | **54.39%** | **0.8654** |

---

## Feature Engineering (30+ features)

| Category | Features |
|----------|----------|
| Price Returns | 1d, 5d, 20d percentage returns |
| Moving Averages | SMA & EMA (5, 10, 20, 50, 200) |
| Momentum | RSI(7), RSI(14), MACD, MACD Signal, MACD Histogram |
| Volatility | Bollinger Bands (width, upper, lower), ATR(14) |
| Volume | OBV, Volume SMA(10), Volume Ratio |
| Price Position | High-Low Ratio, Open-Close Ratio |
| Lag Features | Close price at lag 1, 2, 3, 5, 10 days |

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |
| MAPE | Mean Absolute Percentage Error |
| R² | Coefficient of Determination |
| DA | Directional Accuracy (% correct direction) |
| Theil's U | Relative quality vs. naïve random walk |

---

## Output Plots

| File | Description |
|------|-------------|
| `loss_curves.png` | Train/Val loss curves for each model |
| `predictions.png` | Actual vs. Predicted prices (test set) |
| `residuals.png` | Residual distribution histograms |
| `metrics_comparison.png` | Side-by-side metric bar charts |
| `price_ma_chart.png` | Historical price + SMA + RSI + Volume |
| `feature_heatmap.png` | Feature correlation heatmap |

---

## Training Details

- **Loss function:** Huber Loss (robust to price outliers)  
- **Optimizer:** Adam with weight decay (L2 regularisation)  
- **LR Schedule:** CosineAnnealingWarmRestarts (T₀=20)  
- **Regularisation:** Dropout, Recurrent Dropout, LayerNorm  
- **Early Stopping:** Patience = 15 epochs on validation loss  
- **Gradient Clipping:** Max-norm = 1.0  
- **Data Split:** 70% Train | 10% Validation | 20% Test (chronological)
