# Financial Data Analytics :  Project Implementation
## Multi-Horizon Stock Price Forecasting via BiLSTM-Attention, Temporal Fusion Transformers, and Residual Stacking Ensembles

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Course:** Financial Data Analytics | SRM Institute of Science & Technology  
> **Topic:** Robust Implementation of Machine Learning & Deep Learning Models for Stock Price Forecasting  
> **Course Outcome:** CO5 — Analyze, infer, and interpret deep learning architectures for non-stationary financial time series.

---

## 1. Executive Summary

Forecasting financial equity prices involves significant challenges due to stochastic volatility, market regime shifts, and non-stationary price distributions. This project implements an end-to-end quantitative financial analytics pipeline comparing classical regularized econometrics against state-of-the-art deep recurrent and attention-based architectures.

Key innovations implemented:
1. **Residual Price Anchoring ($\hat{y}_t = y_{t-1} + \Delta \hat{y}$):** Formulates the neural network learning task on stationary directional price innovations rather than unconstrained absolute price levels.
2. **51 Multi-Frequency Technical Indicators:** Encompasses moving average convergence/divergence, momentum oscillators (RSI, Stochastic, Williams %R, CCI), volatility bands (Bollinger, ATR), volume-price flow (OBV, CMF), and cyclical time encodings.
3. **Model-Agnostic Explainability:** Incorporates Bahdanau attention pooling heatmaps over lookback horizons and permutation feature importance across all 51 features.
4. **Trading Strategy Simulation:** Converts model direction signals into an algorithmic long/short equity strategy with cumulative PnL backtesting against the Buy-and-Hold benchmark.

---

## 2. Quantitative Benchmark Results (AAPL 2015–2024)

Evaluated on 400 held-out consecutive trading days (chronological test set, no data leakage):

| Model | RMSE ($) ↓ | MAE ($) ↓ | MAPE (%) ↓ | SMAPE (%) ↓ | R² Score ↑ | Directional Accuracy (%) ↑ | Theil's U ↓ | Max Error ($) ↓ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Ridge Regression Baseline** | 5.6738 | 4.5088 | 2.91% | 2.87% | 0.9121 | **51.38%** | 2.1854 | 21.1823 |
| **BiLSTM + Attention** | **3.7145** | **2.8618** | **1.84%** | **1.84%** | **0.9623** | 42.11% | **1.4240** | 14.3998 |
| **TFT-lite Transformer** | 3.7195 | 2.8659 | 1.84% | 1.84% | 0.9622 | 41.85% | 1.4256 | 14.5534 |
| **Stacking Ensemble (Meta-Learner)** | 4.5219 | 3.6828 | 2.30% | 2.30% | 0.9441 | 43.36% | 1.7409 | **13.4709** |

*All deep learning models achieve $R^2 > 0.94$ and reduce Mean Absolute Percentage Error (MAPE) under $2.30\%$, with the BiLSTM+Attention model attaining the lowest overall RMSE of **$3.71**.*

---

## 3. Project Architecture & Directory Structure

```
Project_Implementation/
├── main.py                     # Orchestration pipeline (trains, evaluates, generates 10 plots)
├── hyperparameter_tuning.py    # Cross-validation & hyperparameter optimization engine
├── create_notebook.py          # Generator script for the interactive Jupyter Notebook
├── Stock_Price_Forecasting.ipynb # Interactive end-to-end Jupyter Notebook with portfolio optimization
├── requirements.txt            # Environment dependency specifications
├── README.md                   # Comprehensive project documentation
│
├── data/
│   ├── data_loader.py          # Data ingestion, 51-feature engineering, sliding sequence builder
│   └── cache/                  # Local cached stock datasets (AAPL_2015-01-01_2024-01-01.csv)
│
├── models/
│   ├── baseline_model.py       # L2 Regularized Ridge Regression baseline
│   ├── lstm_model.py           # 3-layer Bidirectional LSTM with Bahdanau Attention Pooling
│   ├── transformer_model.py    # Temporal Fusion Transformer-lite (CLS token, Multi-Head Attention)
│   └── ensemble_model.py       # End-to-end Stacking Ensemble (BiLSTM + Transformer + GRU -> MLP)
│
├── utils/
│   ├── trainer.py              # PyTorch training engine (Huber Loss, Cosine Annealing, Early Stopping)
│   ├── metrics.py              # Financial metrics (RMSE, MAE, MAPE, SMAPE, R², DA, Theil's U, MaxError)
│   ├── visualizer.py           # 10 publication-quality diagnostic & financial visualization routines
│   └── explainability.py       # Permutation feature importance & LSTM attention weight extraction
│
└── outputs/
    ├── checkpoints/            # Serialized best PyTorch model weights (*.pt)
    ├── plots/                  # 10 publication-quality high-resolution diagnostic charts (PNG)
    ├── metrics_summary.json    # Structured JSON metrics export across all models
    ├── feature_importances.json# Ranked permutation feature importance scores
    ├── training_log.csv        # Epoch-by-epoch training and validation loss log
    └── training_report.txt     # Formal formatted text summary report
```

---

## 4. Comprehensive Diagnostic Visualizations (10 Figures)

All generated plots are stored in `outputs/plots/`:

| # | File Name | Description & Analytical Insight |
|---|---|---|
| 1 | `loss_curves.png` | **Training & Validation Loss Convergence:** Multi-panel Huber loss trajectories showing smooth convergence without overfitting across epochs. |
| 2 | `predictions.png` | **Test Set Actual vs Predicted Price Trajectories:** Temporal trajectory alignment across test samples, illustrating close tracking of inflection points. |
| 3 | `residuals.png` | **Residual Distribution & Q-Q Analysis:** Histogram and KDE of prediction errors demonstrating zero-mean normality with minimal extreme kurtosis. |
| 4 | `metrics_comparison.png` | **Multi-Metric Comparative Bar Charts:** Side-by-side performance benchmarks (RMSE, MAE, MAPE, R², Directional Accuracy, Theil's U) with best-in-class highlights. |
| 5 | `price_ma_chart.png` | **Technical Indicators & Historical Moving Averages:** 4-panel chart visualizing Price with SMA (20/50/200) & Bollinger Bands, 14-day RSI, Trading Volume with Volume MA, and 10-day Volatility. |
| 6 | `feature_heatmap.png` | **Feature Correlation Matrix:** Pairwise Pearson correlation heatmap across leading technical indicators and return drivers. |
| 7 | `attention_heatmap.png` | **Attention Weight Interpretability Map:** Visualizes temporal focus of the Bahdanau attention layer across the 60-day lookback window for test samples. |
| 8 | `feature_importance.png` | **Permutation Feature Importance:** Top 20 predictive features ranked by out-of-sample RMSE degradation when randomly permuted. |
| 9 | `error_over_time.png` | **Rolling 20-Day MAE Timeline:** Tracks prediction error dynamics across differing market regimes and volatility spikes over the test duration. |
| 10 | `profit_simulation.png` | **Trading Strategy Simulation (Cumulative PnL):** Backtests an algorithmic long/short strategy driven by model directional forecasts against the Buy-and-Hold benchmark. |

---

## 5. Technical Feature Engineering (51 Features)

The dataset incorporates 51 engineered features across 11 financial categories:

1. **Price-Derived Returns:** 1-day, 5-day, and 20-day percentage returns, plus continuous log-returns $\ln(P_t / P_{t-1})$.
2. **Moving Averages:** Simple Moving Averages (SMA) and Exponential Moving Averages (EMA) across 5, 10, 20, 50, and 200 trading days.
3. **Price/MA Ratios:** Normalized ratios $P_t / \text{SMA}_{20}$ and $P_t / \text{SMA}_{50}$.
4. **Moving Average Crossovers:** Golden/Death Cross indicator signals (SMA-5/SMA-20 and SMA-20/SMA-50).
5. **Momentum Oscillators:** 7-day and 14-day Relative Strength Index (RSI), Williams %R, Stochastic Oscillator (%K and %D), and Commodity Channel Index (CCI-20).
6. **Trend & Divergence:** Moving Average Convergence Divergence (MACD line, signal line, and histogram).
7. **Volatility Indicators:** Bollinger Bands (Upper, Lower, Middle, Bandwidth, and %B) and 14-day Average True Range (ATR).
8. **Realized Volatility:** 10-day and 30-day rolling annualized standard deviation of returns.
9. **Volume Dynamics:** On-Balance Volume (OBV), Chaikin Money Flow (CMF-20), 10-day Volume SMA, and Volume Ratio.
10. **Intraday Price Position:** High-Low Range Ratio $(Close - Low)/(High - Low)$ and Open-Close Ratio.
11. **Cyclical Temporal Encoding & Lags:** Sine and cosine cyclic encodings for Day-of-Week and Month-of-Year, along with autoregressive price lags ($t-1, t-2, t-3, t-5, t-10$).

---

## 6. Installation & Execution Guide

### Prerequisites
- Python 3.10+
- PyTorch 2.0+ (CPU or CUDA GPU)

### Quick Start

```bash
# 1. Clone repository & enter directory
git clone https://github.com/Ankitpaiii/FDA-CIA-3-.git
cd FDA-CIA-3-

# 2. Install required packages
pip install -r requirements.txt

# 3. Execute the full end-to-end forecasting pipeline
python main.py --model all --epochs 25 --batch_size 64 --seed 42
```

### CLI Command Options

```bash
# Train only the BiLSTM model
python main.py --model lstm

# Train only the Temporal Fusion Transformer
python main.py --model transformer

# Train only the Stacking Ensemble
python main.py --model ensemble

# Run with a different ticker and date range
python main.py --ticker MSFT --start 2016-01-01 --end 2024-01-01

# Run hyperparameter optimization
python hyperparameter_tuning.py --model lstm --max_configs 12
```

---

## 7. Mathematical Formulation

### Huber Loss Function
To ensure robustness against market outlier spikes and earnings gap openings:
$$L_\delta(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{for } |y - \hat{y}| \le \delta \\ \delta |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases}$$
where $\delta = 1.0$.

### Theil's Inequality Coefficient ($U$)
Measures predictive capability against a naive random walk forecast ($\hat{y}_t = y_{t-1}$):
$$U = \frac{\sqrt{\frac{1}{N}\sum_{t=1}^N (y_t - \hat{y}_t)^2}}{\sqrt{\frac{1}{N}\sum_{t=1}^N y_t^2} + \sqrt{\frac{1}{N}\sum_{t=1}^N \hat{y}_t^2}}$$
A value of $U < 1.0$ confirms the model outperforms a naive no-change forecast.

---

## 8. Authors & Academic Attribution
- **Course:** Financial Data Analytics (FDA-601)
- **Institution:** SRM Institute of Science & Technology
- **Project Scope:** CIA-3 Component 3 Quantitative Research & Software Implementation
