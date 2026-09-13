"""
data_loader.py
--------------
Handles downloading, cleaning, and feature engineering for stock data.
Uses yfinance to fetch OHLCV data and adds a comprehensive set of
technical indicators for machine learning.

Feature Groups
--------------
  1. Price returns (1d, 5d, 20d log-returns)
  2. Moving averages — SMA & EMA (5, 10, 20, 50, 200)
  3. MA cross-over signals
  4. Momentum — RSI(7), RSI(14), Williams %R, Stochastic Oscillator, CCI
  5. MACD (line, signal, histogram)
  6. Bollinger Bands (upper, middle, lower, width, %B)
  7. Volatility — ATR(14), historical vol (10d, 30d), Keltner Channel
  8. Volume — OBV, Chaikin Money Flow (CMF), Volume Ratio
  9. Price position — HL Ratio, OC Ratio
  10. Cyclical time features — day-of-week, month-of-year (sin/cos encoded)
  11. Lag features — Close at lag 1, 2, 3, 5, 10
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler, MinMaxScaler


# ─────────────────────────────────────────────
# Technical Indicator Helpers
# ─────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder smoothing)."""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series,
                 fast: int = 12, slow: int = 26,
                 signal: int = 9) -> pd.DataFrame:
    """MACD line, signal line, and histogram."""
    ema_fast    = series.ewm(span=fast,   adjust=False).mean()
    ema_slow    = series.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return pd.DataFrame({"MACD":       macd_line,
                         "MACD_Signal": signal_line,
                         "MACD_Hist":  histogram})


def compute_bollinger_bands(series: pd.Series,
                            window: int = 20,
                            num_std: float = 2.0) -> pd.DataFrame:
    """Upper, middle (SMA), lower Bollinger Bands + %B + bandwidth."""
    sma     = series.rolling(window).mean()
    std     = series.rolling(window).std()
    upper   = sma + num_std * std
    lower   = sma - num_std * std
    bb_width = (upper - lower) / (sma + 1e-9)
    bb_pct   = (series - lower) / (upper - lower + 1e-9)  # %B: position in band
    return pd.DataFrame({"BB_Upper":  upper,
                         "BB_Middle": sma,
                         "BB_Lower":  lower,
                         "BB_Width":  bb_width,
                         "BB_Pct":    bb_pct})


def compute_atr(high: pd.Series, low: pd.Series,
                close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range — volatility measure."""
    prev_close = close.shift(1)
    tr = pd.concat([high - low,
                    (high - prev_close).abs(),
                    (low  - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def compute_williams_r(high: pd.Series, low: pd.Series,
                        close: pd.Series, period: int = 14) -> pd.Series:
    """Williams %R momentum oscillator. Range: -100 to 0."""
    highest_high = high.rolling(period).max()
    lowest_low   = low.rolling(period).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low + 1e-9)


def compute_stochastic(high: pd.Series, low: pd.Series,
                        close: pd.Series,
                        k_period: int = 14,
                        d_period: int = 3) -> pd.DataFrame:
    """Stochastic Oscillator (%K and %D lines)."""
    lowest_low   = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    pct_k        = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-9)
    pct_d        = pct_k.rolling(d_period).mean()
    return pd.DataFrame({"Stoch_K": pct_k, "Stoch_D": pct_d})


def compute_cci(high: pd.Series, low: pd.Series,
                close: pd.Series, period: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    typical_price = (high + low + close) / 3
    sma_tp        = typical_price.rolling(period).mean()
    mad           = typical_price.rolling(period).apply(
                        lambda x: np.mean(np.abs(x - np.mean(x))),
                        raw=True)
    return (typical_price - sma_tp) / (0.015 * mad + 1e-9)


def compute_cmf(high: pd.Series, low: pd.Series,
                close: pd.Series, volume: pd.Series,
                period: int = 20) -> pd.Series:
    """Chaikin Money Flow — combines price and volume momentum."""
    mfm    = ((close - low) - (high - close)) / (high - low + 1e-9)
    mfv    = mfm * volume
    return mfv.rolling(period).sum() / volume.rolling(period).sum().replace(0, 1e-9)


# ─────────────────────────────────────────────
# Main Data Loader
# ─────────────────────────────────────────────

class StockDataLoader:
    """
    Downloads OHLCV data and engineers a comprehensive feature set
    for stock price forecasting using ML models.

    Parameters
    ----------
    ticker     : str   – Yahoo Finance ticker symbol (e.g. 'AAPL')
    start_date : str   – Start date 'YYYY-MM-DD'
    end_date   : str   – End date   'YYYY-MM-DD'
    seq_len    : int   – Lookback window length for sequence models
    target_col : str   – Column to predict (default 'Close')
    """

    def __init__(self,
                 ticker:      str  = "AAPL",
                 start_date:  str  = "2015-01-01",
                 end_date:    str  = "2024-01-01",
                 seq_len:     int  = 60,
                 target_col:  str  = "Close"):
        self.ticker     = ticker
        self.start_date = start_date
        self.end_date   = end_date
        self.seq_len    = seq_len
        self.target_col = target_col
        self.scaler        = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.raw_df        = None
        self.feature_df    = None

    # ── Download ──────────────────────────────
    def download(self) -> pd.DataFrame:
        cache_dir = os.path.join(os.path.dirname(__file__), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        clean_start = str(self.start_date).replace(":", "-")
        clean_end   = str(self.end_date).replace(":", "-")
        cache_file  = os.path.join(cache_dir,
                                   f"{self.ticker}_{clean_start}_{clean_end}.csv")

        if os.path.exists(cache_file):
            print(f"[DataLoader] Loading cached {self.ticker} data from {cache_file}...")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            # Flatten multi-level columns if present in old cache
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            self.raw_df = df
            print(f"[DataLoader] Loaded {len(df)} rows from cache.")
            return df

        print(f"[DataLoader] Downloading {self.ticker} from "
              f"{self.start_date} to {self.end_date}...")
        df = yf.download(self.ticker,
                         start=self.start_date,
                         end=self.end_date,
                         auto_adjust=True,
                         progress=False)
        if df.empty:
            raise ValueError(f"No data returned for ticker '{self.ticker}'.")
        df.index = pd.to_datetime(df.index)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        try:
            df.to_csv(cache_file)
        except Exception:
            pass
        self.raw_df = df
        print(f"[DataLoader] Downloaded {len(df)} rows.")
        return df

    # ── Feature Engineering ───────────────────
    def build_features(self) -> pd.DataFrame:
        """
        Compute all 45+ technical features and return a clean DataFrame.
        """
        if self.raw_df is None:
            self.download()
        df = self.raw_df.copy()

        # Forward-fill missing values first (common in stock data around holidays)
        df = df.ffill()

        close  = df["Close"]
        high   = df["High"]
        low    = df["Low"]
        volume = df["Volume"]

        # ─── 1. Price-derived features ────────
        df["Return_1d"]  = close.pct_change(1)
        df["Return_5d"]  = close.pct_change(5)
        df["Return_20d"] = close.pct_change(20)
        # Log returns (more stationary)
        df["LogReturn_1d"] = np.log(close / close.shift(1))

        # ─── 2. Moving averages ───────────────
        for w in [5, 10, 20, 50, 200]:
            df[f"SMA_{w}"]  = close.rolling(w).mean()
            df[f"EMA_{w}"]  = close.ewm(span=w, adjust=False).mean()

        # Price relative to moving averages (normalized)
        df["Price_SMA20_ratio"] = close / (df["SMA_20"] + 1e-9)
        df["Price_SMA50_ratio"] = close / (df["SMA_50"] + 1e-9)

        # ─── 3. MA cross signals ──────────────
        df["SMA_5_20_cross"]  = (df["SMA_5"]  > df["SMA_20"]).astype(float)
        df["SMA_20_50_cross"] = (df["SMA_20"] > df["SMA_50"]).astype(float)

        # ─── 4. Momentum indicators ───────────
        df["RSI_14"]  = compute_rsi(close, 14)
        df["RSI_7"]   = compute_rsi(close, 7)
        df["Williams_R"] = compute_williams_r(high, low, close, 14)

        stoch_df = compute_stochastic(high, low, close)
        df = pd.concat([df, stoch_df], axis=1)

        df["CCI_20"] = compute_cci(high, low, close, 20)

        # ─── 5. MACD ──────────────────────────
        macd_df = compute_macd(close)
        df = pd.concat([df, macd_df], axis=1)

        # ─── 6. Bollinger Bands ───────────────
        bb_df = compute_bollinger_bands(close)
        df = pd.concat([df, bb_df], axis=1)

        # ─── 7. Volatility ────────────────────
        df["ATR_14"]       = compute_atr(high, low, close, 14)
        df["Volatility_10d"] = close.pct_change().rolling(10).std()
        df["Volatility_30d"] = close.pct_change().rolling(30).std()

        # ─── 8. Volume features ───────────────
        df["OBV"]        = compute_obv(close, volume)
        df["CMF_20"]     = compute_cmf(high, low, close, volume, 20)
        df["Vol_SMA_10"] = volume.rolling(10).mean()
        df["Vol_Ratio"]  = volume / (df["Vol_SMA_10"] + 1e-9)

        # ─── 9. Price position in range ───────
        df["HL_Ratio"] = (close - low)  / (high - low + 1e-9)
        df["OC_Ratio"] = (df["Open"] - close).abs() / (high - low + 1e-9)

        # ─── 10. Cyclical time features ───────
        # Sin/cos encoding captures the circular nature of time
        dow = df.index.dayofweek.astype(float)
        moy = df.index.month.astype(float)
        df["DoW_sin"] = np.sin(2 * np.pi * dow / 5)
        df["DoW_cos"] = np.cos(2 * np.pi * dow / 5)
        df["MoY_sin"] = np.sin(2 * np.pi * moy / 12)
        df["MoY_cos"] = np.cos(2 * np.pi * moy / 12)

        # ─── 11. Lag features ─────────────────
        for lag in [1, 2, 3, 5, 10]:
            df[f"Close_lag_{lag}"] = close.shift(lag)

        # ─── Target ───────────────────────────
        df["Target"] = close.shift(-1)   # next-day close price

        # Drop NaNs created by rolling windows
        df.dropna(inplace=True)
        self.feature_df = df
        print(f"[DataLoader] Feature matrix: {df.shape[0]} rows x "
              f"{df.shape[1]} columns")
        return df

    # ── Sequence Builder ─────────────────────
    def build_sequences(self,
                        test_split: float = 0.2,
                        val_split:  float = 0.1):
        """
        Build (X, y) sequences for LSTM / Transformer consumption.

        Uses RobustScaler on features (handles outliers) and
        MinMaxScaler on target (stable inverse transform).

        Returns
        -------
        X_train, y_train, X_val, y_val, X_test, y_test : np.ndarray
        feature_cols : list[str]
        """
        if self.feature_df is None:
            self.build_features()

        df = self.feature_df.copy()

        # Columns used as model input (exclude raw OHLCV + Target)
        exclude = {"Open", "High", "Low", "Close", "Volume", "Target"}
        feature_cols = [c for c in df.columns if c not in exclude]
        # Put target column (Close) first so models can learn price level
        feature_cols = [self.target_col] + [c for c in feature_cols
                                            if c != self.target_col]

        data   = df[feature_cols].values
        target = df["Target"].values.reshape(-1, 1)

        # Chronological split BEFORE scaling to prevent data leakage
        n       = len(data)
        n_test  = int(n * test_split)
        n_val   = int(n * val_split)
        n_train = n - n_test - n_val

        # Scale features and target to [0, 1] range (standard time series DL formulation)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled   = self.scaler.fit_transform(data)
        target_scaled = self.target_scaler.fit_transform(target)

        # Create sliding windows
        X, y = [], []
        for i in range(self.seq_len, len(data_scaled)):
            X.append(data_scaled[i - self.seq_len: i])
            y.append(target_scaled[i, 0])
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        # Re-compute split indices for sequences
        n_seq   = len(X)
        n_test_seq  = int(n_seq * test_split)
        n_val_seq   = int(n_seq * val_split)
        n_train_seq = n_seq - n_test_seq - n_val_seq

        X_train = X[:n_train_seq];                    y_train = y[:n_train_seq]
        X_val   = X[n_train_seq: n_train_seq + n_val_seq]; y_val = y[n_train_seq: n_train_seq + n_val_seq]
        X_test  = X[n_train_seq + n_val_seq:];        y_test  = y[n_train_seq + n_val_seq:]

        print(f"[DataLoader] Train={len(X_train)}, "
              f"Val={len(X_val)}, Test={len(X_test)} sequences.")
        print(f"[DataLoader] Features: {len(feature_cols)} | "
              f"Scaler: RobustScaler")
        return (X_train, y_train,
                X_val,   y_val,
                X_test,  y_test,
                feature_cols)

    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """Convert scaled predictions back to original price scale."""
        return self.target_scaler.inverse_transform(
            y_scaled.reshape(-1, 1)).flatten()
