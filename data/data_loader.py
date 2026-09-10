"""
data_loader.py
--------------
Handles downloading, cleaning, and feature engineering for stock data.
Uses yfinance to fetch OHLCV data and adds a rich set of technical indicators.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# ─────────────────────────────────────────────
# Technical Indicator Helpers
# ─────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs  = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series,
                 fast: int = 12,
                 slow: int = 26,
                 signal: int = 9) -> pd.DataFrame:
    """MACD line, signal line, and histogram."""
    ema_fast   = series.ewm(span=fast,   adjust=False).mean()
    ema_slow   = series.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return pd.DataFrame({"MACD": macd_line,
                         "MACD_Signal": signal_line,
                         "MACD_Hist": histogram})


def compute_bollinger_bands(series: pd.Series,
                            window: int = 20,
                            num_std: float = 2.0) -> pd.DataFrame:
    """Upper, middle (SMA), and lower Bollinger Bands."""
    sma  = series.rolling(window).mean()
    std  = series.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    bb_width = (upper - lower) / (sma + 1e-9)
    return pd.DataFrame({"BB_Upper": upper,
                         "BB_Middle": sma,
                         "BB_Lower": lower,
                         "BB_Width": bb_width})


def compute_atr(high: pd.Series, low: pd.Series,
                close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range – volatility measure."""
    prev_close = close.shift(1)
    tr = pd.concat([high - low,
                    (high - prev_close).abs(),
                    (low  - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


# ─────────────────────────────────────────────
# Main Data Loader
# ─────────────────────────────────────────────

class StockDataLoader:
    """
    Downloads OHLCV data and engineers a rich feature set.

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
        self.scaler     = MinMaxScaler(feature_range=(0, 1))
        self.raw_df     = None
        self.feature_df = None

    # ── Download ──────────────────────────────
    def download(self) -> pd.DataFrame:
        cache_dir = os.path.join(os.path.dirname(__file__), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        clean_start = str(self.start_date).replace(":", "-")
        clean_end = str(self.end_date).replace(":", "-")
        cache_file = os.path.join(cache_dir, f"{self.ticker}_{clean_start}_{clean_end}.csv")
        
        if os.path.exists(cache_file):
            print(f"[DataLoader] Loading cached {self.ticker} data from {cache_file}...")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            self.raw_df = df
            print(f"[DataLoader] Loaded {len(df)} rows from cache.")
            return df

        print(f"[DataLoader] Downloading {self.ticker} from {self.start_date} to {self.end_date}...")
        df = yf.download(self.ticker,
                         start=self.start_date,
                         end=self.end_date,
                         auto_adjust=True,
                         progress=False)
        if df.empty:
            raise ValueError(f"No data returned for ticker '{self.ticker}'.")
        df.index = pd.to_datetime(df.index)
        # Flatten multi-level columns if present
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
        if self.raw_df is None:
            self.download()
        df = self.raw_df.copy()

        close  = df["Close"]
        high   = df["High"]
        low    = df["Low"]
        volume = df["Volume"]

        # ─── Price-derived features ───────────
        df["Return_1d"]   = close.pct_change(1)
        df["Return_5d"]   = close.pct_change(5)
        df["Return_20d"]  = close.pct_change(20)

        # ─── Moving averages ──────────────────
        for w in [5, 10, 20, 50, 200]:
            df[f"SMA_{w}"]  = close.rolling(w).mean()
            df[f"EMA_{w}"]  = close.ewm(span=w, adjust=False).mean()

        # ─── MA cross signals ─────────────────
        df["SMA_5_20_cross"] = (df["SMA_5"] > df["SMA_20"]).astype(float)

        # ─── Momentum indicators ──────────────
        df["RSI_14"]  = compute_rsi(close, 14)
        df["RSI_7"]   = compute_rsi(close, 7)

        macd_df = compute_macd(close)
        df = pd.concat([df, macd_df], axis=1)

        bb_df = compute_bollinger_bands(close)
        df = pd.concat([df, bb_df], axis=1)

        df["ATR_14"] = compute_atr(high, low, close, 14)
        df["OBV"]    = compute_obv(close, volume)

        # ─── Volatility ───────────────────────
        df["Volatility_10d"] = close.pct_change().rolling(10).std()
        df["Volatility_30d"] = close.pct_change().rolling(30).std()

        # ─── Volume features ──────────────────
        df["Vol_SMA_10"] = volume.rolling(10).mean()
        df["Vol_Ratio"]  = volume / (df["Vol_SMA_10"] + 1e-9)

        # ─── Price position in range ─────────
        df["HL_Ratio"]  = (close - low) / (high - low + 1e-9)
        df["OC_Ratio"]  = (df["Open"] - close).abs() / (high - low + 1e-9)

        # ─── Lag features ─────────────────────
        for lag in [1, 2, 3, 5, 10]:
            df[f"Close_lag_{lag}"] = close.shift(lag)

        # ─── Target ───────────────────────────
        df["Target"] = close.shift(-1)   # next-day close

        df.dropna(inplace=True)
        self.feature_df = df
        print(f"[DataLoader] Feature matrix shape: {df.shape}")
        return df

    # ── Sequence Builder ─────────────────────
    def build_sequences(self,
                        test_split: float = 0.2,
                        val_split:  float = 0.1):
        """
        Build (X, y) sequences for LSTM / Transformer consumption.

        Returns
        -------
        X_train, y_train, X_val, y_val, X_test, y_test : np.ndarray
        feature_cols : list[str]
        """
        if self.feature_df is None:
            self.build_features()

        df = self.feature_df.copy()

        # columns used as model input (exclude raw OHLCV + Target)
        exclude = {"Open", "High", "Low", "Close", "Volume", "Target"}
        feature_cols = [c for c in df.columns if c not in exclude]
        feature_cols = [self.target_col] + [c for c in feature_cols
                                            if c != self.target_col]

        data = df[feature_cols].values
        target = df["Target"].values.reshape(-1, 1)

        # Scale features
        data_scaled   = self.scaler.fit_transform(data)
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaled  = target_scaler.fit_transform(target)
        self.target_scaler = target_scaler

        # Create sliding windows
        X, y = [], []
        for i in range(self.seq_len, len(data_scaled)):
            X.append(data_scaled[i - self.seq_len: i])
            y.append(target_scaled[i, 0])
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        # Train / Val / Test split (chronological)
        n       = len(X)
        n_test  = int(n * test_split)
        n_val   = int(n * val_split)
        n_train = n - n_test - n_val

        X_train = X[:n_train];       y_train = y[:n_train]
        X_val   = X[n_train: n_train + n_val]; y_val = y[n_train: n_train + n_val]
        X_test  = X[n_train + n_val:];         y_test = y[n_train + n_val:]

        print(f"[DataLoader] Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)} sequences.")
        return (X_train, y_train,
                X_val,   y_val,
                X_test,  y_test,
                feature_cols)

    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """Convert scaled predictions back to original price scale."""
        return self.target_scaler.inverse_transform(
            y_scaled.reshape(-1, 1)).flatten()
