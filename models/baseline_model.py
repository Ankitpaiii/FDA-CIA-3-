"""
baseline_model.py
-----------------
Linear / Ridge Regression baseline for stock price forecasting.
Serves as an essential benchmark to demonstrate the value-add of 
deep learning (LSTM, Transformer, Ensemble) over classical linear methods.
"""

import numpy as np
from sklearn.linear_model import Ridge
from utils.metrics import evaluate_all


class RidgeBaseline:
    """
    Ridge Regression baseline operating on flattened sequence inputs.
    """
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        # Flatten (N, seq_len, features) to (N, seq_len * features)
        N, T, D = X_train.shape
        X_flat = X_train.reshape(N, T * D)
        self.model.fit(X_flat, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        N, T, D = X.shape
        X_flat = X.reshape(N, T * D)
        return self.model.predict(X_flat)
