import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from .config import WINDOW, SEED


def _time_feats(dates: pd.DatetimeIndex) -> np.ndarray:
    ang = 2 * np.pi * dates.dayofyear.values / 365.25
    return np.column_stack([np.sin(ang), np.cos(ang)])


def _windows(feat: np.ndarray, window: int):
    X, y = [], []
    for i in range(len(feat) - window):
        X.append(feat[i:i + window])
        y.append(feat[i + window, 0])
    return np.array(X), np.array(y)


class LSTMForecaster:
    """AQI + calendar features (sin/cos of day-of-year) -> next-day AQI.
    Multi-step forecasts are produced recursively."""

    def __init__(self, window: int = WINDOW, epochs: int = 60):
        self.window, self.epochs = window, epochs
        self.scaler = MinMaxScaler()
        self.model = None

    def _build(self, n_feat):
        m = tf.keras.Sequential([
            tf.keras.layers.Input((self.window, n_feat)),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ])
        m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
        return m

    def fit(self, series: pd.Series, verbose: int = 0):
        tf.keras.utils.set_random_seed(SEED)
        vals = self.scaler.fit_transform(series.values.reshape(-1, 1))
        feat = np.hstack([vals, _time_feats(series.index)])
        X, y = _windows(feat, self.window)
        self.model = self._build(feat.shape[1])
        cb = [tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)]
        hist = self.model.fit(X, y, epochs=self.epochs, batch_size=32,
                              validation_split=0.1, shuffle=False,
                              callbacks=cb, verbose=verbose)
        self.history = hist.history
        return self

    def forecast(self, history: pd.Series, steps: int) -> pd.Series:
        vals = self.scaler.transform(history.values.reshape(-1, 1))
        feat = np.hstack([vals, _time_feats(history.index)])
        win = feat[-self.window:].copy()
        dates = pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")
        tf_next = _time_feats(dates)
        preds = []
        for i in range(steps):
            p = float(self.model.predict(win[np.newaxis], verbose=0)[0, 0])
            preds.append(p)
            win = np.vstack([win[1:], [p, *tf_next[i]]])
        preds = self.scaler.inverse_transform(np.array(preds).reshape(-1, 1)).ravel()
        return pd.Series(np.clip(preds, 0, None), index=dates, name="lstm")

    def one_step(self, full: pd.Series, n_last: int) -> pd.Series:
        """Next-day predictions for the last n_last days using real past values."""
        vals = self.scaler.transform(full.values.reshape(-1, 1))
        feat = np.hstack([vals, _time_feats(full.index)])
        X = np.array([feat[i - self.window:i] for i in range(len(full) - n_last, len(full))])
        p = self.model.predict(X, verbose=0).ravel()
        p = self.scaler.inverse_transform(p.reshape(-1, 1)).ravel()
        return pd.Series(np.clip(p, 0, None), index=full.index[-n_last:], name="lstm_1step")
