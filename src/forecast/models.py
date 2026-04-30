"""Forecasting models: Naive, ARIMA, Prophet, XGBoost, LSTM.

Each model exposes a uniform interface:

    model = SomeModel(...)
    model.fit(train_df)              # train_df: ['ds', 'y']
    forecast = model.predict(horizon) # returns np.ndarray of length `horizon`

"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Naive baseline
# ---------------------------------------------------------------------------
@dataclass
class NaiveLast:
    """Repeat the last training value."""
    last_value: float = 0.0

    def fit(self, df: pd.DataFrame) -> "NaiveLast":
        self.last_value = float(df["y"].iloc[-1])
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self.last_value, dtype=float)


@dataclass
class MovingAverage:
    """Predict the mean of the last `window` observations."""
    window: int = 4
    value: float = 0.0

    def fit(self, df: pd.DataFrame) -> "MovingAverage":
        self.value = float(df["y"].tail(self.window).mean())
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self.value, dtype=float)


# ---------------------------------------------------------------------------
# ARIMA / SARIMA
# ---------------------------------------------------------------------------
class ARIMAModel:
    def __init__(self, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_ = None

    def fit(self, df: pd.DataFrame) -> "ARIMAModel":
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        self.model_ = SARIMAX(
            df["y"].values,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        fc = self.model_.forecast(steps=horizon)
        return np.asarray(fc, dtype=float)


# ---------------------------------------------------------------------------
# Prophet
# ---------------------------------------------------------------------------
class ProphetModel:
    def __init__(self, weekly: bool = True, yearly: bool = True):
        self.weekly = weekly
        self.yearly = yearly
        self.model_ = None
        self.freq_ = "W"

    def fit(self, df: pd.DataFrame) -> "ProphetModel":
        from prophet import Prophet

        m = Prophet(
            weekly_seasonality=self.weekly,
            yearly_seasonality=self.yearly,
            daily_seasonality=False,
        )
        m.fit(df.rename(columns={"ds": "ds", "y": "y"}))
        self.model_ = m
        # Detect frequency from training data.
        diffs = pd.Series(pd.to_datetime(df["ds"])).diff().dropna()
        if not diffs.empty:
            median = diffs.median()
            if median <= pd.Timedelta(days=1):
                self.freq_ = "D"
            elif median <= pd.Timedelta(days=7):
                self.freq_ = "W"
            else:
                self.freq_ = "MS"
        return self

    def predict(self, horizon: int) -> np.ndarray:
        future = self.model_.make_future_dataframe(periods=horizon, freq=self.freq_)
        fc = self.model_.predict(future)
        return fc["yhat"].tail(horizon).values.astype(float)


# ---------------------------------------------------------------------------
# XGBoost with lag features
# ---------------------------------------------------------------------------
class XGBLagModel:
    def __init__(self, lags: int = 8, n_estimators: int = 300, max_depth: int = 4):
        self.lags = lags
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model_ = None
        self.history_: list[float] = []

    @staticmethod
    def _make_lags(y: np.ndarray, lags: int):
        X, target = [], []
        for i in range(lags, len(y)):
            X.append(y[i - lags:i])
            target.append(y[i])
        return np.array(X), np.array(target)

    def fit(self, df: pd.DataFrame) -> "XGBLagModel":
        from xgboost import XGBRegressor

        y = df["y"].values.astype(float)
        if len(y) <= self.lags + 1:
            raise ValueError(
                f"Need more than {self.lags} observations to fit XGBLagModel."
            )
        X, t = self._make_lags(y, self.lags)
        self.model_ = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=0.05,
            objective="reg:squarederror",
            verbosity=0,
        )
        self.model_.fit(X, t)
        self.history_ = list(y)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        preds: list[float] = []
        history = list(self.history_)
        for _ in range(horizon):
            x = np.array(history[-self.lags:]).reshape(1, -1)
            yhat = float(self.model_.predict(x)[0])
            preds.append(yhat)
            history.append(yhat)
        return np.array(preds, dtype=float)


# ---------------------------------------------------------------------------
# LSTM (PyTorch)
# ---------------------------------------------------------------------------
class LSTMModel:
    def __init__(
        self,
        lags: int = 12,
        hidden_size: int = 32,
        num_layers: int = 1,
        epochs: int = 200,
        lr: float = 5e-3,
    ):
        self.lags = lags
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.model_ = None
        self.scale_ = 1.0
        self.shift_ = 0.0
        self.history_: list[float] = []

    def _build(self):
        import torch
        from torch import nn

        class Net(nn.Module):
            def __init__(self, hidden, layers):
                super().__init__()
                self.lstm = nn.LSTM(1, hidden, layers, batch_first=True)
                self.fc = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :]).squeeze(-1)

        torch.manual_seed(42)
        return Net(self.hidden_size, self.num_layers)

    def fit(self, df: pd.DataFrame) -> "LSTMModel":
        import torch
        from torch import nn

        y = df["y"].values.astype(float)
        if len(y) <= self.lags + 4:
            raise ValueError("Not enough data for LSTM.")

        # Standardize
        self.shift_ = float(y.mean())
        self.scale_ = float(y.std() + 1e-6)
        ys = (y - self.shift_) / self.scale_

        X, t = [], []
        for i in range(self.lags, len(ys)):
            X.append(ys[i - self.lags:i])
            t.append(ys[i])
        X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
        t = torch.tensor(np.array(t), dtype=torch.float32)

        net = self._build()
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            yhat = net(X)
            loss = loss_fn(yhat, t)
            loss.backward()
            opt.step()
        self.model_ = net
        self.history_ = list(ys)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        import torch

        self.model_.eval()
        preds: list[float] = []
        history = list(self.history_)
        with torch.no_grad():
            for _ in range(horizon):
                x = torch.tensor(history[-self.lags:], dtype=torch.float32).reshape(1, -1, 1)
                yhat = float(self.model_(x).item())
                preds.append(yhat)
                history.append(yhat)
        arr = np.array(preds, dtype=float) * self.scale_ + self.shift_
        return np.clip(arr, a_min=0.0, a_max=None)
