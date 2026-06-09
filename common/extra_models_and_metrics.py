"""
Extra Models and Metrics for IoT Energy Forecasting
====================================================
Version: 2.0.0 (2025-12-31)

This module provides lightweight wrappers around ML/DL models for energy forecasting:
- Linear Models: Ridge, ElasticNet
- Ensemble: RandomForest, ExtraTrees
- Boosting: CatBoost, LightGBM, XGBoost, NGBoost
- Statistical: SARIMAX, Prophet
- Deep Learning: LSTM, GRU, BiLSTM, PatchTST, TFT
- Baseline: Naive

Each adapter follows a consistent fit/predict interface with proper target scaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import math
import os
import warnings

import numpy as np
import pandas as pd

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:  # optional PyTorch dependency
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - handled gracefully at runtime
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _to_numpy(x: Iterable[float]) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(float, copy=False)
    return np.asarray(list(x), dtype=float)


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: Iterable[float], y_pred: Iterable[float], eps: float = 1e-6) -> float:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def wmape(y_true: Iterable[float], y_pred: Iterable[float], eps: float = 1e-6) -> float:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    denom = max(np.sum(np.abs(y_true)), eps)
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0)


def smape(y_true: Iterable[float], y_pred: Iterable[float], eps: float = 1e-6) -> float:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def r2(y_true: Iterable[float], y_pred: Iterable[float], eps: float = 1e-12) -> float:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= eps:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def mase(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    train_y: Iterable[float],
    seasonality: int = 1,
    eps: float = 1e-6,
) -> float:
    """Mean Absolute Scaled Error (Hyndman)."""

    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    train_y = _to_numpy(train_y)

    if len(train_y) <= seasonality:
        return float("nan")
    diff = np.abs(train_y[seasonality:] - train_y[:-seasonality])
    scale = max(np.mean(diff), eps)
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def pinball_loss(y_true: Iterable[float], y_pred: Iterable[float], quantile: float) -> float:
    """Quantile (pinball) loss for q in (0, 1)."""

    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    errors = y_true - y_pred
    return float(np.mean(np.maximum(quantile * errors, (quantile - 1.0) * errors)))


def interval_coverage(
    y_true: Iterable[float],
    y_lo: Iterable[float],
    y_hi: Iterable[float],
) -> float:
    y_true = _to_numpy(y_true)
    y_lo = _to_numpy(y_lo)
    y_hi = _to_numpy(y_hi)
    return float(np.mean((y_true >= y_lo) & (y_true <= y_hi)))


def interval_width(y_lo: Iterable[float], y_hi: Iterable[float]) -> float:
    y_lo = _to_numpy(y_lo)
    y_hi = _to_numpy(y_hi)
    return float(np.mean(np.maximum(0.0, y_hi - y_lo)))


DEFAULT_METRICS = {
    "RMSE": rmse,
    "MAE": mae,
    "MAPE": mape,
    "wMAPE": wmape,
    "SMAPE": smape,
    "R2": r2,
}


def evaluate_metrics(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    train_y: Optional[Iterable[float]] = None,
    seasonality: int = 1,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Return requested metrics (defaults to DEFAULT_METRICS + MASE if train is provided)."""

    metrics = metrics or DEFAULT_METRICS
    results: Dict[str, float] = {}
    for name, func in metrics.items():
        results[name] = func(y_true, y_pred)
    if train_y is not None:
        results["MASE"] = mase(y_true, y_pred, train_y, seasonality)
    return results


# ---------------------------------------------------------------------------
# Model adapters
# ---------------------------------------------------------------------------


@dataclass
class RidgeAdapter:
    """scikit-learn Ridge with CV over alpha grid."""

    alpha_grid: Tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    fit_intercept: bool = True
    _model: Any = None

    def fit(self, X, y):
        from sklearn.linear_model import RidgeCV

        self._model = RidgeCV(alphas=self.alpha_grid, fit_intercept=self.fit_intercept)
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)


@dataclass
class ElasticNetAdapter:
    """ElasticNet with cross-validated l1 ratio."""

    l1_ratio_grid: Tuple[float, ...] = (0.1, 0.5, 0.9)
    fit_intercept: bool = True
    _model: Any = None

    def fit(self, X, y):
        from sklearn.linear_model import ElasticNetCV

        self._model = ElasticNetCV(
            l1_ratio=self.l1_ratio_grid,
            cv=3,
            fit_intercept=self.fit_intercept,
            n_jobs=-1,
        )
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)


@dataclass
class ExtraTreesAdapter:
    """ExtraTreesRegressor with sensible defaults for time-series features."""

    n_estimators: int = 400
    max_depth: Optional[int] = None
    min_samples_leaf: int = 1
    n_jobs: int = -1
    random_state: int = 42
    _model: Any = None

    def fit(self, X, y):
        from sklearn.ensemble import ExtraTreesRegressor

        self._model = ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)


@dataclass
class LGBMAdapter:
    """LightGBM regressor wrapper for residual/hybrid pipelines."""

    n_estimators: int = 500
    learning_rate: float = 0.05
    num_leaves: int = 63
    max_depth: int = -1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    objective: str = "regression"
    tweedie_variance_power: Optional[float] = None
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    random_state: int = 42
    n_jobs: int = -1
    _model: Any = None

    def fit(self, X, y):
        from lightgbm import LGBMRegressor

        params = dict(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        if self.tweedie_variance_power is not None:
            params["objective"] = "tweedie"
            params["tweedie_variance_power"] = self.tweedie_variance_power
        else:
            params["objective"] = self.objective

        self._model = LGBMRegressor(**params)
        self._model.fit(X, y)
        return self

    def predict(self, X):
        if self._model is None:
            raise RuntimeError("LGBMAdapter must be fitted before calling predict().")
        return self._model.predict(X)


@dataclass
class XGBoostAdapter:
    """XGBoost regressor wrapper (works with numpy/pandas inputs)."""

    n_estimators: int = 800
    learning_rate: float = 0.05
    max_depth: int = 8
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    objective: str = "reg:squarederror"
    random_state: int = 42
    n_jobs: int = -1
    tree_method: Optional[str] = None
    _model: Any = None

    def fit(self, X, y):
        try:
            import xgboost as xgb  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dep
            raise ImportError("xgboost is required for XGBoostAdapter") from exc

        params = dict(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            objective=self.objective,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        if self.tree_method:
            params["tree_method"] = self.tree_method

        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32)
        self._model = xgb.XGBRegressor(**params)
        self._model.fit(X_np, y_np, verbose=False)
        return self

    def predict(self, X):
        if self._model is None:
            raise RuntimeError("XGBoostAdapter must be fitted before predict().")
        X_np = np.asarray(X, dtype=np.float32)
        return self._model.predict(X_np)


@dataclass
class NGBoostAdapter:
    """NGBoost regressor wrapper returning the predictive mean."""

    n_estimators: int = 600
    learning_rate: float = 0.03
    natural_gradient: bool = True
    minibatch_frac: Optional[float] = None
    verbose: bool = False
    random_state: int = 42
    Dist: Optional[Any] = None
    Score: Optional[Any] = None
    _model: Any = None

    def fit(self, X, y):
        try:
            from ngboost import NGBRegressor  # type: ignore
            from ngboost.distns import Normal  # type: ignore
            from ngboost.scores import MLE  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dep
            raise ImportError("ngboost is required for NGBoostAdapter") from exc

        kwargs = dict(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            natural_gradient=self.natural_gradient,
            verbose=self.verbose,
            random_state=self.random_state,
            Dist=self.Dist or Normal,
            Score=self.Score or MLE,
        )
        if self.minibatch_frac is not None:
            kwargs["minibatch_frac"] = self.minibatch_frac

        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32)
        self._model = NGBRegressor(**kwargs)
        # ngboost supports optional validation but falls back gracefully if not given
        self._model.fit(X_np, y_np)
        return self

    def predict(self, X):
        if self._model is None:
            raise RuntimeError("NGBoostAdapter must be fitted before predict().")
        X_np = np.asarray(X, dtype=np.float32)
        return self._model.predict(X_np)


class SARIMAXAdapter:
    """Lightweight statsmodels SARIMAX wrapper with optional exog filtering."""

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 0),
        seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
        exog_regex: Optional[str] = None,
        fit_kwargs: Optional[Dict[str, Any]] = None,
        **model_kwargs: Any,
    ) -> None:

        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.exog_regex = exog_regex
        self.model_kwargs = dict(model_kwargs) if model_kwargs else {}
        default_fit = {"disp": False, "maxiter": 50}
        self.fit_kwargs = {**default_fit, **(fit_kwargs or {})}
        self._res: Any = None
        self._exog_columns: Optional[Sequence[str]] = None

    def _ensure_dataframe(self, X: Any) -> Optional[pd.DataFrame]:
        if X is None:
            return None
        if isinstance(X, pd.DataFrame):
            return X
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            cols = [f"x{i}" for i in range(X.shape[1])]
            return pd.DataFrame(X, columns=cols)
        # Fallback: try to construct DataFrame
        return pd.DataFrame(X)

    def _select_exog(self, X_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if X_df is None or X_df.empty:
            return None
        if self.exog_regex is None:
            return X_df
        return X_df.filter(regex=self.exog_regex)

    def _prepare_exog(
        self, X_df: Optional[pd.DataFrame], *, store_columns: bool
    ) -> Optional[np.ndarray]:
        if X_df is None:
            return None
        exog_df = self._select_exog(X_df)
        if exog_df is None or exog_df.empty:
            return None
        if store_columns:
            self._exog_columns = list(exog_df.columns)
        elif self._exog_columns is not None:
            missing = [c for c in self._exog_columns if c not in exog_df.columns]
            if missing:
                for col in missing:
                    exog_df[col] = 0.0
            exog_df = exog_df.loc[:, list(self._exog_columns)]
        return np.asarray(exog_df, dtype=np.float64)

    def fit(self, X, y):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        y_arr = np.asarray(y, dtype=np.float64)
        X_df = self._ensure_dataframe(X)
        exog = self._prepare_exog(X_df, store_columns=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                endog=y_arr,
                exog=exog,
                order=self.order,
                seasonal_order=self.seasonal_order,
                **self.model_kwargs,
            )
            self._res = model.fit(**self.fit_kwargs)
        return self

    def predict(self, X):
        if self._res is None:
            raise RuntimeError("Model must be fitted before calling predict().")

        X_df = self._ensure_dataframe(X)
        steps = len(X_df) if X_df is not None else 1
        exog = self._prepare_exog(X_df, store_columns=False)
        forecast = self._res.forecast(steps=steps, exog=exog)
        if hasattr(forecast, "values"):
            return forecast.values
        return np.asarray(forecast, dtype=np.float64)


if torch is not None and nn is not None:

    class SimpleLSTM(nn.Module):
        """Βασικό LSTM δίκτυο για tabular features."""

        def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2, output_size: int = 1):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out.squeeze(-1)


    class SimpleGRU(nn.Module):
        """Βασικό GRU δίκτυο για tabular features (lighter than LSTM)."""

        def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2, output_size: int = 1):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out, _ = self.gru(x)
            out = self.fc(out[:, -1, :])
            return out.squeeze(-1)


    class NHITSBlock(nn.Module):
        """Simplified NHITS block producing backcast and forecast components."""

        def __init__(self, in_features: int, hidden_size: int, n_layers: int, dropout: float) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            last_dim = in_features
            for _ in range(max(1, n_layers)):
                layers.append(nn.Linear(last_dim, hidden_size))
                layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
                last_dim = hidden_size
            self.mlp = nn.Sequential(*layers)
            self.backcast = nn.Linear(hidden_size, in_features)
            self.forecast = nn.Linear(hidden_size, 1)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            h = self.mlp(x)
            backcast = self.backcast(h)
            forecast = self.forecast(h)
            return backcast, forecast


    class SimpleNHITS(nn.Module):
        """Stacked NHITS blocks accumulating hierarchical forecasts."""

        def __init__(
            self,
            in_features: int,
            hidden_size: int,
            n_blocks: int,
            n_layers: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.blocks = nn.ModuleList(
                NHITSBlock(in_features, hidden_size, n_layers, dropout)
                for _ in range(max(1, n_blocks))
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            forecast_total = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
            for block in self.blocks:
                backcast, forecast = block(residual)
                forecast_total = forecast_total + forecast
                residual = residual - backcast
            return forecast_total.squeeze(-1)

else:

    class SimpleLSTM:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for LSTMAdapter (pip install torch)")


    class SimpleNHITS:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for NHITSAdapter (pip install torch)")


class LSTMAdapter:
    """PyTorch LSTM adapter με windowed sequences, early stopping και scaling."""

    def __init__(
        self,
        seq_len: int = 48,
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 256,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        seed: int = 42,
        patience: int = 5,
        max_windows: int = 20000,
    ) -> None:
        if torch is None or nn is None or DataLoader is None or TensorDataset is None:
            raise ImportError("PyTorch is required for LSTMAdapter (pip install torch)")

        self.seq_len = int(seq_len)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.seed = int(seed)
        self.patience = max(1, int(patience))
        self.max_windows = max(1000, int(max_windows))

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()  # NEW: Scale target too for better training

        self.model: Optional[SimpleLSTM] = None
        self._effective_seq_len: int = self.seq_len
        self._fallback_value: Optional[float] = None
        self._is_fitted = False

    def _set_seed(self) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    @staticmethod
    def _build_windows_from_indices(X: np.ndarray, seq_len: int, idx: np.ndarray) -> np.ndarray:
        if idx.size == 0:
            return np.empty((0, seq_len, X.shape[1]), dtype=np.float32)
        return np.stack([X[i - seq_len : i] for i in idx], axis=0).astype(np.float32, copy=False)

    def fit(self, X, y):
        self._set_seed()

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).ravel()
        if X_arr.ndim != 2:
            raise ValueError("LSTMAdapter expects 2D feature arrays")
        if len(X_arr) <= 1:
            self._fallback_value = float(np.mean(y_arr)) if len(y_arr) else 0.0
            self.model = None
            self._is_fitted = True
            return self

        # Scale features
        self.scaler.fit(X_arr)
        X_scaled = self.scaler.transform(X_arr)

        # NEW: Scale target for better training convergence (critical for high-value datasets)
        self.target_scaler.fit(y_arr.reshape(-1, 1))
        y_scaled = self.target_scaler.transform(y_arr.reshape(-1, 1)).ravel().astype(np.float32)

        self._effective_seq_len = min(self.seq_len, max(1, len(X_arr) - 1))
        total_points = X_scaled.shape[0]
        if total_points <= self._effective_seq_len:
            self._fallback_value = float(np.mean(y_arr))
            self.model = None
            self._is_fitted = True
            return self

        idx_end = np.arange(self._effective_seq_len, total_points)
        if idx_end.size > self.max_windows:
            idx_end = np.linspace(
                self._effective_seq_len,
                total_points - 1,
                num=self.max_windows,
                dtype=int,
            )

        Xw = self._build_windows_from_indices(X_scaled, self._effective_seq_len, idx_end)
        yw = y_scaled[idx_end]  # Use scaled targets

        if len(Xw) == 0 or len(yw) == 0:
            self._fallback_value = float(np.mean(y_arr))
            self.model = None
            self._is_fitted = True
            return self

        split_idx = int(len(Xw) * 0.9)
        if len(Xw) > 1:
            split_idx = min(max(split_idx, 1), len(Xw) - 1)
        else:
            split_idx = len(Xw)

        Xtr, ytr = Xw[:split_idx], yw[:split_idx]
        Xva, yva = Xw[split_idx:], yw[split_idx:]


        train_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
        drop_last = len(Xtr) > self.batch_size
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=drop_last,
        )

        val_loader = None
        if len(Xva) > 0:
            val_ds = TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva))
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        input_size = X_arr.shape[1]
        self.model = SimpleLSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        best_state = None
        best_val = float("inf")
        no_improve = 0

        for _ in range(self.epochs):
            self.model.train()
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

            if val_loader is None:
                continue

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    preds = self.model(xb)
                    val_losses.append(criterion(preds, yb).item())

            val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
            if val_loss + 1e-6 < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self._fallback_value = None
        self._is_fitted = True
        return self

    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError("LSTMAdapter must be fitted before calling predict().")
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2:
            raise ValueError("LSTMAdapter expects 2D feature arrays")
        if self.model is None or self._fallback_value is not None:
            fallback = self._fallback_value if self._fallback_value is not None else 0.0
            return np.full(len(X_arr), fallback, dtype=np.float32)

        X_scaled = self.scaler.transform(X_arr)
        seq_len = max(1, self._effective_seq_len)
        total_points = X_scaled.shape[0]
        if total_points <= seq_len:
            fallback = self._fallback_value if self._fallback_value is not None else 0.0
            return np.full(len(X_arr), fallback, dtype=np.float32)

        idx_end = np.arange(seq_len, total_points)
        windows = self._build_windows_from_indices(X_scaled, seq_len, idx_end)

        if len(windows) == 0:
            fallback = self._fallback_value if self._fallback_value is not None else 0.0
            return np.full(len(X_arr), fallback, dtype=np.float32)

        pred_ds = TensorDataset(torch.from_numpy(windows))
        pred_loader = DataLoader(pred_ds, batch_size=self.batch_size, shuffle=False)

        preds = []
        self.model.eval()
        with torch.no_grad():
            for (xb,) in pred_loader:
                xb = xb.to(self.device)
                preds.append(self.model(xb).detach().cpu().numpy().astype(np.float32))

        preds_arr = np.concatenate(preds, axis=0) if preds else np.zeros(0, dtype=np.float32)

        # NEW: Inverse transform to get predictions in original scale
        preds_arr = self.target_scaler.inverse_transform(preds_arr.reshape(-1, 1)).ravel().astype(np.float32)

        pad_value = preds_arr[0] if preds_arr.size else (
            self._fallback_value if self._fallback_value is not None else 0.0
        )
        padding = np.full(seq_len, pad_value, dtype=np.float32)
        full = np.concatenate([padding, preds_arr], axis=0)
        return full[: len(X_arr)]


class GRUAdapter:
    """PyTorch GRU adapter (lighter alternative to LSTM) με windowed sequences."""

    def __init__(
        self,
        seq_len: int = 48,
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 256,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        seed: int = 42,
        patience: int = 5,
        max_windows: int = 20000,
    ) -> None:
        if torch is None or nn is None or DataLoader is None or TensorDataset is None:
            raise ImportError("PyTorch is required for GRUAdapter (pip install torch)")

        self.seq_len = int(seq_len)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.seed = int(seed)
        self.patience = max(1, int(patience))
        self.max_windows = max(1000, int(max_windows))

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()  # NEW: Scale target too

        self.model: Optional[SimpleGRU] = None
        self._effective_seq_len: int = self.seq_len
        self._fallback_value: Optional[float] = None
        self._is_fitted = False

    def _set_seed(self) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    @staticmethod
    def _build_windows_from_indices(X: np.ndarray, seq_len: int, idx: np.ndarray) -> np.ndarray:
        if idx.size == 0:
            return np.empty((0, seq_len, X.shape[1]), dtype=np.float32)
        return np.stack([X[i - seq_len : i] for i in idx], axis=0).astype(np.float32, copy=False)

    def fit(self, X, y):
        self._set_seed()

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).ravel()
        if X_arr.ndim != 2:
            raise ValueError("GRUAdapter expects 2D feature arrays")
        if len(X_arr) <= 1:
            self._fallback_value = float(np.mean(y_arr)) if len(y_arr) else 0.0
            self.model = None
            self._is_fitted = True
            return self

        # Scale features
        self.scaler.fit(X_arr)
        X_scaled = self.scaler.transform(X_arr)

        # NEW: Scale target for better training convergence
        self.target_scaler.fit(y_arr.reshape(-1, 1))
        y_scaled = self.target_scaler.transform(y_arr.reshape(-1, 1)).ravel().astype(np.float32)

        self._effective_seq_len = min(self.seq_len, max(1, len(X_arr) - 1))
        total_points = X_scaled.shape[0]
        if total_points <= self._effective_seq_len:
            self._fallback_value = float(np.mean(y_arr))
            self.model = None
            self._is_fitted = True
            return self

        idx_end = np.arange(self._effective_seq_len, total_points)
        if idx_end.size > self.max_windows:
            idx_end = np.linspace(
                self._effective_seq_len,
                total_points - 1,
                num=self.max_windows,
                dtype=int,
            )

        Xw = self._build_windows_from_indices(X_scaled, self._effective_seq_len, idx_end)
        yw = y_scaled[idx_end]  # Use scaled targets

        if len(Xw) == 0 or len(yw) == 0:
            self._fallback_value = float(np.mean(y_arr))
            self.model = None
            self._is_fitted = True
            return self

        split_idx = int(len(Xw) * 0.9)
        if len(Xw) > 1:
            split_idx = min(max(split_idx, 1), len(Xw) - 1)
        else:
            split_idx = len(Xw)

        Xtr, ytr = Xw[:split_idx], yw[:split_idx]
        Xva, yva = Xw[split_idx:], yw[split_idx:]

        train_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
        drop_last = len(Xtr) > self.batch_size
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=drop_last,
        )

        val_loader = None
        if len(Xva) > 0:
            val_ds = TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva))
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        input_size = X_arr.shape[1]
        self.model = SimpleGRU(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        best_state = None
        best_val = float("inf")
        no_improve = 0

        for _ in range(self.epochs):
            self.model.train()
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

            if val_loader is None:
                continue

            self.model.eval()
            val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    preds = self.model(xb)
                    loss = criterion(preds, yb)
                    val_loss += loss.item() * len(xb)
                    val_count += len(xb)

            val_loss = val_loss / max(val_count, 1)
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self._fallback_value = float(np.mean(y_arr))
        self._is_fitted = True
        return self

    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError("GRUAdapter must be fitted before predict().")

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2:
            raise ValueError("GRUAdapter expects 2D feature arrays")
        if len(X_arr) == 0:
            return np.zeros(0, dtype=np.float32)

        if self.model is None:
            return np.full(len(X_arr), self._fallback_value or 0.0, dtype=np.float32)

        X_scaled = self.scaler.transform(X_arr)
        seq_len = self._effective_seq_len
        total_pts = X_scaled.shape[0]

        idx_end = np.arange(seq_len, total_pts)
        if idx_end.size == 0:
            return np.full(total_pts, self._fallback_value or 0.0, dtype=np.float32)

        Xw = self._build_windows_from_indices(X_scaled, seq_len, idx_end)
        pred_ds = TensorDataset(torch.from_numpy(Xw))
        pred_loader = DataLoader(pred_ds, batch_size=self.batch_size, shuffle=False)

        preds = []
        self.model.eval()
        with torch.no_grad():
            for (xb,) in pred_loader:
                xb = xb.to(self.device)
                preds.append(self.model(xb).detach().cpu().numpy().astype(np.float32))

        preds_arr = np.concatenate(preds, axis=0) if preds else np.zeros(0, dtype=np.float32)

        # NEW: Inverse transform to get predictions in original scale
        preds_arr = self.target_scaler.inverse_transform(preds_arr.reshape(-1, 1)).ravel().astype(np.float32)

        pad_value = preds_arr[0] if preds_arr.size else (
            self._fallback_value if self._fallback_value is not None else 0.0
        )
        padding = np.full(seq_len, pad_value, dtype=np.float32)
        full = np.concatenate([padding, preds_arr], axis=0)
        return full[: len(X_arr)]


class SimpleBiLSTM(nn.Module):
    """Bidirectional LSTM για time-series features (captures past + future context)."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2, output_size: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,  # KEY: bidirectional=True
        )
        # Output από BiLSTM είναι 2 * hidden_size (forward + backward)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        out, _ = self.bilstm(x)
        out = self.fc(out[:, -1, :])  # Take last timestep
        return out.squeeze(-1)


class BiLSTMAdapter:
    """PyTorch BiLSTM adapter (bidirectional LSTM) με windowed sequences."""

    def __init__(
        self,
        seq_len: int = 48,
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 256,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        seed: int = 42,
        patience: int = 5,
        max_windows: int = 20000,
    ) -> None:
        if torch is None or nn is None or DataLoader is None or TensorDataset is None:
            raise ImportError("PyTorch is required for BiLSTMAdapter (pip install torch)")

        self.seq_len = int(seq_len)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.seed = int(seed)
        self.patience = max(1, int(patience))
        self.max_windows = max(1000, int(max_windows))

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()  # NEW: Scale target too

        self.model: Optional[SimpleBiLSTM] = None
        self._effective_seq_len: int = self.seq_len
        self._fallback_value: Optional[float] = None
        self._is_fitted = False

    def _set_seed(self) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    @staticmethod
    def _build_windows_from_indices(X: np.ndarray, seq_len: int, idx: np.ndarray) -> np.ndarray:
        if idx.size == 0:
            return np.empty((0, seq_len, X.shape[1]), dtype=np.float32)
        return np.stack([X[i - seq_len : i] for i in idx], axis=0).astype(np.float32, copy=False)

    def fit(self, X, y):
        self._set_seed()

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).ravel()
        if X_arr.ndim != 2:
            raise ValueError("BiLSTMAdapter expects 2D feature arrays")
        if len(X_arr) <= 1:
            self._fallback_value = float(np.mean(y_arr)) if len(y_arr) else 0.0
            self.model = None
            self._is_fitted = True
            return self

        # Scale features
        self.scaler.fit(X_arr)
        X_scaled = self.scaler.transform(X_arr)

        # NEW: Scale target for better training convergence
        self.target_scaler.fit(y_arr.reshape(-1, 1))
        y_scaled = self.target_scaler.transform(y_arr.reshape(-1, 1)).ravel().astype(np.float32)

        self._effective_seq_len = min(self.seq_len, max(1, len(X_arr) - 1))
        total_points = X_scaled.shape[0]
        if total_points <= self._effective_seq_len:
            self._fallback_value = float(np.mean(y_arr))
            self.model = None
            self._is_fitted = True
            return self

        idx_end = np.arange(self._effective_seq_len, total_points)
        if idx_end.size > self.max_windows:
            idx_end = np.linspace(
                self._effective_seq_len,
                total_points - 1,
                num=self.max_windows,
                dtype=int,
            )

        Xw = self._build_windows_from_indices(X_scaled, self._effective_seq_len, idx_end)
        yw = y_scaled[idx_end]  # Use scaled targets

        if len(Xw) == 0 or len(yw) == 0:
            self._fallback_value = float(np.mean(y_arr))
            self.model = None
            self._is_fitted = True
            return self

        split_idx = int(len(Xw) * 0.9)
        if len(Xw) > 1:
            split_idx = min(max(split_idx, 1), len(Xw) - 1)
        else:
            split_idx = len(Xw)

        Xtr, ytr = Xw[:split_idx], yw[:split_idx]
        Xva, yva = Xw[split_idx:], yw[split_idx:]

        train_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
        drop_last = len(Xtr) > self.batch_size
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=drop_last,
        )

        val_loader = None
        if len(Xva) > 0:
            val_ds = TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva))
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        input_size = X_arr.shape[1]
        self.model = SimpleBiLSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        best_state = None
        best_val = float("inf")
        no_improve = 0

        for _ in range(self.epochs):
            self.model.train()
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

            if val_loader is None:
                continue

            self.model.eval()
            val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    preds = self.model(xb)
                    loss = criterion(preds, yb)
                    val_loss += loss.item() * len(xb)
                    val_count += len(xb)

            val_loss = val_loss / max(val_count, 1)
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self._fallback_value = float(np.mean(y_arr))
        self._is_fitted = True
        return self

    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError("BiLSTMAdapter must be fitted before predict().")

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2:
            raise ValueError("BiLSTMAdapter expects 2D feature arrays")
        if len(X_arr) == 0:
            return np.zeros(0, dtype=np.float32)

        if self.model is None:
            return np.full(len(X_arr), self._fallback_value or 0.0, dtype=np.float32)

        X_scaled = self.scaler.transform(X_arr)
        seq_len = self._effective_seq_len
        total_pts = X_scaled.shape[0]

        idx_end = np.arange(seq_len, total_pts)
        if idx_end.size == 0:
            return np.full(total_pts, self._fallback_value or 0.0, dtype=np.float32)

        Xw = self._build_windows_from_indices(X_scaled, seq_len, idx_end)
        pred_ds = TensorDataset(torch.from_numpy(Xw))
        pred_loader = DataLoader(pred_ds, batch_size=self.batch_size, shuffle=False)

        preds = []
        self.model.eval()
        with torch.no_grad():
            for (xb,) in pred_loader:
                xb = xb.to(self.device)
                preds.append(self.model(xb).detach().cpu().numpy().astype(np.float32))

        preds_arr = np.concatenate(preds, axis=0) if preds else np.zeros(0, dtype=np.float32)

        # NEW: Inverse transform to get predictions in original scale
        preds_arr = self.target_scaler.inverse_transform(preds_arr.reshape(-1, 1)).ravel().astype(np.float32)

        pad_value = preds_arr[0] if preds_arr.size else (
            self._fallback_value if self._fallback_value is not None else 0.0
        )
        padding = np.full(seq_len, pad_value, dtype=np.float32)
        full = np.concatenate([padding, preds_arr], axis=0)
        return full[: len(X_arr)]


class NHITSAdapter:
    """NHITS-inspired deep model for direct horizon regression on engineered features."""

    def __init__(
        self,
        hidden_size: int = 512,
        n_blocks: int = 3,
        n_layers: int = 2,
        dropout: float = 0.1,
        epochs: int = 200,
        batch_size: int = 512,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        loss: str = "mape",
        patience: int = 12,
        validation_fraction: float = 0.1,
        device: str = "auto",
        seed: int = 42,
        mape_eps: float = 1e-3,
        smape_eps: float = 1e-3,
    ) -> None:
        if torch is None or nn is None or DataLoader is None or TensorDataset is None:
            raise ImportError("PyTorch is required for NHITSAdapter (pip install torch)")

        self.hidden_size = int(hidden_size)
        self.n_blocks = max(1, int(n_blocks))
        self.n_layers = max(1, int(n_layers))
        self.dropout = float(dropout)
        self.epochs = max(1, int(epochs))
        self.batch_size = max(1, int(batch_size))
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.loss_name = loss.lower()
        self.patience = max(1, int(patience))
        self.validation_fraction = float(np.clip(validation_fraction, 0.0, 0.4))
        self.seed = int(seed)
        self.mape_eps = float(mape_eps)
        self.smape_eps = float(smape_eps)

        device = (device or "cpu").lower()
        if device == "auto":
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        elif device in {"cpu", "cuda", "mps"}:
            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA ζητήθηκε αλλά δεν είναι διαθέσιμη στο σύστημα")
            if device == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
                raise RuntimeError("MPS ζητήθηκε αλλά δεν είναι διαθέσιμο στο σύστημα")
            self.device = torch.device(device)
        else:
            raise ValueError(f"Unsupported device '{device}'. Επιλογές: cpu, cuda, mps, auto")

        self.scaler = StandardScaler()
        self.model: Optional[SimpleNHITS] = None
        self._is_fitted = False
        self._fallback_value: Optional[float] = None
        self._best_state: Optional[dict[str, torch.Tensor]] = None
        self._input_dim: Optional[int] = None

        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

    def _set_seed(self) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    @staticmethod
    def _ensure_2d(X: np.ndarray) -> np.ndarray:
        return X.reshape(-1, 1) if X.ndim == 1 else X

    def _compute_loss(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_name == "mape":
            denom = torch.clamp(target.abs(), min=self.mape_eps)
            return torch.mean(torch.abs((target - preds) / denom))
        if self.loss_name == "smape":
            denom = torch.clamp((target.abs() + preds.abs()) * 0.5, min=self.smape_eps)
            return torch.mean(torch.abs(target - preds) / denom)
        if self.loss_name == "mae":
            return torch.mean(torch.abs(target - preds))
        if self.loss_name == "huber":
            return torch.nn.functional.smooth_l1_loss(preds, target)
        return torch.mean((target - preds) ** 2)

    def _prepare_loaders(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        train_loader = DataLoader(
            train_ds,
            batch_size=min(self.batch_size, max(1, len(train_ds))),
            shuffle=True,
            drop_last=False,
        )

        val_loader = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
            val_loader = DataLoader(
                val_ds,
                batch_size=min(self.batch_size, max(1, len(val_ds))),
                shuffle=False,
                drop_last=False,
            )
        return train_loader, val_loader

    def _split_train_val(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        if X_val is not None and y_val is not None and len(X_val) > 0:
            return X, y, X_val, y_val

        if len(X) < 10 or self.validation_fraction <= 0:
            return X, y, None, None

        val_size = max(1, int(len(X) * self.validation_fraction))
        if val_size >= len(X):
            return X, y, None, None

        return (
            X[:-val_size],
            y[:-val_size],
            X[-val_size:],
            y[-val_size:],
        )

    def fit(self, X, y, X_val=None, y_val=None):
        self._set_seed()

        X_train = self._ensure_2d(np.asarray(X, dtype=np.float32))
        y_train = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        if len(X_train) == 0:
            self._fallback_value = 0.0
            self.model = None
            self._is_fitted = True
            return self

        if len(X_train) <= 3:
            self._fallback_value = float(np.mean(y_train))
            self.model = None
            self._is_fitted = True
            return self

        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train).astype(np.float32)
        X_val_scaled = None
        y_val_arr = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            X_val_scaled = self._ensure_2d(np.asarray(X_val, dtype=np.float32))
            X_val_scaled = self.scaler.transform(X_val_scaled).astype(np.float32)
            y_val_arr = np.asarray(y_val, dtype=np.float32).reshape(-1, 1)

        Xtr, ytr, Xva, yva = self._split_train_val(X_train_scaled, y_train, X_val_scaled, y_val_arr)

        Xtr_t = Xtr.astype(np.float32, copy=False)
        ytr_t = ytr.astype(np.float32, copy=False)
        Xva_t = Xva.astype(np.float32, copy=False) if Xva is not None else None
        yva_t = yva.astype(np.float32, copy=False) if yva is not None else None

        self._input_dim = Xtr_t.shape[1]
        self.model = SimpleNHITS(
            in_features=self._input_dim,
            hidden_size=self.hidden_size,
            n_blocks=self.n_blocks,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        train_loader, val_loader = self._prepare_loaders(Xtr_t, ytr_t, Xva_t, yva_t)

        best_loss = float("inf")
        best_state = None
        epochs_since_improve = 0

        for epoch in range(self.epochs):
            self.model.train()
            epoch_losses = []
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                preds = self.model(xb).unsqueeze(-1)
                loss = self._compute_loss(preds, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_losses.append(loss.detach().cpu().item())

            val_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")

            if val_loader is not None:
                self.model.eval()
                losses = []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(self.device)
                        yb = yb.to(self.device)
                        preds = self.model(xb).unsqueeze(-1)
                        losses.append(self._compute_loss(preds, yb).cpu().item())
                if losses:
                    val_loss = float(np.mean(losses))

            if val_loss + 1e-6 < best_loss:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= self.patience:
                    break

        if best_state is not None:
            self._best_state = best_state
            self.model.load_state_dict(best_state)
            self._fallback_value = None
        else:
            self._fallback_value = float(np.mean(y_train))
            self.model = None

        self._is_fitted = True
        return self

    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError("NHITSAdapter must be fitted before calling predict().")

        X_arr = self._ensure_2d(np.asarray(X, dtype=np.float32))
        if len(X_arr) == 0:
            return np.asarray([], dtype=np.float32)

        if self.model is None or self._fallback_value is not None:
            fallback = self._fallback_value if self._fallback_value is not None else 0.0
            return np.full(len(X_arr), fallback, dtype=np.float32)

        if self._input_dim is None:
            raise RuntimeError("NHITSAdapter fitted state is corrupted (missing input dim)")

        X_scaled = self.scaler.transform(X_arr).astype(np.float32, copy=False)
        pred_ds = TensorDataset(torch.from_numpy(X_scaled))
        pred_loader = DataLoader(
            pred_ds,
            batch_size=min(self.batch_size, max(1, len(pred_ds))),
            shuffle=False,
            drop_last=False,
        )

        preds: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for (xb,) in pred_loader:
                xb = xb.to(self.device)
                out = self.model(xb)
                preds.append(out.detach().cpu().numpy().astype(np.float32))

        if not preds:
            fallback = self._fallback_value if self._fallback_value is not None else 0.0
            return np.full(len(X_arr), fallback, dtype=np.float32)

        return np.concatenate(preds, axis=0).ravel()


if torch is not None and nn is not None:

    class SimplePatchTST(nn.Module):
        """Απλοποιημένη υλοποίηση PatchTST (patch embedding + Transformer encoder)."""

        def __init__(
            self,
            input_size: int,
            patch_len: int = 16,
            stride: int = 8,
            d_model: int = 128,
            n_heads: int = 8,
            n_layers: int = 2,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.input_size = input_size
            self.patch_len = patch_len
            self.stride = stride
            self.d_model = d_model
            self.patch_embed = nn.Linear(input_size * patch_len, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, 1)
            self.dropout = nn.Dropout(dropout)

        @staticmethod
        def _positional_encoding(length: int, dim: int, device: torch.device) -> torch.Tensor:
            position = torch.arange(length, device=device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))
            pe = torch.zeros(length, dim, device=device)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            return pe.unsqueeze(0)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, seq_len, input_dim)
            if x.size(1) < self.patch_len:
                raise ValueError("Sequence length smaller than patch length")

            patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
            # patches: (batch, num_patches, patch_len, input_dim)
            batch, num_patches, *_ = patches.shape
            patches = patches.contiguous().view(batch, num_patches, -1)
            tokens = self.patch_embed(patches)
            tokens = tokens + self._positional_encoding(num_patches, self.d_model, tokens.device)
            tokens = self.dropout(tokens)
            encoded = self.transformer(tokens)
            encoded = self.norm(encoded[:, -1, :])  # χρήση της τελευταίας αναπαράστασης
            out = self.head(encoded)
            return out.squeeze(-1)

else:

    class SimplePatchTST:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for PatchTSTAdapter (pip install torch)")


class PatchTSTAdapter:
    """Lightweight PatchTST adapter με windowed είσοδο και early stopping."""

    def __init__(
        self,
        seq_len: int = 96,
        patch_len: int = 16,
        stride: int = 8,
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 256,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        seed: int = 42,
        patience: int = 6,
        max_windows: int = 25000,
        device: str = "cpu",
    ) -> None:
        if torch is None or nn is None or DataLoader is None or TensorDataset is None:
            raise ImportError("PyTorch is required for PatchTSTAdapter (pip install torch)")

        self.seq_len = int(seq_len)
        self.patch_len = int(patch_len)
        self.stride = max(1, int(stride))
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.dropout = float(dropout)
        self.seed = int(seed)
        self.patience = max(1, int(patience))
        self.max_windows = max(1000, int(max_windows))

        device = (device or "cpu").lower()
        if device == "auto":
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        elif device in {"cpu", "cuda", "mps"}:
            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA ζητήθηκε αλλά δεν είναι διαθέσιμη στο σύστημα")
            if device == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
                raise RuntimeError("MPS ζητήθηκε αλλά δεν είναι διαθέσιμο στο σύστημα")
            self.device = torch.device(device)
        else:
            raise ValueError(f"Unsupported device '{device}'. Επιλογές: cpu, cuda, mps, auto")

        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()  # NEW: Scale target too
        self.model: Optional[SimplePatchTST] = None
        self._effective_seq_len: int = self.seq_len
        self._effective_patch_len: int = self.patch_len
        self._fallback_value: Optional[float] = None
        self._is_fitted = False

        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

    def _set_seed(self) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    @staticmethod
    def _build_windows_for_indices(X: np.ndarray, seq_len: int, idx: np.ndarray) -> np.ndarray:
        if idx.size == 0:
            return np.empty((0, seq_len, X.shape[1]), dtype=np.float32)
        return np.stack([X[i - seq_len : i] for i in idx], axis=0).astype(np.float32, copy=False)

    def fit(self, X, y):
        self._set_seed()

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).ravel()
        if X_arr.ndim != 2:
            raise ValueError("PatchTSTAdapter expects 2D feature arrays")
        if len(X_arr) <= 1:
            self._fallback_value = float(np.mean(y_arr)) if len(y_arr) else 0.0
            self.model = None
            self._is_fitted = True
            return self

        # Scale features
        self.scaler.fit(X_arr)
        X_scaled = self.scaler.transform(X_arr)

        # NEW: Scale target for better training convergence
        self.target_scaler.fit(y_arr.reshape(-1, 1))
        y_scaled = self.target_scaler.transform(y_arr.reshape(-1, 1)).ravel().astype(np.float32)

        self._effective_seq_len = min(self.seq_len, max(1, len(X_arr) - 1))
        self._effective_patch_len = min(self.patch_len, self._effective_seq_len)
        if self._effective_patch_len <= 0:
            self._fallback_value = float(np.mean(y_arr))
            self.model = None
            self._is_fitted = True
            return self

        total_points = X_scaled.shape[0]
        if total_points <= self._effective_seq_len:
            self._fallback_value = float(np.mean(y_arr))
            self.model = None
            self._is_fitted = True
            return self

        idx_end = np.arange(self._effective_seq_len, total_points)
        if idx_end.size > self.max_windows:
            idx_end = np.linspace(
                self._effective_seq_len,
                total_points - 1,
                num=self.max_windows,
                dtype=int,
            )

        Xw = self._build_windows_for_indices(X_scaled, self._effective_seq_len, idx_end)
        yw = y_scaled[idx_end]  # Use scaled targets

        if len(Xw) == 0 or len(yw) == 0:
            self._fallback_value = float(np.mean(y_arr))
            self.model = None
            self._is_fitted = True
            return self

        split_idx = int(len(Xw) * 0.9)
        if len(Xw) > 1:
            split_idx = min(max(split_idx, 1), len(Xw) - 1)
        else:
            split_idx = len(Xw)

        Xtr, ytr = Xw[:split_idx], yw[:split_idx]
        Xva, yva = Xw[split_idx:], yw[split_idx:]

        train_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
        drop_last = len(Xtr) > self.batch_size
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=drop_last,
        )


        val_loader = None
        if len(Xva) > 0:
            val_ds = TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva))
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        input_size = X_arr.shape[1]
        self.model = SimplePatchTST(
            input_size=input_size,
            patch_len=self._effective_patch_len,
            stride=min(self.stride, max(1, self._effective_patch_len)),
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        best_state = None
        best_val = float("inf")
        no_improve = 0

        for _ in range(self.epochs):
            self.model.train()
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

            if val_loader is None:
                continue

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    preds = self.model(xb)
                    val_losses.append(criterion(preds, yb).item())

            val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
            if val_loss + 1e-6 < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        if best_state is not None and self.model is not None:
            self.model.load_state_dict(best_state)

        self._fallback_value = None
        self._is_fitted = True
        return self

    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError("PatchTSTAdapter must be fitted before calling predict().")

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2:
            raise ValueError("PatchTSTAdapter expects 2D feature arrays")

        if self.model is None or self._fallback_value is not None:
            fallback = self._fallback_value if self._fallback_value is not None else 0.0
            return np.full(len(X_arr), fallback, dtype=np.float32)

        X_scaled = self.scaler.transform(X_arr)
        seq_len = max(1, self._effective_seq_len)
        total_points = X_scaled.shape[0]
        if total_points <= seq_len:
            fallback = self._fallback_value if self._fallback_value is not None else 0.0
            return np.full(len(X_arr), fallback, dtype=np.float32)

        idx_end = np.arange(seq_len, total_points)
        windows = self._build_windows_for_indices(X_scaled, seq_len, idx_end)

        pred_ds = TensorDataset(torch.from_numpy(windows))
        pred_loader = DataLoader(pred_ds, batch_size=self.batch_size, shuffle=False)

        preds = []
        self.model.eval()
        with torch.no_grad():
            for (xb,) in pred_loader:
                xb = xb.to(self.device)
                preds.append(self.model(xb).detach().cpu().numpy().astype(np.float32))

        preds_arr = np.concatenate(preds, axis=0) if preds else np.zeros(0, dtype=np.float32)

        # NEW: Inverse transform to get predictions in original scale
        preds_arr = self.target_scaler.inverse_transform(preds_arr.reshape(-1, 1)).ravel().astype(np.float32)

        pad_value = preds_arr[0] if preds_arr.size else (
            self._fallback_value if self._fallback_value is not None else 0.0
        )
        padding = np.full(seq_len, pad_value, dtype=np.float32)
        full = np.concatenate([padding, preds_arr], axis=0)
        return full[: len(X_arr)]


if torch is not None:
    class SimplifiedTFT(nn.Module):
        """Simplified Temporal Fusion Transformer with key components:
        - Variable selection (learnable feature weights)
        - LSTM temporal processing
        - Multi-head attention
        - Gated residual connections
        """

        def __init__(self, input_size, hidden_size=64, num_heads=4, num_layers=2, dropout=0.1):
            super().__init__()

            self.input_size = input_size
            self.hidden_size = hidden_size

            # Variable selection network (learnable feature importance)
            self.variable_selection = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, input_size),
                nn.Softmax(dim=-1)
            )

            # LSTM encoder for temporal processing
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )

            # Multi-head attention
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )

            # Gated residual connection (GLU - Gated Linear Unit)
            self.gate = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Sigmoid()
            )

            # Layer normalization
            self.layer_norm1 = nn.LayerNorm(hidden_size)
            self.layer_norm2 = nn.LayerNorm(hidden_size)

            # Final output projection
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1)
            )

        def forward(self, x):
            # x shape: (batch, seq_len, input_size)
            batch_size, seq_len, _ = x.shape

            # Variable selection: apply learnable weights to features
            # Average across time for variable importance
            x_mean = x.mean(dim=1)  # (batch, input_size)
            var_weights = self.variable_selection(x_mean)  # (batch, input_size)
            var_weights = var_weights.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, input_size)
            x_selected = x * var_weights  # Weighted features

            # LSTM temporal encoding
            lstm_out, _ = self.lstm(x_selected)  # (batch, seq_len, hidden)

            # Self-attention
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)  # (batch, seq_len, hidden)

            # Gated residual connection
            gate_values = self.gate(attn_out)
            gated = gate_values * attn_out + (1 - gate_values) * lstm_out
            gated = self.layer_norm1(gated)

            # Use last timestep output
            final_hidden = gated[:, -1, :]  # (batch, hidden)
            final_hidden = self.layer_norm2(final_hidden)

            # Final projection
            output = self.output_layer(final_hidden)  # (batch, 1)
            return output.squeeze(-1)
else:
    class SimplifiedTFT:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for TFT")


class TFTAdapter:
    """Temporal Fusion Transformer adapter for interpretable multi-horizon forecasting."""

    def __init__(
        self,
        seq_len: int = 48,
        hidden_size: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 256,
        seed: int = 42,
        patience: int = 7,
        max_windows: int = 20000,
    ):
        if torch is None or nn is None:
            raise ImportError("PyTorch required for TFT (pip install torch)")

        self.seq_len = int(seq_len)
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.patience = max(1, int(patience))
        self.max_windows = max(1000, int(max_windows))

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except:
            pass

        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()  # NEW: Scale target too
        self.model = None
        self._effective_seq_len = self.seq_len
        self._fallback_value = None
        self._is_fitted = False

    def _set_seed(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    @staticmethod
    def _build_windows(X, seq_len, idx):
        if idx.size == 0:
            return np.empty((0, seq_len, X.shape[1]), dtype=np.float32)
        return np.stack([X[i - seq_len : i] for i in idx], axis=0).astype(np.float32)

    def fit(self, X, y):
        self._set_seed()

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).ravel()

        if X_arr.ndim != 2:
            raise ValueError("TFTAdapter expects 2D feature arrays")
        if len(X_arr) <= 1:
            self._fallback_value = float(np.mean(y_arr)) if len(y_arr) else 0.0
            self.model = None
            self._is_fitted = True
            return self

        # Scale features
        self.scaler.fit(X_arr)
        X_scaled = self.scaler.transform(X_arr)

        # NEW: Scale target for better training convergence
        self.target_scaler.fit(y_arr.reshape(-1, 1))
        y_scaled = self.target_scaler.transform(y_arr.reshape(-1, 1)).ravel().astype(np.float32)

        self._effective_seq_len = min(self.seq_len, max(1, len(X_arr) - 1))
        total_points = X_scaled.shape[0]

        if total_points <= self._effective_seq_len:
            self._fallback_value = float(np.mean(y_arr))
            self.model = None
            self._is_fitted = True
            return self

        idx_end = np.arange(self._effective_seq_len, total_points)
        if idx_end.size > self.max_windows:
            idx_end = np.linspace(
                self._effective_seq_len,
                total_points - 1,
                num=self.max_windows,
                dtype=int
            )

        Xw = self._build_windows(X_scaled, self._effective_seq_len, idx_end)
        yw = y_scaled[idx_end]  # Use scaled targets

        if len(Xw) == 0:
            self._fallback_value = float(np.mean(y_arr))
            self.model = None
            self._is_fitted = True
            return self

        # Train/val split
        split_idx = int(len(Xw) * 0.9)
        split_idx = min(max(split_idx, 1), len(Xw) - 1) if len(Xw) > 1 else len(Xw)

        Xtr, ytr = Xw[:split_idx], yw[:split_idx]
        Xva, yva = Xw[split_idx:], yw[split_idx:]

        train_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=len(Xtr) > self.batch_size
        )

        val_loader = None
        if len(Xva) > 0:
            val_ds = TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva))
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        # Create model
        input_size = X_arr.shape[1]
        self.model = SimplifiedTFT(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        best_state = None
        best_val = float("inf")
        no_improve = 0

        for epoch in range(self.epochs):
            self.model.train()
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

            if val_loader is None:
                continue

            # Validation
            self.model.eval()
            val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    preds = self.model(xb)
                    val_loss += criterion(preds, yb).item() * len(xb)
                    val_count += len(xb)

            val_loss = val_loss / max(val_count, 1)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self._fallback_value = float(np.mean(y_arr))
        self._is_fitted = True
        return self

    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError("TFTAdapter must be fitted first")

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2:
            raise ValueError("TFTAdapter expects 2D arrays")
        if len(X_arr) == 0:
            return np.zeros(0, dtype=np.float32)

        if self.model is None:
            return np.full(len(X_arr), self._fallback_value or 0.0, dtype=np.float32)

        X_scaled = self.scaler.transform(X_arr)
        seq_len = self._effective_seq_len
        total_pts = X_scaled.shape[0]

        idx_end = np.arange(seq_len, total_pts)
        if idx_end.size == 0:
            return np.full(total_pts, self._fallback_value or 0.0, dtype=np.float32)

        Xw = self._build_windows(X_scaled, seq_len, idx_end)
        pred_ds = TensorDataset(torch.from_numpy(Xw))
        pred_loader = DataLoader(pred_ds, batch_size=self.batch_size, shuffle=False)

        preds = []
        self.model.eval()
        with torch.no_grad():
            for (xb,) in pred_loader:
                xb = xb.to(self.device)
                preds.append(self.model(xb).detach().cpu().numpy())

        preds_arr = np.concatenate(preds, axis=0) if preds else np.zeros(0, dtype=np.float32)

        # NEW: Inverse transform to get predictions in original scale
        preds_arr = self.target_scaler.inverse_transform(preds_arr.reshape(-1, 1)).ravel().astype(np.float32)

        pad_value = preds_arr[0] if preds_arr.size else (self._fallback_value or 0.0)
        padding = np.full(seq_len, pad_value, dtype=np.float32)
        full = np.concatenate([padding, preds_arr], axis=0)
        return full[:len(X_arr)]


class HybridAdapter:
    """Συνδυάζει baseline + residual μοντέλο (stacking δύο adapters)."""

    def __init__(self, base_model, residual_model):
        self.base_model = base_model
        self.residual_model = residual_model
        self._fitted = False

    def fit(self, X, y):
        self.base_model.fit(X, y)
        base_pred = self.base_model.predict(X)
        residuals = y - base_pred
        self.residual_model.fit(X, residuals)
        self._fitted = True
        return self

    def predict(self, X):
        assert self._fitted, "HybridAdapter must be fitted before predict"
        base_pred = self.base_model.predict(X)
        resid_pred = self.residual_model.predict(X)
        return base_pred + resid_pred


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_model(name: str, params: Optional[Dict[str, Any]] = None):
    """Return a model adapter by name."""

    params = params or {}
    key = name.lower()
    if key in {"ridge", "linear_ridge"}:
        return RidgeAdapter(**params)
    if key in {"enet", "elasticnet", "elastic_net"}:
        return ElasticNetAdapter(**params)
    if key in {"extratrees", "extra_trees", "et"}:
        return ExtraTreesAdapter(**params)
    if key in {"lgbm", "lightgbm"}:
        return LGBMAdapter(**params)
    if key in {"xgb", "xgboost"}:
        return XGBoostAdapter(**params)
    if key in {"ngboost", "ngb"}:
        return NGBoostAdapter(**params)
    if key in {"sarimax", "arimax"}:
        return SARIMAXAdapter(**params)
    if key in {"lstm", "lstm_torch"}:
        return LSTMAdapter(**params)
    if key in {"gru", "gru_torch"}:
        return GRUAdapter(**params)
    if key in {"bilstm", "bi_lstm", "bidirectional_lstm"}:
        return BiLSTMAdapter(**params)
    if key in {"nhits", "n_hits"}:
        return NHITSAdapter(**params)
    if key in {"patchtst", "patch_tst"}:
        return PatchTSTAdapter(**params)
    if key in {"tft", "temporal_fusion_transformer"}:
        return TFTAdapter(**params)
    if key in {"hybrid", "hybrid_ridge_lgbm"}:
        base_name = params.get("base_model", "ridge")
        resid_name = params.get("residual_model", "lgbm")
        base_kwargs = dict(params.get("base_params", {}))
        resid_kwargs = dict(params.get("resid_params", {}))
        base_model = get_model(base_name, base_kwargs)
        residual_model = get_model(resid_name, resid_kwargs)
        return HybridAdapter(base_model=base_model, residual_model=residual_model)
    raise ValueError(f"Unknown model name: {name}")
