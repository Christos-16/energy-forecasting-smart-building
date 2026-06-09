#!/usr/bin/env python3
"""
Walk-Forward Backtest + Optional Residual ARIMA Boost - OOP Refactored Version
- Input: cleaned v3 dataset (clean_full.csv) με στήλη ts_utc και targets value_t+H
- Model: Multi-algorithm support with unified interface
- Residual ARIMA: ταιριάζει ARIMA στα residuals της validation και προβλέπει τα residuals στο test,
  προσθέτοντάς τα στις baseline προβλέψεις (Hybrid).
- Outputs: CSV με metrics ανά fold/horizon και averages, json σύνοψη, png plots προαιρετικά.

Usage:
python walkforward_backtest.py \
  --clean-csv results/v3/cleaned_data/clean_full.csv \
  --horizons 1 5 10 \
  --n-folds 3 \
  --train-ratio 0.60 --val-ratio 0.10 --test-ratio 0.30 \
  --output-dir results/v3/backtest \
  --model lgbm \
  --residual-model arima \
  --arima-order 1 0 1
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightgbm as lgb
from lightgbm import LGBMRegressor

try:  # Optional dependency for Prophet-based forecasting
    from prophet import Prophet
except Exception:  # pragma: no cover - handled at runtime with informative errors
    Prophet = None

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA


@dataclass
class BacktestConfig:
    """Configuration container for walkforward backtest parameters."""

    # Data parameters
    clean_csv: Optional[str] = None
    raw_csv: Optional[str] = None
    timestamp_col: Optional[str] = None
    value_col: str = "value"

    # Backtest parameters
    horizons: List[int] = None
    horizons_hours: Optional[List[int]] = None
    horizons_minutes: Optional[List[int]] = None
    n_folds: int = 3
    train_ratio: float = 0.60
    val_ratio: float = 0.10
    test_ratio: float = 0.30

    # Model parameters
    model: str = "lgbm"
    residual_model: str = "none"
    arima_order: Tuple[int, int, int] = (1, 0, 1)
    model_params_override: Optional[Dict[str, Any]] = None

    # Feature engineering (for leak-safe preprocessing)
    leak_safe_preprocessing: bool = False
    leak_clip_quantiles: Optional[Tuple[float, float]] = None
    leak_disable_clip: bool = False
    leak_hampel: bool = True
    leak_hampel_window: int = 25
    leak_hampel_sigma: float = 3.0
    leak_daily_fourier_K: int = 3
    leak_weekly_fourier_K: int = 1
    leak_lags: List[int] = None
    leak_no_temporal: bool = False

    # Output and behavior
    output_dir: str = "results/backtest"
    drift_scan: bool = False
    hybrid_gbm: bool = False

    def __post_init__(self):
        if self.horizons is None:
            self.horizons = [1, 5, 10]
        if self.leak_lags is None:
            self.leak_lags = [1, 5, 10, 30, 60, 120, 720]


class ModelFactory:
    """Factory για δημιουργία ML models με unified interface."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.model_params_override = config.model_params_override or {}

    def create_model(self, model_name: str) -> 'BaseModelAdapter':
        """Create model instance based on name."""
        if model_name == "lgbm":
            return LGBMModelAdapter(self.model_params_override)
        elif model_name == "xgb":
            return XGBModelAdapter(self.model_params_override)
        elif model_name == "cat":
            return CatBoostModelAdapter(self.model_params_override)
        elif model_name == "ngboost":
            return NGBoostModelAdapter(self.model_params_override)
        elif model_name == "rf":
            return RandomForestModelAdapter(self.model_params_override)
        elif model_name == "knn":
            return KNNModelAdapter(self.model_params_override)
        elif model_name == "quantile_lgbm":
            return QuantileLGBMModelAdapter(self.model_params_override)
        elif model_name == "prophet":
            return ProphetModelAdapter(self.model_params_override)
        elif model_name == "naive":
            return NaiveModelAdapter(self.model_params_override)
        elif model_name in ["ridge", "enet", "extratrees", "sarimax", "lstm", "gru", "bilstm", "nhits", "patchtst", "tft", "hybrid"]:
            return ExtraModelAdapter(model_name, self.model_params_override)
        elif model_name == "chronos2":
            return FoundationModelAdapter("Chronos-2", self.model_params_override)
        elif model_name == "timesfm":
            return FoundationModelAdapter("TimesFM", self.model_params_override)
        else:
            raise ValueError(f"Unsupported model: {model_name}")


class BaseModelAdapter:
    """Βασική κλάση για όλους τους ML adapters."""

    def __init__(self, params_override: Dict[str, Any]):
        self.params_override = params_override
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Fit the model on training data."""
        raise NotImplementedError

    def predict(self, X):
        """Make predictions."""
        raise NotImplementedError

    def get_default_params(self):
        """Get default parameters for this model."""
        return {}


class FoundationModelAdapter(BaseModelAdapter):
    """Adapter for foundation models (Chronos-2, TimesFM) in walk-forward framework."""

    def __init__(self, model_name: str, params_override: Dict[str, Any]):
        super().__init__(params_override)
        self.foundation_model_name = model_name
        self._adapter = None

    def _get_adapter(self):
        if self._adapter is not None:
            return self._adapter
        from common.foundation_models import create_foundation_model
        device = self.params_override.get("device", "cpu")
        self._adapter = create_foundation_model(self.foundation_model_name, device=device)
        return self._adapter

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        adapter = self._get_adapter()
        adapter.fit(X_train, y_train)
        self.model = adapter

    def predict(self, X):
        adapter = self._get_adapter()
        return adapter.predict(X)


class LGBMModelAdapter(BaseModelAdapter):
    """LightGBM Tweedie adapter."""

    def get_default_params(self):
        return {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": -1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "tweedie",
            "tweedie_variance_power": 1.2,
            "random_state": 42,
            "n_jobs": -1,
        }

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        params = self.get_default_params()

        # Handle aliases
        if "feature_fraction" in self.params_override:
            self.params_override["colsample_bytree"] = self.params_override["feature_fraction"]
        if "bagging_fraction" in self.params_override:
            self.params_override["subsample"] = self.params_override["bagging_fraction"]
        if "min_data_in_leaf" in self.params_override:
            self.params_override["min_child_samples"] = self.params_override["min_data_in_leaf"]

        params.update(self.params_override)

        self.model = LGBMRegressor(**params)

        if X_val is not None and y_val is not None:
            try:
                self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                             callbacks=[lgb.early_stopping(50)])
            except TypeError:
                self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        else:
            self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X, num_iteration=getattr(self.model, "best_iteration_", None))


class XGBModelAdapter(BaseModelAdapter):
    """XGBoost adapter."""

    def get_default_params(self):
        return {
            "n_estimators": 800,
            "learning_rate": 0.05,
            "max_depth": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": -1,
        }

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        try:
            import xgboost as xgb
        except ImportError as e:
            raise RuntimeError(f"XGBoost not available: {e}")

        params = self.get_default_params()
        params.update(self.params_override)

        self.model = xgb.XGBRegressor(**params)

        if X_val is not None and y_val is not None:
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)


class CatBoostModelAdapter(BaseModelAdapter):
    """CatBoost adapter."""

    def get_default_params(self):
        return {
            "iterations": 1000,
            "learning_rate": 0.1,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "subsample": 0.8,
            "random_seed": 42,
            "loss_function": "RMSE",
            "verbose": False,
            "thread_count": -1,
        }

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        try:
            from catboost import CatBoostRegressor
        except ImportError as e:
            raise RuntimeError(f"CatBoost not available: {e}")

        params = self.get_default_params()
        params.update(self.params_override)

        self.model = CatBoostRegressor(**params)

        if X_val is not None and y_val is not None:
            self.model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        else:
            self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)


class NGBoostModelAdapter(BaseModelAdapter):
    """NGBoost adapter."""

    def get_default_params(self):
        return {
            "n_estimators": 600,
            "learning_rate": 0.03,
            "natural_gradient": False,
            "verbose": False,
            "random_state": 42,
            "minibatch_frac": 0.5,
        }

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        try:
            from ngboost import NGBRegressor
            from ngboost.distns import Normal
            from ngboost.scores import MLE
        except ImportError as e:
            raise RuntimeError(f"NGBoost not available: {e}")

        params = self.get_default_params()
        params.update(self.params_override)

        # Extract NGBoost-specific parameters
        dist = params.pop("Dist", Normal)
        score = params.pop("Score", MLE)
        base = params.pop("Base", None)

        if base is not None:
            self.model = NGBRegressor(Base=base, Dist=dist, Score=score, **params)
        else:
            self.model = NGBRegressor(Dist=dist, Score=score, **params)

        X_train_np = np.asarray(X_train, dtype=np.float32)
        y_train_np = np.asarray(y_train, dtype=np.float64)

        if X_val is not None and y_val is not None:
            X_val_np = np.asarray(X_val, dtype=np.float32)
            y_val_np = np.asarray(y_val, dtype=np.float64)
            try:
                self.model.fit(X_train_np, y_train_np, X_val=X_val_np, Y_val=y_val_np)
            except TypeError:
                self.model.fit(X_train_np, y_train_np)
        else:
            self.model.fit(X_train_np, y_train_np)

    def predict(self, X):
        return self.model.predict(X)


class RandomForestModelAdapter(BaseModelAdapter):
    """Random Forest adapter."""

    def get_default_params(self):
        return {
            "n_estimators": 300,
            "max_depth": 12,
            "max_features": "sqrt",
            "min_samples_leaf": 5,
            "random_state": 42,
            "n_jobs": -1,
        }

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        from sklearn.ensemble import RandomForestRegressor

        params = self.get_default_params()
        params.update(self.params_override)

        self.model = RandomForestRegressor(**params)
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)


class KNNModelAdapter(BaseModelAdapter):
    """K-Nearest Neighbors adapter με preprocessing pipeline."""

    def get_default_params(self):
        return {
            "n_neighbors": 10,
            "weights": "distance",
            "metric": "minkowski",
            "p": 2,
            "algorithm": "auto",
            "n_jobs": -1,
        }

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        params = self.get_default_params()

        # Feature engineering params
        use_pca = self.params_override.pop("use_pca", False)
        pca_components = self.params_override.pop("pca_components", 40)

        params.update(self.params_override)

        # Preprocessing pipeline
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Optional PCA
        if use_pca and X_train_scaled.shape[1] > pca_components:
            self.pca = PCA(n_components=min(pca_components, X_train_scaled.shape[1]))
            X_train_scaled = self.pca.fit_transform(X_train_scaled)
        else:
            self.pca = None

        self.model = KNeighborsRegressor(**params)
        self.model.fit(X_train_scaled, y_train)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        if self.pca is not None:
            X_scaled = self.pca.transform(X_scaled)
        return self.model.predict(X_scaled)


class ProphetModelAdapter(BaseModelAdapter):
    """Adapter για Facebook Prophet σε direct multi-horizon forecasts."""

    def __init__(self, params_override: Dict[str, Any]):
        super().__init__(params_override)
        self.horizon_steps: Optional[int] = None
        self.step_seconds: Optional[int] = None
        self.fallback_value: float = 0.0

    def set_context(self, horizon: int, step_seconds: int) -> None:
        self.horizon_steps = int(horizon)
        self.step_seconds = max(1, int(step_seconds))

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "daily_seasonality": True,
            "weekly_seasonality": True,
            "yearly_seasonality": False,
            "seasonality_mode": "additive",
            "changepoint_prior_scale": 0.1,
        }

    @staticmethod
    def _extract_epoch_seconds(X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if "prophet_time" in X.columns:
                arr = X["prophet_time"].to_numpy(dtype=np.float64)
            else:
                arr = X.to_numpy(dtype=np.float64).ravel()
        elif isinstance(X, pd.Series):
            arr = X.to_numpy(dtype=np.float64)
        else:
            arr = np.asarray(X, dtype=np.float64)
            if arr.ndim > 1:
                arr = arr[:, 0]
        return arr

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if Prophet is None:
            raise ImportError("Prophet is required for prophet model (pip install prophet)")
        if self.horizon_steps is None or self.step_seconds is None:
            raise RuntimeError("ProphetModelAdapter context is not set")

        epochs = self._extract_epoch_seconds(X_train)
        y_arr = np.asarray(y_train, dtype=np.float64)
        mask = np.isfinite(epochs) & np.isfinite(y_arr)
        epochs = epochs[mask]
        y_arr = y_arr[mask]

        if epochs.size == 0:
            raise ValueError("ProphetModelAdapter received empty training data")

        ds = pd.to_datetime(
            epochs + self.horizon_steps * self.step_seconds,
            unit="s",
            utc=True,
        ).tz_convert(None)
        train_df = pd.DataFrame({"ds": ds, "y": y_arr})
        train_df = train_df.dropna().drop_duplicates(subset="ds").sort_values("ds")

        if train_df.empty:
            raise ValueError("ProphetModelAdapter training data empty after cleaning")

        self.fallback_value = float(train_df["y"].mean()) if not train_df["y"].empty else 0.0

        max_history = int(self.params_override.get("max_history", 20000) or 0)
        if max_history > 0 and len(train_df) > max_history:
            train_df = train_df.iloc[-max_history:].copy()

        params = self.get_default_params()
        for key, value in self.params_override.items():
            if key in {"max_history"}:
                continue
            params[key] = value

        self.model = Prophet(**params)
        self.model.fit(train_df)
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("ProphetModelAdapter must be fitted before predict().")
        if self.horizon_steps is None or self.step_seconds is None:
            raise RuntimeError("ProphetModelAdapter context is not set")

        epochs = self._extract_epoch_seconds(X)
        if epochs.size == 0:
            return np.asarray([], dtype=np.float64)

        future = pd.DataFrame(
            {
                "ds": pd.to_datetime(
                    epochs + self.horizon_steps * self.step_seconds,
                    unit="s",
                    utc=True,
                ).tz_convert(None)
            }
        )
        future_unique = future.drop_duplicates().sort_values("ds")
        forecast = self.model.predict(future_unique)
        merged = future.merge(forecast[["ds", "yhat"]], on="ds", how="left")
        preds = merged["yhat"].to_numpy(dtype=np.float64)

        if np.isnan(preds).any():
            fallback = self.fallback_value
            preds = np.nan_to_num(preds, nan=fallback)

        return preds


class NaiveModelAdapter(BaseModelAdapter):
    """Naive persistence baseline: y_pred[t] = y_actual[t-1].

    Uses the 'value_lag1' feature which contains the actual previous timestep value.
    This is the proper persistence baseline for time series forecasting.
    """

    def __init__(self, params_override: Dict[str, Any]):
        super().__init__(params_override)
        self.lag1_col_idx = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Find the value_lag1 column index."""
        # Convert to DataFrame if needed to find column names
        if isinstance(X_train, pd.DataFrame):
            if 'value_lag1' in X_train.columns:
                self.lag1_col_idx = X_train.columns.get_loc('value_lag1')
            else:
                # Fallback: assume first lag feature
                lag_cols = [c for c in X_train.columns if 'lag1' in c.lower()]
                if lag_cols:
                    self.lag1_col_idx = X_train.columns.get_loc(lag_cols[0])
                else:
                    self.lag1_col_idx = 0  # Fallback to first column
        else:
            # If numpy array, assume first column is lag1 (common convention)
            self.lag1_col_idx = 0

        return self

    def predict(self, X):
        """Predict using value_lag1 feature (persistence)."""
        if isinstance(X, pd.DataFrame):
            if 'value_lag1' in X.columns:
                return X['value_lag1'].to_numpy(dtype=np.float64)
            else:
                lag_cols = [c for c in X.columns if 'lag1' in c.lower()]
                if lag_cols:
                    return X[lag_cols[0]].to_numpy(dtype=np.float64)
                # Fallback
                return X.iloc[:, self.lag1_col_idx].to_numpy(dtype=np.float64)
        else:
            # Numpy array
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim == 1:
                return X_arr
            return X_arr[:, self.lag1_col_idx]


class QuantileLGBMModelAdapter(BaseModelAdapter):
    """Quantile LightGBM adapter για uncertainty quantification."""

    def get_default_params(self):
        return {
            "n_estimators": 1000,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
        }

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        quantiles = self.params_override.pop("quantiles", [0.1, 0.5, 0.9])
        base_params = self.get_default_params()
        base_params.update(self.params_override)

        self.models = {}
        for q in quantiles:
            params = base_params.copy()
            params.update({
                "objective": "quantile",
                "alpha": q,
            })

            model = LGBMRegressor(**params)

            if X_val is not None and y_val is not None:
                model.fit(X_train, y_train, eval_set=(X_val, y_val),
                         callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            else:
                model.fit(X_train, y_train)

            self.models[f"q{int(q*100)}"] = model

    def predict(self, X):
        # Return median (q50) as primary prediction
        return self.models["q50"].predict(X, num_iteration=getattr(self.models["q50"], "best_iteration_", None))

    def predict_quantiles(self, X):
        """Return all quantile predictions."""
        results = {}
        for qname, model in self.models.items():
            results[f"pred_{qname}"] = model.predict(X, num_iteration=getattr(model, "best_iteration_", None))
        return results


class ExtraModelAdapter(BaseModelAdapter):
    """Adapter για ridge, elastic net, extra trees, etc. μέσω common.extra_models_and_metrics."""

    def __init__(self, model_name: str, params_override: Dict[str, Any]):
        super().__init__(params_override)
        self.model_name = model_name

        # Import the external model factory
        ANALYSIS_DIR = Path(__file__).resolve().parent
        ROOT_DIR = ANALYSIS_DIR.parent
        if str(ROOT_DIR) not in sys.path:
            sys.path.insert(0, str(ROOT_DIR))

        from common.extra_models_and_metrics import get_model as get_extra_model
        self.get_extra_model = get_extra_model

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        params = dict(self.params_override) if self.params_override else {}

        if self.model_name == "sarimax" and "exog_regex" not in params:
            params["exog_regex"] = r"^(fourier_|hour_|dow_|is_)"

        self.adapter = self.get_extra_model(self.model_name, params)

        print(f"[ModelAdapter] fitting {self.model_name} with params={params}", flush=True)

        if self.model_name == "sarimax":
            self.adapter.fit(X_train, y_train.values)
        else:
            self.adapter.fit(X_train.values, y_train.values)

    def predict(self, X):
        if self.model_name == "sarimax":
            return self.adapter.predict(X)
        else:
            return self.adapter.predict(X.values)


class MetricsCalculator:
    """Υπολογισμός performance metrics με υποστήριξη για extended metrics."""

    @staticmethod
    def rmse(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    @staticmethod
    def mape(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        denom = np.clip(np.abs(y_true), 1e-8, None)
        return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

    @staticmethod
    def evaluate_all(y_true, y_pred, ljung_lags=(10, 20, 40)):
        """Compute all standard metrics plus Ljung-Box test."""
        res = y_true - y_pred
        metrics = {
            "RMSE": MetricsCalculator.rmse(y_true, y_pred),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "MAPE": MetricsCalculator.mape(y_true, y_pred),
            "R2": float(r2_score(y_true, y_pred))
        }
        pvals = acorr_ljungbox(res, lags=list(ljung_lags), return_df=True)["lb_pvalue"].to_dict()
        pvals = {str(k): float(v) for k, v in pvals.items()}
        return metrics, pvals

    @staticmethod
    def evaluate_extended_metrics(y_true, y_pred, train_y=None, seasonality=1):
        """Compute extended metrics μέσω common.extra_models_and_metrics."""
        try:
            ANALYSIS_DIR = Path(__file__).resolve().parent
            ROOT_DIR = ANALYSIS_DIR.parent
            if str(ROOT_DIR) not in sys.path:
                sys.path.insert(0, str(ROOT_DIR))

            from common.extra_models_and_metrics import evaluate_metrics
            return evaluate_metrics(y_true, y_pred, train_y=train_y, seasonality=seasonality)
        except ImportError:
            return {}


class ResidualModeling:
    """Residual modeling με ARIMA ή ETS για hybrid predictions."""

    @staticmethod
    def arima_boost(residuals_val, steps: int, order=(1, 0, 1)):
        """Fit ARIMA on validation residuals and forecast test residuals."""
        res = pd.Series(np.asarray(residuals_val, dtype=np.float64))
        if res.std() < 1e-12:
            return np.zeros(steps, dtype=np.float64)

        try:
            model = ARIMA(res, order=order)
            fitted = model.fit(method_kwargs={"warn_convergence": False})
            fc = fitted.forecast(steps=steps)
            return np.asarray(fc, dtype=np.float64)
        except Exception as e:
            print(f"Warning: ARIMA fitting failed: {e}, returning zeros")
            return np.zeros(steps, dtype=np.float64)

    @staticmethod
    def ets_boost(residuals_val, steps: int):
        """Fit ETS on validation residuals and forecast test residuals."""
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            res = pd.Series(np.asarray(residuals_val, dtype=np.float64))
            if res.std() < 1e-12:
                return np.zeros(steps, dtype=np.float64)
            model = ExponentialSmoothing(res, trend=None, seasonal=None, initialization_method="estimated")
            fit = model.fit(optimized=True)
            fc = fit.forecast(steps)
            return np.asarray(fc, dtype=np.float64)
        except Exception as e:
            print(f"Warning: ETS fitting failed: {e}, returning zeros")
            return np.zeros(steps, dtype=np.float64)


class FoldManager:
    """Δημιουργία και διαχείριση walk-forward folds."""

    @staticmethod
    def make_folds_idx(n: int, train_ratio: float, val_ratio: float, test_ratio: float, n_folds: int):
        """Generate expanding window fold indices."""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        test_len_total = int(round(n * test_ratio))
        fold_len = max(1, test_len_total // n_folds)
        val_len = int(round(n * val_ratio))
        base_train_end = int(round(n * train_ratio))

        folds = []
        for i in range(n_folds):
            train_end = base_train_end + i * fold_len
            val_end = train_end + val_len
            test_end = val_end + fold_len
            if test_end > n:
                break
            folds.append((0, train_end, val_end, test_end))  # expanding: train starts at 0

        return folds


class TimeSeriesUtils:
    """Utilities για time series preprocessing και inference."""

    @staticmethod
    def infer_step_seconds(dt_index: pd.DatetimeIndex) -> int:
        """Estimate base sampling step in seconds from a DatetimeIndex."""
        if not isinstance(dt_index, pd.DatetimeIndex) or len(dt_index) < 3:
            return 300  # default to 5 minutes if uncertain

        diffs = dt_index.view("int64")[1:] - dt_index.view("int64")[:-1]
        step_ns = np.median(diffs)
        step_s = int(round(step_ns / 1e9))
        return max(step_s, 1)

    @staticmethod
    def to_steps_from_hours(hours_list: List[int], step_seconds: int) -> List[int]:
        return [max(1, int(round((h * 3600) / step_seconds))) for h in hours_list]

    @staticmethod
    def to_steps_from_minutes(mins_list: List[int], step_seconds: int) -> List[int]:
        return [max(1, int(round((m * 60) / step_seconds))) for m in mins_list]

    @staticmethod
    def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
        return df.apply(pd.to_numeric, errors="coerce")


class LeakSafeFoldBuilder:
    """Παράγει train/val/test frames με fit-on-train preprocessing ανά fold."""

    def __init__(
        self,
        raw_df: pd.DataFrame,
        prototype: 'LeakSafePreprocessor',
        horizons: Iterable[int],
    ):
        self.raw_df = raw_df.reset_index(drop=True).copy()
        self.prototype = prototype
        self.horizons = tuple(sorted({int(h) for h in horizons}))
        self.max_horizon = max(self.horizons, default=0)
        self.timestamp_col = prototype.timestamp_col
        self.value_col = prototype.value_col

    def __call__(self, tr0: int, tr1: int, v1: int, te1: int):
        pre = self.prototype.clone()
        pre.fit(self.raw_df.iloc[tr0:tr1].copy())

        train_proc = self._build_split(pre, tr0, tr1, include_history=False)
        val_proc = self._build_split(pre, tr1, v1, include_history=True)
        test_proc = self._build_split(pre, v1, te1, include_history=True)

        return train_proc, val_proc, test_proc

    def _build_split(
        self,
        pre: 'LeakSafePreprocessor',
        start: int,
        end: int,
        include_history: bool,
    ) -> pd.DataFrame:
        if start >= end or start >= len(self.raw_df):
            return pd.DataFrame(columns=self.raw_df.columns)

        hist = max(pre.history_lookback, 0) if include_history else 0
        hist_start = max(0, start - hist)
        target_end = min(len(self.raw_df), end + self.max_horizon)

        subset = self.raw_df.iloc[hist_start:target_end].copy()
        if subset.empty:
            return subset

        transformed = pre.transform(subset)
        if transformed.empty:
            return transformed

        # Import here to avoid circular dependency
        try:
            from analysis.leak_safe_preprocess import add_targets as leak_add_targets
        except ModuleNotFoundError:
            from leak_safe_preprocess import add_targets as leak_add_targets

        transformed = leak_add_targets(transformed, self.value_col, self.horizons)
        transformed = transformed.dropna().reset_index(drop=True)

        start_ts = self.raw_df.iloc[start][self.timestamp_col]
        end_ts = self.raw_df.iloc[end - 1][self.timestamp_col]
        mask = (transformed[self.timestamp_col] >= start_ts) & (
            transformed[self.timestamp_col] <= end_ts
        )
        transformed = transformed.loc[mask].reset_index(drop=True)

        numeric_cols = transformed.select_dtypes(include=[np.number]).columns.tolist()
        keep_cols = [c for c in transformed.columns if c in numeric_cols or c == self.timestamp_col]
        transformed = transformed.loc[:, keep_cols]

        return transformed


class WalkForwardBacktester:
    """Main class για walk-forward backtesting με OOP architecture."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.model_factory = ModelFactory(config)
        self.fold_builder = None

    def run_backtest(self) -> Dict[str, Any]:
        """Run complete backtest pipeline and return results."""

        # Setup data and preprocessing
        df, step_seconds = self._load_and_prepare_data()

        # Compute effective horizons
        horizons_steps = self._compute_horizons(step_seconds)

        # Setup folds
        folds = self._setup_folds(len(df))

        print(f"=== WALK-FORWARD BACKTEST ===")
        print(f"Dataset: {len(df)} rows")
        print(f"Model: {self.config.model}")
        print(f"Folds: {len(folds)}")
        print(f"Horizons: {horizons_steps}")
        print()

        # Run backtest for each horizon
        all_results = []
        summaries = {}

        for horizon in horizons_steps:
            print(f"=== HORIZON +{horizon} ===")
            fold_results, avg_results = self._backtest_one_horizon(
                df, horizon, folds, step_seconds
            )
            all_results.append(fold_results)
            summaries[str(horizon)] = {
                "avg": avg_results.to_dict(orient="records"),
                "folds": fold_results.to_dict(orient="records")
            }
            print()

        # Aggregate results
        all_results_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
        overall_avg = self._compute_overall_averages(all_results_df)

        # Save results
        self._save_results(all_results_df, overall_avg, summaries, horizons_steps, step_seconds)

        return {
            "fold_results": all_results_df,
            "overall_averages": overall_avg,
            "per_horizon_summaries": summaries,
            "config": self.config
        }

    def _load_and_prepare_data(self) -> Tuple[pd.DataFrame, int]:
        """Load and prepare input data."""
        use_leak_safe = self.config.leak_safe_preprocessing

        if use_leak_safe:
            if not self.config.raw_csv:
                raise ValueError("Leak-safe preprocessing requires --raw-csv")

            df = pd.read_csv(self.config.raw_csv)
            timestamp_col = self.config.timestamp_col or ("ts_utc" if "ts_utc" in df.columns else "ts")

            # Setup leak-safe preprocessing
            self._setup_leak_safe_preprocessing(df, timestamp_col)

        else:
            if not self.config.clean_csv:
                raise ValueError("Standard processing requires --clean-csv")

            df = pd.read_csv(self.config.clean_csv)
            timestamp_col = self.config.timestamp_col or ("ts_utc" if "ts_utc" in df.columns else "ts")

        # Ensure timestamp column exists and is datetime
        if timestamp_col not in df.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found")

        df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
        df = df.dropna(subset=[timestamp_col]).sort_values(timestamp_col).reset_index(drop=True)

        # Update config with resolved timestamp column
        self.config.timestamp_col = timestamp_col

        # Infer step size
        dt_index = pd.DatetimeIndex(df[timestamp_col])
        step_seconds = TimeSeriesUtils.infer_step_seconds(dt_index)

        return df, step_seconds

    def _setup_leak_safe_preprocessing(self, df: pd.DataFrame, timestamp_col: str):
        """Setup leak-safe preprocessing pipeline."""
        try:
            from analysis.leak_safe_preprocess import LeakSafePreprocessor
        except ModuleNotFoundError:
            from leak_safe_preprocess import LeakSafePreprocessor

        # Determine clipping quantiles
        clip = None
        if not self.config.leak_disable_clip:
            if self.config.leak_clip_quantiles:
                clip = self.config.leak_clip_quantiles
            else:
                clip = (0.01, 0.99)

        # Create preprocessor template
        preprocessor = LeakSafePreprocessor(
            value_col=self.config.value_col,
            timestamp_col=timestamp_col,
            clip_quantiles=clip,
            use_hampel=self.config.leak_hampel,
            hampel_window=self.config.leak_hampel_window,
            hampel_sigma=self.config.leak_hampel_sigma,
            daily_fourier_K=self.config.leak_daily_fourier_K,
            weekly_fourier_K=self.config.leak_weekly_fourier_K,
            lags=self.config.leak_lags,
            add_temporal=not self.config.leak_no_temporal,
        )

        # This will be used later during fold creation
        self.leak_safe_preprocessor = preprocessor

    def _compute_horizons(self, step_seconds: int) -> List[int]:
        """Compute effective horizon steps from configuration."""
        horizons_steps = []

        # Add hour-based horizons
        if self.config.horizons_hours:
            horizons_steps.extend(
                TimeSeriesUtils.to_steps_from_hours(self.config.horizons_hours, step_seconds)
            )

        # Add minute-based horizons
        if self.config.horizons_minutes:
            horizons_steps.extend(
                TimeSeriesUtils.to_steps_from_minutes(self.config.horizons_minutes, step_seconds)
            )

        # Add direct step horizons
        horizons_steps.extend(self.config.horizons)

        # Remove duplicates and sort
        horizons_steps = sorted(set(int(max(1, h)) for h in horizons_steps))

        if not horizons_steps:
            raise ValueError("No valid horizons provided")

        return horizons_steps

    def _setup_folds(self, n_samples: int) -> List[Tuple[int, int, int, int]]:
        """Setup walk-forward folds."""
        folds = FoldManager.make_folds_idx(
            n_samples,
            self.config.train_ratio,
            self.config.val_ratio,
            self.config.test_ratio,
            self.config.n_folds
        )

        if not folds:
            raise ValueError("No valid folds generated. Adjust ratios or n_folds.")

        return folds

    def _backtest_one_horizon(
        self,
        df: pd.DataFrame,
        horizon: int,
        folds: List[Tuple[int, int, int, int]],
        step_seconds: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Backtest single horizon across all folds."""

        target = f"{self.config.value_col}_t+{horizon}"
        outdir = Path(self.config.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        # Setup fold builder if using leak-safe preprocessing
        if self.config.leak_safe_preprocessing and hasattr(self, 'leak_safe_preprocessor'):
            fold_builder = LeakSafeFoldBuilder(df, self.leak_safe_preprocessor, [horizon])
        else:
            fold_builder = None
            # Verify target column exists
            if target not in df.columns:
                raise ValueError(f"Target column '{target}' not found in dataset")

        # Calculate seasonality for metrics
        freq_seconds = df[self.config.timestamp_col].diff().dt.total_seconds().dropna().median()
        seasonality = 1
        if pd.notna(freq_seconds) and freq_seconds > 0:
            try:
                seasonality = max(1, int(round(86400.0 / float(freq_seconds))))
            except (ValueError, ZeroDivisionError):
                seasonality = 1

        fold_results = []

        for fold_idx, (tr0, tr1, v1, te1) in enumerate(folds, 1):
            print(f"  Fold {fold_idx}: train[{tr0}:{tr1}], val[{tr1}:{v1}], test[{v1}:{te1}]")

            # Prepare fold data
            if fold_builder:
                df_train, df_val, df_test = fold_builder(tr0, tr1, v1, te1)
            else:
                df_train = df.iloc[tr0:tr1].copy()
                df_val = df.iloc[tr1:v1].copy()
                df_test = df.iloc[v1:te1].copy()

            # Process fold
            fold_result = self._process_single_fold(
                fold_idx,
                df_train,
                df_val,
                df_test,
                horizon,
                target,
                step_seconds,
                seasonality,
                outdir,
                fold_builder is not None,
            )

            if fold_result:
                fold_results.extend(fold_result)

        # Aggregate fold results
        if not fold_results:
            print(f"    No valid folds for horizon +{horizon}")
            empty_df = pd.DataFrame(columns=["fold", "horizon", "kind", "RMSE", "MAE", "MAPE", "R2"])
            return empty_df, empty_df

        fold_results_df = pd.DataFrame(fold_results)
        avg_results = fold_results_df.groupby(["horizon", "kind"], as_index=False)[
            ["RMSE", "MAE", "MAPE", "R2", "wMAPE", "SMAPE", "MASE"]
        ].mean()

        # Save fold-specific results
        fold_results_df.to_csv(outdir / f"folds_h{horizon}.csv", index=False)
        avg_results.to_csv(outdir / f"summary_h{horizon}.csv", index=False)

        return fold_results_df, avg_results

    def _process_single_fold(
        self,
        fold_idx: int,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
        horizon: int,
        target: str,
        step_seconds: int,
        seasonality: int,
        outdir: Path,
        is_leak_safe: bool
    ) -> Optional[List[Dict[str, Any]]]:
        """Process a single fold."""

        if df_train.empty:
            print(f"    [skip] empty train split")
            return None

        if self.config.model == "prophet":
            for frame in (df_train, df_val, df_test):
                if frame.empty:
                    continue
                ts = frame[self.config.timestamp_col].astype("int64", copy=False) / 1e9
                frame.loc[:, "prophet_time"] = ts.astype(np.float64)

        # Select features
        features = self._select_features(df_train, target)
        if not features:
            print(f"    [skip] no usable features")
            return None

        # Prepare datasets
        try:
            X_train, y_train = self._prepare_dataset(df_train, features, target)
            X_val, y_val = self._prepare_dataset(df_val, features, target)
            X_test, y_test = self._prepare_dataset(df_test, features, target)
        except Exception as e:
            print(f"    [skip] dataset preparation failed: {e}")
            return None

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"    [skip] insufficient data after preparation")
            return None

        # Check target variance
        target_std = float(np.std(y_train))
        if target_std < 1e-9:
            print(f"    [skip] target nearly constant (std={target_std:.2e})")
            return None

        print(f"    Data: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

        # Train model and make predictions
        try:
            model_adapter = self.model_factory.create_model(self.config.model)
            if hasattr(model_adapter, "set_context"):
                model_adapter.set_context(horizon=horizon, step_seconds=step_seconds)
            model_adapter.fit(X_train, y_train, X_val if len(X_val) > 0 else None, y_val if len(X_val) > 0 else None)

            y_pred_val = model_adapter.predict(X_val) if len(X_val) > 0 else np.array([])
            y_pred_test = model_adapter.predict(X_test)

        except Exception as e:
            print(f"    [skip] model training failed: {e}")
            return None

        # Evaluate baseline model
        base_metrics, base_ljung = MetricsCalculator.evaluate_all(y_test.values, y_pred_test)
        extended_metrics = MetricsCalculator.evaluate_extended_metrics(
            y_test.values, y_pred_test, train_y=y_train.values, seasonality=seasonality
        )

        baseline_result = {
            "fold": fold_idx,
            "horizon": horizon,
            "kind": "baseline",
            **base_metrics,
            **{f"LB_p@{k}": v for k, v in base_ljung.items()},
            "wMAPE": extended_metrics.get("wMAPE"),
            "SMAPE": extended_metrics.get("SMAPE"),
            "MASE": extended_metrics.get("MASE"),
        }

        results = [baseline_result]

        # Save predictions
        self._save_fold_predictions(
            outdir, horizon, fold_idx, df_test, y_test, y_pred_test,
            model_adapter if self.config.model == "quantile_lgbm" else None
        )

        print(f"    Baseline: RMSE={base_metrics['RMSE']:.3f}, MAPE={base_metrics['MAPE']:.2f}%")

        # Residual modeling (hybrid approach)
        if self.config.residual_model != "none" and len(X_val) > 0:
            hybrid_result = self._apply_residual_modeling(
                y_val, y_pred_val, y_test, y_pred_test, y_train,
                fold_idx, horizon, seasonality, outdir
            )
            if hybrid_result:
                results.append(hybrid_result)

        return results

    def _select_features(self, df: pd.DataFrame, target: str) -> List[str]:
        """Select feature columns."""
        timestamp_col = self.config.timestamp_col
        value_col = self.config.value_col

        if self.config.model == "prophet":
            return ["prophet_time"] if "prophet_time" in df.columns else []

        features = [
            c for c in df.columns
            if c != timestamp_col
            and not c.startswith("ts")
            and not c.startswith(f"{value_col}_t+")
            and c != target
        ]

        # Filter out non-numeric or problematic columns
        if features:
            X_sample = TimeSeriesUtils.ensure_numeric(df[features])
            features = [c for c in X_sample.columns if np.isfinite(X_sample[c].values).any()]

        return features

    def _prepare_dataset(self, df: pd.DataFrame, features: List[str], target: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare X, y from dataframe."""
        if df.empty:
            return pd.DataFrame(columns=features), pd.Series(dtype=np.float32)

        X = TimeSeriesUtils.ensure_numeric(df.reindex(columns=features)).astype(np.float32)
        y = pd.to_numeric(df[target], errors="coerce").astype(np.float32)

        # Remove rows with invalid data
        valid_mask = np.isfinite(X.values).all(axis=1) & np.isfinite(y.values)
        X = X.loc[valid_mask]
        y = y.loc[valid_mask]

        return X, y

    def _apply_residual_modeling(
        self,
        y_val: pd.Series,
        y_pred_val: np.ndarray,
        y_test: pd.Series,
        y_pred_test: np.ndarray,
        y_train: pd.Series,
        fold_idx: int,
        horizon: int,
        seasonality: int,
        outdir: Path
    ) -> Optional[Dict[str, Any]]:
        """Apply residual modeling for hybrid predictions."""

        residuals_val = y_val.values - y_pred_val

        if self.config.residual_model == "arima":
            print(f"    Fitting ARIMA{self.config.arima_order} on validation residuals...")
            residual_forecast = ResidualModeling.arima_boost(
                residuals_val, len(y_test), self.config.arima_order
            )
        elif self.config.residual_model == "ets":
            print(f"    Fitting ETS on validation residuals...")
            residual_forecast = ResidualModeling.ets_boost(residuals_val, len(y_test))
        else:
            return None

        # Hybrid predictions
        y_pred_hybrid = y_pred_test + residual_forecast

        # Evaluate hybrid model
        hybrid_metrics, hybrid_ljung = MetricsCalculator.evaluate_all(y_test.values, y_pred_hybrid)
        extended_hybrid = MetricsCalculator.evaluate_extended_metrics(
            y_test.values, y_pred_hybrid, train_y=y_train.values, seasonality=seasonality
        )

        hybrid_result = {
            "fold": fold_idx,
            "horizon": horizon,
            "kind": "hybrid",
            **hybrid_metrics,
            **{f"LB_p@{k}": v for k, v in hybrid_ljung.items()},
            "wMAPE": extended_hybrid.get("wMAPE"),
            "SMAPE": extended_hybrid.get("SMAPE"),
            "MASE": extended_hybrid.get("MASE"),
        }

        # Save hybrid predictions
        fold_dir = outdir / f"h{horizon}" / f"fold{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        hybrid_df = pd.DataFrame({
            "y_true": y_test.values,
            "y_pred_baseline": y_pred_test,
            "resid_fc": residual_forecast,
            "y_pred_hybrid": y_pred_hybrid,
            "residual_hybrid": y_test.values - y_pred_hybrid,
        })

        hybrid_df.to_csv(fold_dir / "predictions_hybrid.csv", index=False)

        # Calculate improvement
        baseline_rmse = np.sqrt(mean_squared_error(y_test.values, y_pred_test))
        hybrid_rmse = hybrid_metrics["RMSE"]
        improvement = (baseline_rmse - hybrid_rmse) / baseline_rmse * 100

        print(f"    Hybrid: RMSE={hybrid_rmse:.3f} ({improvement:+.1f}% vs baseline)")

        return hybrid_result

    def _save_fold_predictions(
        self,
        outdir: Path,
        horizon: int,
        fold_idx: int,
        df_test: pd.DataFrame,
        y_test: pd.Series,
        y_pred_test: np.ndarray,
        quantile_model: Optional[QuantileLGBMModelAdapter] = None
    ):
        """Save fold predictions to CSV."""
        fold_dir = outdir / f"h{horizon}" / f"fold{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        pred_dict = {
            "y_true": y_test.values,
            "y_pred": y_pred_test,
            "residual": y_test.values - y_pred_test,
        }

        # Add timestamp if available
        if self.config.timestamp_col in df_test.columns:
            pred_dict["ts_utc"] = df_test[self.config.timestamp_col].values

        # Add quantile predictions if available
        if quantile_model and hasattr(quantile_model, 'predict_quantiles'):
            X_test_features = self._select_features(df_test, f"{self.config.value_col}_t+{horizon}")
            if X_test_features:
                X_test = self._prepare_dataset(df_test, X_test_features, f"{self.config.value_col}_t+{horizon}")[0]
                quantile_preds = quantile_model.predict_quantiles(X_test)
                pred_dict.update(quantile_preds)

        pd.DataFrame(pred_dict).to_csv(fold_dir / "predictions_baseline.csv", index=False)

    def _compute_overall_averages(self, all_results_df: pd.DataFrame) -> pd.DataFrame:
        """Compute overall averages across all horizons and folds."""
        if all_results_df.empty:
            return pd.DataFrame()

        metrics_cols = ["RMSE", "MAE", "MAPE", "R2"]
        # Only include columns that exist
        available_cols = [c for c in metrics_cols if c in all_results_df.columns]
        if "wMAPE" in all_results_df.columns:
            available_cols.append("wMAPE")
        if "SMAPE" in all_results_df.columns:
            available_cols.append("SMAPE")
        if "MASE" in all_results_df.columns:
            available_cols.append("MASE")

        overall_avg = all_results_df.groupby(["horizon", "kind"], as_index=False)[available_cols].mean()
        return overall_avg

    def _save_results(
        self,
        all_results_df: pd.DataFrame,
        overall_avg: pd.DataFrame,
        summaries: Dict[str, Any],
        horizons_steps: List[int],
        step_seconds: int
    ):
        """Save all results to files."""
        outdir = Path(self.config.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        if not all_results_df.empty:
            all_results_df.to_csv(outdir / "folds_all.csv", index=False)

        # Save overall averages
        if not overall_avg.empty:
            overall_avg.to_csv(outdir / "summary_overall.csv", index=False)

            # Print summary
            print("📊 OVERALL AVERAGES:")
            print("=" * 50)
            for _, row in overall_avg.iterrows():
                print(f"H+{row['horizon']:2d} {row['kind']:8s}: RMSE={row['RMSE']:6.3f}, MAPE={row['MAPE']:5.2f}%, R²={row['R2']:6.3f}")

        # Save complete summary as JSON
        summary_data = {
            "params": {
                "model": self.config.model,
                "residual_model": self.config.residual_model,
                "arima_order": self.config.arima_order,
                "n_folds": self.config.n_folds,
                "ratios": {
                    "train": self.config.train_ratio,
                    "val": self.config.val_ratio,
                    "test": self.config.test_ratio
                },
                "step_seconds": step_seconds,
                "horizons_steps": horizons_steps,
                "horizons_hours": self.config.horizons_hours,
                "horizons_minutes": self.config.horizons_minutes,
            },
            "per_horizon": summaries,
            "overall_avg": overall_avg.to_dict(orient="records") if not overall_avg.empty else []
        }

        with open(outdir / "summary.json", "w") as f:
            json.dump(summary_data, f, indent=2)

        print(f"\n✅ BACKTEST COMPLETED!")
        print(f"📂 Results saved in: {outdir}")


def main():
    """Main entry point with enhanced argument parsing."""
    ap = argparse.ArgumentParser(description="Walk-Forward Backtest με OOP Architecture")

    # Data arguments
    ap.add_argument("--clean-csv", help="Καθαρισμένο dataset (clean_full.csv)")
    ap.add_argument("--raw-csv", help="Αρχικό CSV για leak-safe preprocessing")
    ap.add_argument("--timestamp-col", help="Όνομα timestamp στήλης (auto-detect αν δεν δοθεί)")
    ap.add_argument("--value-col", default="value", help="Όνομα στήλης με την ισχύ")

    # Horizon arguments
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 10])
    ap.add_argument("--horizons-hours", nargs="+", type=int, help="Ορίζοντες σε ώρες")
    ap.add_argument("--horizons-minutes", nargs="+", type=int, help="Ορίζοντες σε λεπτά")

    # Backtest parameters
    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--train-ratio", type=float, default=0.60)
    ap.add_argument("--val-ratio", type=float, default=0.10)
    ap.add_argument("--test-ratio", type=float, default=0.30)

    # Model arguments
    ap.add_argument(
        "--model",
        choices=[
            "lgbm",
            "xgb",
            "ngboost",
            "cat",
            "rf",
            "quantile_lgbm",
            "knn",
            "ridge",
            "enet",
            "extratrees",
            "sarimax",
            "lstm",
            "gru",
            "bilstm",
            "nhits",
            "patchtst",
            "tft",
            "prophet",
            "naive",
            "hybrid",
            "chronos2",
            "timesfm",
        ],
        default="lgbm"
    )
    ap.add_argument("--residual-model", choices=["none", "arima", "ets"], default="none")
    ap.add_argument("--arima-order", nargs=3, type=int, default=[1, 0, 1], help="p d q")
    ap.add_argument("--model-params-json", help="JSON dict to override model params")

    # Leak-safe preprocessing arguments
    ap.add_argument("--leak-safe-preprocessing", action="store_true")
    ap.add_argument("--leak-clip-quantiles", nargs=2, type=float, metavar=("QLOW", "QHIGH"))
    ap.add_argument("--leak-disable-clip", action="store_true")
    ap.add_argument("--leak-hampel", dest="leak_hampel", action="store_true", default=True)
    ap.add_argument("--leak-disable-hampel", dest="leak_hampel", action="store_false")
    ap.add_argument("--leak-hampel-window", type=int, default=25)
    ap.add_argument("--leak-hampel-sigma", type=float, default=3.0)
    ap.add_argument("--leak-daily-fourier-K", type=int, default=3)
    ap.add_argument("--leak-weekly-fourier-K", type=int, default=1)
    ap.add_argument("--leak-lags", nargs="*", type=int, default=[1, 5, 10, 30, 60, 120, 720])
    ap.add_argument("--leak-no-temporal", action="store_true")

    # Output and behavior
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--drift-scan", action="store_true")
    ap.add_argument("--hybrid-gbm", action="store_true")

    # Legacy support
    ap.add_argument("--with-arima", action="store_true", help="Deprecated: use --residual-model arima")
    ap.add_argument("--lgbm-params-json", help="Deprecated: use --model-params-json")

    args = ap.parse_args()

    # Handle legacy arguments
    if args.with_arima and args.residual_model == "none":
        args.residual_model = "arima"

    # Parse model parameters JSON
    model_params_override = None
    params_json = args.model_params_json or args.lgbm_params_json or os.getenv("MODEL_PARAMS_JSON")
    if params_json:
        try:
            model_params_override = json.loads(params_json)
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse model params JSON: {e}")

    # Create configuration
    config = BacktestConfig(
        clean_csv=args.clean_csv,
        raw_csv=args.raw_csv,
        timestamp_col=args.timestamp_col,
        value_col=args.value_col,
        horizons=args.horizons,
        horizons_hours=args.horizons_hours,
        horizons_minutes=args.horizons_minutes,
        n_folds=args.n_folds,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        model=args.model.lower(),
        residual_model=args.residual_model,
        arima_order=tuple(args.arima_order),
        model_params_override=model_params_override,
        leak_safe_preprocessing=args.leak_safe_preprocessing,
        leak_clip_quantiles=tuple(args.leak_clip_quantiles) if args.leak_clip_quantiles else None,
        leak_disable_clip=args.leak_disable_clip,
        leak_hampel=args.leak_hampel,
        leak_hampel_window=args.leak_hampel_window,
        leak_hampel_sigma=args.leak_hampel_sigma,
        leak_daily_fourier_K=args.leak_daily_fourier_K,
        leak_weekly_fourier_K=args.leak_weekly_fourier_K,
        leak_lags=args.leak_lags,
        leak_no_temporal=args.leak_no_temporal,
        output_dir=args.output_dir,
        drift_scan=args.drift_scan,
        hybrid_gbm=args.hybrid_gbm,
    )

    # Validation
    if not config.clean_csv and not config.raw_csv:
        ap.error("Provide either --clean-csv or --raw-csv")
    if config.leak_safe_preprocessing and not config.raw_csv:
        ap.error("--leak-safe-preprocessing requires --raw-csv")
    if not config.leak_safe_preprocessing and not config.clean_csv:
        ap.error("Standard processing requires --clean-csv")

    # Run backtest
    backtester = WalkForwardBacktester(config)
    results = backtester.run_backtest()

    return results


if __name__ == "__main__":
    main()
