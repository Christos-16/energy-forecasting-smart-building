"""
Foundation Model Adapters for IoT Energy Forecasting
=====================================================
Version: 1.0.0 (2026-02-24)

Provides adapter classes for:
- Chronos-2 (amazon/chronos-2): Encoder-only, ~120M params, native covariate support
- TimesFM 2.0 (google/timesfm-2.0-500m-pytorch): Decoder-only, ~500M params (HF transformers)
- TimesFM 2.5 (google/timesfm-2.5-200m-pytorch): Decoder-only, ~200M params (native timesfm)

Both adapters follow the fit/predict interface used throughout the codebase.
Foundation models are zero-shot (no weight updates during fit); fit() stores
the training context and predict() performs inference.

Memory management: models are loaded lazily on first fit() call and can be
explicitly released via release_model().
"""

from __future__ import annotations

import gc
import time
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Chronos-2 Adapter
# =============================================================================


class Chronos2Adapter:
    """Chronos-2 foundation model with native covariate support.

    Uses BaseChronosPipeline.from_pretrained("amazon/chronos-2") with
    predict_df() for long-format covariate-aware forecasting.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier.
    max_context : int
        Maximum context length (number of past observations).
    device : str
        Device for inference ('cpu', 'cuda', 'mps').
    covariate_cols : list or None
        Feature column names to pass as covariates. If None, uses all features.
    """

    PARAM_COUNT = 120_000_000  # Published parameter count

    def __init__(
        self,
        model_id: str = "amazon/chronos-2",
        max_context: int = 512,
        device: str = "cpu",
        covariate_cols: Optional[List[str]] = None,
    ):
        self.model_id = model_id
        self.max_context = max_context
        self.device = device
        self.covariate_cols = covariate_cols

        self._pipeline = None
        self._context_y: Optional[np.ndarray] = None
        self._context_X: Optional[np.ndarray] = None
        self._feature_names: Optional[List[str]] = None
        self._fit_time: float = 0.0

    def _load_pipeline(self):
        """Lazy-load the Chronos-2 pipeline.

        Uses Chronos2Pipeline (not BaseChronosPipeline) to get predict_df()
        with native covariate support.
        """
        if self._pipeline is not None:
            return

        import torch
        from chronos import Chronos2Pipeline

        print(f"  [Chronos-2] Loading {self.model_id} on {self.device}...")
        self._pipeline = Chronos2Pipeline.from_pretrained(
            self.model_id,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
        )
        print(f"  [Chronos-2] Model loaded.")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            feature_names: Optional[List[str]] = None) -> "Chronos2Adapter":
        """Store training context. No weight updates (zero-shot model).

        Parameters
        ----------
        X_train : np.ndarray
            Training features (n_samples, n_features). Used as covariates.
        y_train : np.ndarray
            Training target values (the time series to forecast).
        feature_names : list of str, optional
            Names for each feature column (for covariate DataFrame construction).
        """
        t0 = time.time()
        self._load_pipeline()

        # Store the tail of training context (up to max_context)
        ctx_len = min(len(y_train), self.max_context)
        self._context_y = y_train[-ctx_len:].copy()
        self._context_X = X_train[-ctx_len:].copy()

        if feature_names is not None:
            self._feature_names = list(feature_names)
        elif self._feature_names is None:
            self._feature_names = [f"feat_{i}" for i in range(X_train.shape[1])]

        self._fit_time = time.time() - t0
        return self

    def predict(self, X_test: np.ndarray, horizon: int = 1) -> np.ndarray:
        """Forecast using Chronos-2 with covariates via predict_df().

        Uses expanding context: after each batch, predictions are appended to
        the context for the next batch (autoregressive evaluation).

        Parameters
        ----------
        X_test : np.ndarray
            Test features (n_test, n_features). Used as future covariates.
        horizon : int
            Forecast horizon (number of steps ahead).

        Returns
        -------
        np.ndarray
            Point predictions for each test sample.
        """
        if self._pipeline is None:
            raise RuntimeError("Must call fit() before predict()")

        n_test = len(X_test)

        # Try covariate-aware prediction first, fall back to univariate
        try:
            predictions = self._predict_with_covariates(X_test, horizon)
        except Exception as e:
            warnings.warn(f"Chronos-2 predict_df failed ({e}), falling back to univariate")
            predictions = self._predict_univariate(X_test, n_test, horizon)

        return predictions

    def _predict_with_covariates(
        self, X_test: np.ndarray, horizon: int
    ) -> np.ndarray:
        """Covariate-aware prediction using Chronos2Pipeline.predict_df().

        Key fixes over v1:
        - Uses BATCH_SIZE (128) instead of horizon for chunking
        - Expanding context: context grows with each batch (timestamp continuity)
        - Ensures future_df always has >= 2 rows for frequency inference
        - Progress logging every batch
        """
        import torch

        n_test = len(X_test)
        predictions = np.full(n_test, np.nan)
        BATCH_SIZE = 128

        # Expanding context arrays (grow after each batch)
        ctx_y = self._context_y.copy()
        ctx_X = self._context_X.copy()

        total_batches = (n_test + BATCH_SIZE - 1) // BATCH_SIZE
        for batch_idx, start in enumerate(range(0, n_test, BATCH_SIZE)):
            end = min(start + BATCH_SIZE, n_test)
            pred_len = end - start

            # Trim context to max_context (keep most recent)
            ctx_len = min(len(ctx_y), self.max_context)
            cur_ctx_y = ctx_y[-ctx_len:]
            cur_ctx_X = ctx_X[-ctx_len:]

            # Build continuous timestamps: context + future (no gaps)
            request_len = max(pred_len, 2)  # >= 2 rows for freq inference
            total_ts = ctx_len + request_len
            timestamps = pd.date_range("2024-01-01", periods=total_ts, freq="5min")

            # Context DataFrame
            ctx_data = {
                "item_id": "sensor",
                "timestamp": timestamps[:ctx_len],
                "target": cur_ctx_y,
            }
            for j, col_name in enumerate(self._feature_names):
                if j < cur_ctx_X.shape[1]:
                    ctx_data[col_name] = cur_ctx_X[:, j]
            ctx_df = pd.DataFrame(ctx_data)

            # Future DataFrame (continuous from context, >= 2 rows)
            future_X = X_test[start:end]
            if request_len > pred_len:
                # Pad with last row repeated to ensure >= 2 rows
                pad_rows = request_len - pred_len
                future_X = np.vstack([future_X, np.tile(future_X[-1:], (pad_rows, 1))])

            future_data = {
                "item_id": "sensor",
                "timestamp": timestamps[ctx_len: ctx_len + request_len],
            }
            for j, col_name in enumerate(self._feature_names):
                if j < future_X.shape[1]:
                    future_data[col_name] = future_X[:, j]
            future_df = pd.DataFrame(future_data)

            try:
                pred_df = self._pipeline.predict_df(
                    df=ctx_df,
                    future_df=future_df,
                    prediction_length=request_len,
                    quantile_levels=[0.5],
                    id_column="item_id",
                    timestamp_column="timestamp",
                    target="target",
                )
                # Extract point predictions
                if "predictions" in pred_df.columns:
                    preds = pred_df["predictions"].values[:pred_len]
                elif "0.5" in pred_df.columns:
                    preds = pred_df["0.5"].values[:pred_len]
                else:
                    skip = {"item_id", "timestamp", "target_name"}
                    numeric_cols = [c for c in pred_df.select_dtypes(include=[np.number]).columns
                                    if c not in skip]
                    preds = pred_df[numeric_cols[0]].values[:pred_len]

                predictions[start:end] = preds

            except Exception as chunk_err:
                # Per-chunk fallback: univariate predict_quantiles
                ctx_array = np.array(cur_ctx_y, dtype=np.float32)
                _, quantile_forecast = self._pipeline.predict_quantiles(
                    [torch.tensor(ctx_array)],
                    prediction_length=pred_len,
                    quantile_levels=[0.5],
                )
                preds = quantile_forecast[0][:, 0].numpy()[:pred_len]
                predictions[start:end] = preds

            # Expand context with predictions for next batch
            ctx_y = np.concatenate([ctx_y, predictions[start:end]])
            ctx_X = np.vstack([ctx_X, X_test[start:end]])

            if (batch_idx + 1) % 5 == 0 or batch_idx == total_batches - 1:
                pct = (batch_idx + 1) / total_batches * 100
                print(f"      [Chronos-2] batch {batch_idx+1}/{total_batches} ({pct:.0f}%)", flush=True)

        return predictions

    def _predict_univariate(self, X_test: np.ndarray, n_test: int, horizon: int) -> np.ndarray:
        """Fallback: univariate prediction using predict_quantiles (no covariates)."""
        import torch

        predictions = np.full(n_test, np.nan)
        context = self._context_y.copy()
        BATCH_SIZE = 128

        total_batches = (n_test + BATCH_SIZE - 1) // BATCH_SIZE
        for batch_idx, start in enumerate(range(0, n_test, BATCH_SIZE)):
            end = min(start + BATCH_SIZE, n_test)
            pred_len = end - start

            ctx_array = np.array(context[-self.max_context:], dtype=np.float32)
            _, quantile_forecast = self._pipeline.predict_quantiles(
                [torch.tensor(ctx_array)],
                prediction_length=pred_len,
                quantile_levels=[0.5],
            )
            preds = quantile_forecast[0][:, 0].numpy()[:pred_len]
            predictions[start:end] = preds

            # Expand context with predictions
            context = np.concatenate([context, preds])

            if (batch_idx + 1) % 5 == 0 or batch_idx == total_batches - 1:
                pct = (batch_idx + 1) / total_batches * 100
                print(f"      [Chronos-2 univariate] batch {batch_idx+1}/{total_batches} ({pct:.0f}%)", flush=True)

        return predictions

    def release_model(self):
        """Release model from memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    @property
    def fit_time(self) -> float:
        return self._fit_time


# =============================================================================
# TimesFM 2.0 Adapter
# =============================================================================


class TimesFMAdapter:
    """TimesFM 2.0 decoder-only foundation model (HuggingFace transformers).

    Uses TimesFmModelForPrediction from the transformers library for
    zero-shot time series forecasting. TimesFM 2.5 (200M) is not yet
    supported in stable transformers releases (PR #41763 pending), so we
    use the stable 2.0 checkpoint (500M params) instead.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier.
    max_context : int
        Maximum context length for the model.
    device : str
        Device for inference ('cpu', 'cuda', 'mps').
    """

    PARAM_COUNT = 500_000_000  # TimesFM 2.0 published parameter count

    def __init__(
        self,
        model_id: str = "google/timesfm-2.0-500m-pytorch",
        max_context: int = 512,
        device: str = "cpu",
    ):
        self.model_id = model_id
        self.max_context = max_context
        self.device = device

        self._model = None
        self._context_y: Optional[np.ndarray] = None
        self._fit_time: float = 0.0

    @staticmethod
    def is_available() -> bool:
        """Check if TimesFmModelForPrediction is importable from transformers."""
        try:
            from transformers import TimesFmModelForPrediction  # noqa: F401
            return True
        except ImportError:
            return False

    def _load_model(self):
        """Lazy-load the TimesFM model from HuggingFace."""
        if self._model is not None:
            return

        import torch
        from transformers import TimesFmModelForPrediction

        print(f"  [TimesFM] Loading {self.model_id} on {self.device}...")

        # Try bfloat16 first (best for Apple Silicon & modern GPUs)
        try:
            self._model = TimesFmModelForPrediction.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
            )
        except Exception:
            warnings.warn("bfloat16 not supported, falling back to float32")
            self._model = TimesFmModelForPrediction.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,
            )

        self._model.to(self.device)
        self._model.eval()
        print(f"  [TimesFM] Model loaded ({self._model.dtype}).")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> "TimesFMAdapter":
        """Store training context. No weight updates (zero-shot model).

        Parameters
        ----------
        X_train : np.ndarray
            Training features (unused for TimesFM — univariate model).
        y_train : np.ndarray
            Training target values (the time series to forecast).
        """
        t0 = time.time()
        self._load_model()

        # Store the tail of training context
        ctx_len = min(len(y_train), self.max_context)
        self._context_y = y_train[-ctx_len:].astype(np.float32).copy()

        self._fit_time = time.time() - t0
        return self

    def predict(self, X_test: np.ndarray, horizon: int = 1,
                y_actual: Optional[np.ndarray] = None) -> np.ndarray:
        """Forecast using TimesFM with per-point horizon-aware prediction.

        For each test point i, builds context from all observations up to that
        point, forecasts `horizon` steps ahead, and returns the prediction at
        step `horizon` (index horizon-1). When y_actual is provided, context
        grows with real observed values (not predictions).

        Parameters
        ----------
        X_test : np.ndarray
            Test features (n_test, n_features). Only used for determining
            the number of predictions needed.
        horizon : int
            Forecast horizon (number of steps ahead).
        y_actual : np.ndarray, optional
            Actual observed raw values during the test period (n_test,).
            When provided, context is extended with real values instead of
            predictions, giving each test point the best possible context.

        Returns
        -------
        np.ndarray
            Point predictions for each test sample (the horizon-th step).
        """
        if self._model is None:
            raise RuntimeError("Must call fit() before predict()")

        import torch

        n_test = len(X_test)
        predictions = np.full(n_test, np.nan)
        base_context = self._context_y.copy()
        model_dtype = next(self._model.parameters()).dtype

        # Batch multiple contexts per API call for efficiency
        # HF TimesFM accepts past_values as list of tensors
        BATCH_SIZE = 64

        total_batches = (n_test + BATCH_SIZE - 1) // BATCH_SIZE
        for batch_idx, start in enumerate(range(0, n_test, BATCH_SIZE)):
            end = min(start + BATCH_SIZE, n_test)

            # Build one context per test point in this batch
            batch_tensors = []
            for i in range(start, end):
                if y_actual is not None:
                    ctx = np.concatenate([base_context, y_actual[:i]])
                else:
                    ctx = np.concatenate([base_context, predictions[:i][~np.isnan(predictions[:i])]])
                ctx = ctx[-self.max_context:].astype(np.float32)
                batch_tensors.append(
                    torch.tensor(ctx, dtype=model_dtype).to(self.device)
                )

            try:
                with torch.no_grad():
                    # freq=0 → high-frequency data (5-min IoT readings)
                    freq_tensor = torch.tensor(
                        [0] * len(batch_tensors), dtype=torch.long
                    ).to(self.device)
                    output = self._model(
                        past_values=batch_tensors,
                        freq=freq_tensor,
                    )
                # mean_predictions shape: (batch_size, horizon_length)
                # Take the horizon-th step (index horizon-1) for each point
                for j, i in enumerate(range(start, end)):
                    if horizon - 1 < output.mean_predictions.shape[1]:
                        predictions[i] = output.mean_predictions[j, horizon - 1].float().cpu().item()
                    else:
                        predictions[i] = output.mean_predictions[j, -1].float().cpu().item()
            except Exception as e:
                warnings.warn(f"TimesFM forecast failed at batch {batch_idx}: {e}")
                for i in range(start, end):
                    if y_actual is not None:
                        ctx = np.concatenate([base_context, y_actual[:i]])
                    else:
                        ctx = base_context
                    predictions[i] = ctx[-1] if len(ctx) > 0 else np.nan

            if (batch_idx + 1) % 5 == 0 or batch_idx == total_batches - 1:
                pct = (batch_idx + 1) / total_batches * 100
                print(f"      [TimesFM] batch {batch_idx+1}/{total_batches} ({pct:.0f}%)", flush=True)

        return predictions

    def release_model(self):
        """Release model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if hasattr(torch, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except (ImportError, AttributeError):
                pass

    @property
    def fit_time(self) -> float:
        return self._fit_time


# =============================================================================
# TimesFM 2.5 Native Adapter
# =============================================================================


class TimesFM25NativeAdapter:
    """TimesFM 2.5 decoder-only foundation model (native timesfm package).

    Uses the google-research/timesfm package directly for zero-shot
    time series forecasting. This avoids the HuggingFace transformers
    dependency issue (PR #41763 still pending).

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier for weight download.
    max_context : int
        Maximum context length (up to 16384 supported by 2.5).
    max_horizon : int
        Maximum forecast horizon per call.
    """

    PARAM_COUNT = 200_000_000  # TimesFM 2.5 published parameter count

    def __init__(
        self,
        model_id: str = "google/timesfm-2.5-200m-pytorch",
        max_context: int = 1024,
        max_horizon: int = 256,
    ):
        self.model_id = model_id
        self.max_context = max_context
        self.max_horizon = max_horizon

        self._model = None
        self._context_y: Optional[np.ndarray] = None
        self._fit_time: float = 0.0

    @staticmethod
    def is_available() -> bool:
        """Check if the native timesfm package is importable."""
        try:
            import timesfm  # noqa: F401
            return hasattr(timesfm, "TimesFM_2p5_200M_torch")
        except ImportError:
            return False

    def _load_model(self):
        """Lazy-load the TimesFM 2.5 model via native package."""
        if self._model is not None:
            return

        import timesfm

        print(f"  [TimesFM-2.5] Loading {self.model_id} (native)...")
        self._model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.model_id)
        self._model.compile(timesfm.ForecastConfig(
            max_context=self.max_context,
            max_horizon=self.max_horizon,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        ))
        print(f"  [TimesFM-2.5] Model loaded and compiled.")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> "TimesFM25NativeAdapter":
        """Store training context. No weight updates (zero-shot model).

        Parameters
        ----------
        X_train : np.ndarray
            Training features (unused — univariate model).
        y_train : np.ndarray
            Training target values (the time series to forecast).
        """
        t0 = time.time()
        self._load_model()

        # Store the tail of training context
        ctx_len = min(len(y_train), self.max_context)
        self._context_y = y_train[-ctx_len:].astype(np.float32).copy()

        self._fit_time = time.time() - t0
        return self

    def predict(self, X_test: np.ndarray, horizon: int = 1,
                y_actual: Optional[np.ndarray] = None) -> np.ndarray:
        """Forecast using TimesFM 2.5 with per-point horizon-aware prediction.

        For each test point i, builds context from all observations up to that
        point, forecasts `horizon` steps ahead, and returns the prediction at
        step `horizon` (index horizon-1). When y_actual is provided, context
        grows with real observed values (not predictions), ensuring each
        forecast is conditioned on ground truth.

        Parameters
        ----------
        X_test : np.ndarray
            Test features (n_test, n_features). Only used for determining
            the number of predictions needed.
        horizon : int
            Forecast horizon (number of steps ahead).
        y_actual : np.ndarray, optional
            Actual observed raw values during the test period (n_test,).
            When provided, context is extended with real values instead of
            predictions, giving each test point the best possible context.

        Returns
        -------
        np.ndarray
            Point predictions for each test sample (the horizon-th step).
        """
        if self._model is None:
            raise RuntimeError("Must call fit() before predict()")

        n_test = len(X_test)
        predictions = np.full(n_test, np.nan)
        base_context = self._context_y.copy()

        # Batch multiple contexts per API call for efficiency
        BATCH_SIZE = 64

        total_batches = (n_test + BATCH_SIZE - 1) // BATCH_SIZE
        for batch_idx, start in enumerate(range(0, n_test, BATCH_SIZE)):
            end = min(start + BATCH_SIZE, n_test)

            # Build one context per test point in this batch
            batch_contexts = []
            for i in range(start, end):
                if y_actual is not None:
                    # Extend base context with actual observed values up to point i
                    ctx = np.concatenate([base_context, y_actual[:i]])
                else:
                    # Fallback: extend with predictions made so far
                    ctx = np.concatenate([base_context, predictions[:i][~np.isnan(predictions[:i])]])
                # Trim to max_context
                batch_contexts.append(ctx[-self.max_context:].astype(np.float32))

            try:
                # Native API accepts list of arrays — one forecast per context
                point_forecast, _ = self._model.forecast(
                    horizon=horizon,
                    inputs=batch_contexts,
                )
                # point_forecast shape: (batch_size, horizon)
                # Take the horizon-th step (index horizon-1) for each point
                for j, i in enumerate(range(start, end)):
                    predictions[i] = point_forecast[j, horizon - 1]
            except Exception as e:
                warnings.warn(f"TimesFM-2.5 forecast failed at batch {batch_idx}: {e}")
                for i in range(start, end):
                    if y_actual is not None:
                        ctx = np.concatenate([base_context, y_actual[:i]])
                    else:
                        ctx = base_context
                    predictions[i] = ctx[-1] if len(ctx) > 0 else np.nan

            if (batch_idx + 1) % 5 == 0 or batch_idx == total_batches - 1:
                pct = (batch_idx + 1) / total_batches * 100
                print(f"      [TimesFM-2.5] batch {batch_idx+1}/{total_batches} ({pct:.0f}%)", flush=True)

        return predictions

    def release_model(self):
        """Release model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if hasattr(torch, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except (ImportError, AttributeError):
                pass

    @property
    def fit_time(self) -> float:
        return self._fit_time


# =============================================================================
# Factory
# =============================================================================


FOUNDATION_MODELS = {
    "Chronos-2": {
        "class": Chronos2Adapter,
        "params": 120_000_000,
        "description": "Amazon Chronos-2 (encoder-only, covariate-aware)",
        "available": True,
    },
    "TimesFM": {
        "class": TimesFMAdapter,
        "params": 500_000_000,
        "description": "Google TimesFM 2.0 (decoder-only, 500M)",
        "available": TimesFMAdapter.is_available(),
    },
    "TimesFM-2.5": {
        "class": TimesFM25NativeAdapter,
        "params": 200_000_000,
        "description": "Google TimesFM 2.5 (decoder-only, 200M, native)",
        "available": TimesFM25NativeAdapter.is_available(),
    },
}


def get_available_models() -> dict:
    """Return only foundation models whose packages are importable."""
    return {k: v for k, v in FOUNDATION_MODELS.items() if v.get("available", True)}


def create_foundation_model(name: str, **kwargs):
    """Create a foundation model adapter by name.

    Parameters
    ----------
    name : str
        Model name: 'Chronos-2' or 'TimesFM'
    **kwargs
        Additional keyword arguments passed to the adapter constructor.

    Returns
    -------
    Chronos2Adapter or TimesFMAdapter

    Raises
    ------
    ValueError
        If model name is unknown.
    ImportError
        If the model's package is not installed.
    """
    if name not in FOUNDATION_MODELS:
        raise ValueError(f"Unknown foundation model: {name}. Available: {list(FOUNDATION_MODELS.keys())}")
    if not FOUNDATION_MODELS[name].get("available", True):
        raise ImportError(
            f"{name} is not available in this environment. "
            f"Check Python version and package requirements."
        )
    return FOUNDATION_MODELS[name]["class"](**kwargs)
