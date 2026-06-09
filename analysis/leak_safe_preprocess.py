"""
Leak-Safe Preprocessing for Walk-Forward Validation
====================================================
Version: 1.0.0 (2026-02-24)

Provides LeakSafePreprocessor that fits preprocessing parameters (quantile
bounds) on training data only, then applies them consistently to any split.
This prevents test-set information from leaking into training-set preprocessing.

Used by LeakSafeFoldBuilder in walkforward_backtest.py.
"""

from __future__ import annotations

import copy
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


class LeakSafePreprocessor:
    """Fit preprocessing parameters on training data only, apply to any split.

    Interface contract (required by LeakSafeFoldBuilder):
        - Attributes: timestamp_col, value_col, history_lookback
        - Methods: clone(), fit(df_train), transform(df_subset)
    """

    def __init__(
        self,
        value_col: str = "value",
        timestamp_col: str = "ts_utc",
        clip_quantiles: Optional[Tuple[float, float]] = (0.01, 0.99),
        use_hampel: bool = True,
        hampel_window: int = 25,
        hampel_sigma: float = 3.0,
        daily_fourier_K: int = 3,
        weekly_fourier_K: int = 1,
        lags: Optional[List[int]] = None,
        rolls: Optional[List[int]] = None,
        add_temporal: bool = True,
        freq_seconds: int = 300,
    ):
        self.value_col = value_col
        self.timestamp_col = timestamp_col
        self.clip_quantiles = clip_quantiles
        self.use_hampel = use_hampel
        self.hampel_window = hampel_window
        self.hampel_sigma = hampel_sigma
        self.daily_fourier_K = daily_fourier_K
        self.weekly_fourier_K = weekly_fourier_K
        self.lags = lags if lags is not None else [1, 5, 10, 30, 60, 120, 720]
        self.rolls = rolls if rolls is not None else [5, 10, 30, 60, 120]
        self._add_temporal = add_temporal
        self.freq_seconds = freq_seconds

        # Fitted state
        self._qlo: Optional[float] = None
        self._qhi: Optional[float] = None
        self._train_tail: Optional[np.ndarray] = None
        self._is_fitted: bool = False

    @property
    def history_lookback(self) -> int:
        """Number of historical rows needed for lags/rolling at split boundaries."""
        max_lag = max(self.lags) if self.lags else 0
        max_roll = max(self.rolls) if self.rolls else 0
        return max(max_lag, max_roll, self.hampel_window) + 1

    def fit(self, df_train: pd.DataFrame) -> "LeakSafePreprocessor":
        """Fit preprocessing parameters from training data only."""
        series = df_train[self.value_col]

        # Fit quantile bounds
        if self.clip_quantiles is not None:
            lo, hi = self.clip_quantiles
            self._qlo = float(series.quantile(lo))
            self._qhi = float(series.quantile(hi))

        # Store training tail for Hampel context on val/test splits
        tail_len = self.history_lookback
        self._train_tail = series.iloc[-tail_len:].values.copy()

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted preprocessing to any data split."""
        if not self._is_fitted:
            raise RuntimeError("LeakSafePreprocessor must be fitted before transform()")

        out = df.copy()
        vcol = self.value_col
        tcol = self.timestamp_col

        # 1. Quantile clipping with train-derived bounds
        if self._qlo is not None and self._qhi is not None:
            out[vcol] = out[vcol].clip(self._qlo, self._qhi)

        # 2. Hampel filter (causal: center=False, so no look-ahead)
        if self.use_hampel:
            from hampel_with_log import hampel_filter_with_log
            cleaned, _ = hampel_filter_with_log(
                out[vcol], window_size=self.hampel_window, n_sigmas=self.hampel_sigma
            )
            out[vcol] = cleaned

        # 3. Temporal features
        if self._add_temporal and tcol in out.columns:
            ts = out[tcol]
            if not pd.api.types.is_datetime64_any_dtype(ts):
                ts = pd.to_datetime(ts, utc=True, errors="coerce")
            out["hour"] = ts.dt.hour
            out["dow"] = ts.dt.dayofweek
            out["is_weekend"] = (out["dow"] >= 5).astype(int)
            out["month"] = ts.dt.month

        # 4. Lag features
        for lag in self.lags:
            out[f"{vcol}_lag{lag}"] = out[vcol].shift(lag)

        # 5. Rolling statistics (causal: shift(1) before rolling)
        for w in self.rolls:
            s = out[vcol].shift(1).rolling(w, min_periods=max(1, w // 2))
            out[f"{vcol}_roll{w}_mean"] = s.mean()
            out[f"{vcol}_roll{w}_std"] = s.std()
            out[f"{vcol}_roll{w}_min"] = s.min()
            out[f"{vcol}_roll{w}_max"] = s.max()

        # 6. Fourier terms
        daily_rows = int(round(86400 / self.freq_seconds)) if self.freq_seconds > 0 else 288
        weekly_rows = daily_rows * 7
        t = np.arange(len(out))
        if self.daily_fourier_K > 0 and daily_rows > 0:
            for k in range(1, self.daily_fourier_K + 1):
                ang = 2.0 * np.pi * k * t / float(daily_rows)
                out[f"fourier_daily_sin{k}"] = np.sin(ang)
                out[f"fourier_daily_cos{k}"] = np.cos(ang)
        if self.weekly_fourier_K > 0 and weekly_rows > 0:
            for k in range(1, self.weekly_fourier_K + 1):
                ang = 2.0 * np.pi * k * t / float(weekly_rows)
                out[f"fourier_weekly_sin{k}"] = np.sin(ang)
                out[f"fourier_weekly_cos{k}"] = np.cos(ang)

        return out

    def clone(self) -> "LeakSafePreprocessor":
        """Return a fresh unfitted copy with the same configuration."""
        new = LeakSafePreprocessor(
            value_col=self.value_col,
            timestamp_col=self.timestamp_col,
            clip_quantiles=self.clip_quantiles,
            use_hampel=self.use_hampel,
            hampel_window=self.hampel_window,
            hampel_sigma=self.hampel_sigma,
            daily_fourier_K=self.daily_fourier_K,
            weekly_fourier_K=self.weekly_fourier_K,
            lags=list(self.lags),
            rolls=list(self.rolls),
            add_temporal=self._add_temporal,
            freq_seconds=self.freq_seconds,
        )
        return new


def add_targets(
    df: pd.DataFrame, value_col: str, horizons: Tuple[int, ...]
) -> pd.DataFrame:
    """Add forecast target columns. Module-level function for LeakSafeFoldBuilder."""
    for h in horizons:
        df[f"{value_col}_t+{h}"] = df[value_col].shift(-h)
    return df
