# common/features.py
from __future__ import annotations
import numpy as np
import pandas as pd

def add_lags(df: pd.DataFrame, col: str, lags=(1,2,3,6,12,24)) -> pd.DataFrame:
    for L in lags:
        df[f"{col}_lag{L}"] = df[col].shift(L)
    return df

def add_rolling(df: pd.DataFrame, col: str, windows=(3,6,12,24)) -> pd.DataFrame:
    """Add rolling statistics with causal shift to prevent data leakage.

    Uses .shift(1) before rolling to ensure only past values are used,
    preventing information from time t leaking into predictions for t+h.
    """
    for w in windows:
        # IMPORTANT: shift(1) ensures we only use past values (causal)
        s = df[col].shift(1).rolling(w, min_periods=max(1, w//2))
        df[f"{col}_rollmean{w}"] = s.mean()
        df[f"{col}_rollstd{w}"]  = s.std()
    return df

def add_diff(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[f"{col}_diff1"] = df[col].diff()
    df[f"{col}_pct1"]  = df[col].pct_change().replace([np.inf, -np.inf], np.nan)
    return df

def make_supervised(df: pd.DataFrame, y_col: str, horizon: int = 6) -> pd.DataFrame:
    """Δημιουργεί y(t+h). Για 5min step, horizon=6 => 30 λεπτά μπροστά."""
    df = df.copy()
    df[f"{y_col}_tplus{horizon}"] = df[y_col].shift(-horizon)
    return df