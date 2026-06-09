# common/io.py
from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Optional

def read_sensor_csv(path: str | Path,
                    ts_col: str = "ts_utc",
                    value_col: str = "value",
                    parse_dates: bool = True) -> pd.DataFrame:
    """Διαβάζει standard CSV από τα exports, κρατά [ts_utc, value]."""
    path = Path(path)
    df = pd.read_csv(path)
    if parse_dates and ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    keep = [c for c in [ts_col, value_col] if c in df.columns]
    df = df[keep].dropna()
    df = df.sort_values(ts_col).reset_index(drop=True)
    return df

def resample(df: pd.DataFrame,
             rule: str = "5min",
             ts_col: str = "ts_utc",
             value_col: str = "value",
             how: str = "mean") -> pd.DataFrame:
    """Ομογενοποίηση δειγματοληψίας σε σταθερό βήμα."""
    df = df.set_index(ts_col)
    if how == "mean":
        out = df[value_col].resample(rule).mean()
    elif how == "sum":
        out = df[value_col].resample(rule).sum()
    elif how == "median":
        out = df[value_col].resample(rule).median()
    else:
        out = df[value_col].resample(rule).mean()
    out = out.interpolate(limit_direction="both")
    return out.to_frame(name=value_col).reset_index()

def add_calendar(df: pd.DataFrame, ts_col: str = "ts_utc") -> pd.DataFrame:
    """Calendar features (χωρίς εξωτερικά deps)."""
    t = pd.to_datetime(df[ts_col], utc=True)
    df["hour"] = t.dt.hour
    df["dow"] = t.dt.dayofweek
    df["dom"] = t.dt.day
    df["month"] = t.dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    return df