#!/usr/bin/env python3
"""
Energy Dataset Cleaning – v3 (Fourier Seasonality + Weekly Lag)
===============================================================
Version: 2.0.0 (2025-12-31)

Preprocessing pipeline for IoT energy sensor data:
- Timestamp parsing and resampling to 5-minute intervals
- Missing value interpolation
- Outlier detection with Hampel filter
- Feature engineering: rolling stats, Fourier seasonality, weekly lag
- Multi-horizon target creation (H1, H5, H12)

Usage:
    python clean_energy_dataset_v3.py --sensor SENSOR_NAME
"""

import argparse
from pathlib import Path
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from hampel_with_log import hampel_filter_with_log

def load_csv(path, timestamp_col, value_col):
    df = pd.read_csv(path)
    if timestamp_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"Missing columns: {timestamp_col}, {value_col}")
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
    df = df.dropna(subset=[timestamp_col, value_col])
    df = df.sort_values(timestamp_col).drop_duplicates(subset=[timestamp_col])
    return df

def resample_uniform(df, timestamp_col, value_col, rule=None, origin=None, offset=None):
    if not rule:
        return df
    idxed = df.set_index(timestamp_col).sort_index()[[value_col]]
    try:
        # pandas >= 1.1 supports origin/offset
        r = idxed.resample(rule, origin=origin, offset=offset).mean()
    except TypeError:
        # Fallback if origin/offset not supported
        r = idxed.resample(rule).mean()
    r[value_col] = r[value_col].interpolate("time").ffill().bfill()
    r.reset_index(inplace=True)
    return r

def clip_outliers_quantiles(df, value_col, lo=0.001, hi=0.999):
    qlo, qhi = df[value_col].quantile([lo, hi])
    df[value_col] = df[value_col].clip(qlo, qhi)
    return df, (float(qlo), float(qhi))

def apply_hampel(df, value_col, ts_col, k=7, n_sig=3.0, log_path: Path | None = None):
    cleaned_series, outliers = hampel_filter_with_log(df[value_col], window_size=k, n_sigmas=n_sig)
    df[value_col] = cleaned_series
    if log_path is not None and not outliers.empty:
        outliers = outliers.copy()
        outliers["timestamp"] = df.loc[outliers["index"], ts_col].values
        outliers.to_csv(log_path, index=False)
    return df, len(outliers)

def add_temporal(df, timestamp_col):
    ts = df[timestamp_col].dt
    df["hour"] = ts.hour
    df["dow"] = ts.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["month"] = ts.month
    return df

def add_lags(df, value_col, lags):
    for L in lags:
        df[f"{value_col}_lag{L}"] = df[value_col].shift(L)
    return df

def add_rollings(df, value_col, windows):
    for w in windows:
        s = df[value_col].shift(1)  # causal
        df[f"{value_col}_roll{w}_mean"] = s.rolling(w).mean()
        df[f"{value_col}_roll{w}_std"]  = s.rolling(w).std()
        df[f"{value_col}_roll{w}_min"]  = s.rolling(w).min()
        df[f"{value_col}_roll{w}_max"]  = s.rolling(w).max()
    return df

def add_fourier_terms(df, period_rows, K, label):
    """Add K pairs of sin/cos seasonal terms for a given period (in rows)."""
    if period_rows <= 0:
        return df
    t = np.arange(len(df))
    for k in range(1, K+1):
        ang = 2.0 * np.pi * k * t / float(period_rows)
        df[f"fourier_{label}_sin{k}"] = np.sin(ang)
        df[f"fourier_{label}_cos{k}"] = np.cos(ang)
    return df

def add_targets(df, value_col, horizons):
    for h in horizons:
        df[f"{value_col}_t+{h}"] = df[value_col].shift(-h)
    return df

def time_split(df, test_size=0.2):
    cut = int(len(df) * (1 - test_size))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()

def make_plot(df, timestamp_col, value_col, path):
    plt.figure(figsize=(10,3))
    plt.plot(df[timestamp_col], df[value_col])
    plt.xlabel("time"); plt.ylabel(value_col)
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()

def main(args):
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)

    print(f"[fourier] daily K={args.daily_fourier_K}, weekly K={args.weekly_fourier_K}")
    print(f"[lags] using: {args.lags}")

    df = load_csv(args.input, args.timestamp, args.value)
    if args.resample:
        df = resample_uniform(df, args.timestamp, args.value, args.resample,
                              origin=args.resample_origin, offset=args.resample_offset)
    if args.non_negative: df[args.value] = df[args.value].clip(lower=0)

    # --- LEAK-SAFE: split BEFORE quantile clipping ---
    # Compute quantile bounds from training portion only, then apply to all data.
    # This prevents test-set distribution from influencing clipping thresholds.
    split_idx = int(len(df) * (1 - args.test_size))
    if args.clip_quantiles:
        lo, hi = args.clip_quantiles
        train_series = df[args.value].iloc[:split_idx]
        qlo, qhi = float(train_series.quantile(lo)), float(train_series.quantile(hi))
        df[args.value] = df[args.value].clip(qlo, qhi)
        print(f"[clip] quantiles (from train only, n={split_idx}): [{qlo:.3f}, {qhi:.3f}]")

    # Hampel filter is causal (center=False) so applying to full series in
    # temporal order is leak-safe: each point's threshold uses only past values.
    if args.hampel:
        log_file = Path(args.output_dir) / "outliers_hampel.csv" if args.hampel_log else None
        df, n = apply_hampel(df, args.value, args.timestamp,
                             k=args.hampel_window, n_sig=args.hampel_sigma,
                             log_path=log_file)
        print(f"[hampel] replaced points: {n}")

    df = add_temporal(df, args.timestamp)
    if args.lags:  df = add_lags(df, args.value, args.lags)
    if args.rolls: df = add_rollings(df, args.value, args.rolls)

    daily_rows = int(round(86400 / args.freq_s))
    weekly_rows = int(round(7 * 86400 / args.freq_s))
    if args.daily_fourier_K > 0:
        df = add_fourier_terms(df, daily_rows, args.daily_fourier_K, label="daily")
    if args.weekly_fourier_K > 0:
        df = add_fourier_terms(df, weekly_rows, args.weekly_fourier_K, label="weekly")

    df = add_targets(df, args.value, args.horizons)

    required_cols = {args.timestamp, args.value}
    keep_cols = []
    for c in df.columns:
        if c in required_cols or df[c].notna().any():
            keep_cols.append(c)
    df = df.loc[:, keep_cols]

    before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"[dropna] removed rows: {before - len(df)}")

    train, test = time_split(df, test_size=args.test_size)

    raw_path = outdir / "clean_full.csv"
    train_path = outdir / "train.csv"
    test_path = outdir / "test.csv"
    plot_path = outdir / "overview.png"

    df.to_csv(raw_path, index=False)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    make_plot(train, args.timestamp, args.value, plot_path)

    print("Saved:")
    print(" -", raw_path)
    print(" -", train_path)
    print(" -", test_path)
    print(" -", plot_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--timestamp", default="ts_utc")
    ap.add_argument("--value", default="value")
    ap.add_argument("--resample", default=None, help="e.g., 60s")
    ap.add_argument("--resample-origin", dest="resample_origin", default=None, help="e.g., start_day, start, epoch, or timestamp")
    ap.add_argument("--resample-offset", dest="resample_offset", default=None, help="e.g., 0min, 30s")
    ap.add_argument("--freq-s", type=int, default=60, help="Sampling frequency in seconds (used for Fourier periods)")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--clip-quantiles", dest="clip_quantiles", nargs=2, type=float, default=[0.001, 0.999])
    ap.add_argument("--hampel", action="store_true")
    ap.add_argument("--hampel-window", type=int, default=7)
    ap.add_argument("--hampel-sigma", type=float, default=3.0)
    ap.add_argument("--hampel-log", action="store_true", help="Write outliers_hampel.csv with replaced points")
    ap.add_argument("--lags", nargs="*", type=int, default=[1,5,10,30,60,120,1440,10080,20160])
    ap.add_argument("--rolls", nargs="*", type=int, default=[5,10,30,60,120,1440])
    ap.add_argument("--daily-fourier-K", type=int, default=5)
    ap.add_argument("--weekly-fourier-K", type=int, default=1)
    ap.add_argument("--horizons", nargs="*", type=int, default=[1,5,10])
    ap.add_argument("--non-negative", action="store_true")
    args = ap.parse_args()
    main(args)
