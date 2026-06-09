#!/usr/bin/env python3
"""
Foundation Model Runner - Chronos-2 & TimesFM
==============================================
Version: 1.0.0 (2026-02-24)

Runs Chronos-2 and TimesFM foundation models across all 10 sensors and
3 horizons using 3-fold expanding window walk-forward validation.

Uses FoldManager.make_folds_idx() from analysis/walkforward_backtest.py
to generate identical fold splits as the existing 18-model experiments.

Memory management: loads one foundation model at a time, releases before
loading the next to stay within 16GB RAM.

Output: foundation_model_results.csv with columns:
  model, dataset, horizon, fold, RMSE, MAE, MAPE, R2, time_seconds, parameter_count

Usage:
    python run_foundation_models.py [--models chronos2 timesfm] [--sensors SENSOR1 ...]
"""

import argparse
import gc
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
CLEANED_DIR = BASE_DIR / "cleaned_data"

# All 10 sensors
ALL_SENSORS = [
    "ZWave2_Metered_Wall_Plug_Switch_computers_Sensor_power",
    "ZWave_Node_016_the_whole_multiplier_over_the_window_Sensor_power",
    "ZWave_Node_042_ZW095_Home_Energy_Meter__Gen_5_Electric_meter_watts",
    "ZWave_Node_042_ZW095_Home_Energy_Meter__Gen_5_Electric_meter_watts_2",
    "Wall_Plug_2_Sensor_power",
    "WavePlug_UPS_Sensor_power",
    "ZWave5_network_switch_Sensor_power",
    "ZWave7_bookcase_Sensor_power",
    "ZWave8_on_desk_Sensor_power",
    "ZWave_Node_031_SERVER_SB_Sensor_power",
]

HORIZONS = [1, 5, 12]

# Walk-forward parameters (must match run_complete_benchmark.py)
N_FOLDS = 3
TRAIN_RATIO = 0.60
VAL_RATIO = 0.10
TEST_RATIO = 0.30


def make_folds_idx(n, train_ratio, val_ratio, test_ratio, n_folds):
    """Generate expanding window fold indices.

    Reimplemented here to avoid importing the full walkforward_backtest module.
    Logic matches FoldManager.make_folds_idx() exactly.
    """
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
        folds.append((0, train_end, val_end, test_end))

    return folds


def load_dataset(dataset_name):
    """Load clean_full.csv and return DataFrame with feature/target info."""
    data_path = CLEANED_DIR / dataset_name / "clean_full.csv"
    if not data_path.exists():
        return None

    df = pd.read_csv(data_path)

    # Identify timestamp column
    ts_col = "timestamp" if "timestamp" in df.columns else "ts_utc"

    # Feature columns (exclude timestamp and targets)
    exclude_patterns = ["timestamp", "ts_utc", "value_t+"]
    feature_cols = [c for c in df.columns if not any(p in c for p in exclude_patterns)]

    return df, feature_cols, ts_col


def compute_metrics(y_true, y_pred):
    """Compute RMSE, MAE, MAPE, R2."""
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    if mask.sum() == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan, "R2": np.nan}

    yt, yp = y_true[mask], y_pred[mask]
    rmse_val = np.sqrt(mean_squared_error(yt, yp))
    mae_val = mean_absolute_error(yt, yp)
    r2_val = r2_score(yt, yp)

    # MAPE with protection against zeros
    nonzero = np.abs(yt) > 1e-6
    if nonzero.sum() > 0:
        mape_val = np.mean(np.abs((yt[nonzero] - yp[nonzero]) / yt[nonzero])) * 100
    else:
        mape_val = np.nan

    return {"RMSE": rmse_val, "MAE": mae_val, "MAPE": mape_val, "R2": r2_val}


def run_foundation_model(model_name, model_adapter, datasets, horizons, n_folds=3):
    """Run one foundation model across all datasets and horizons.

    Parameters
    ----------
    model_name : str
        Display name ('Chronos-2' or 'TimesFM').
    model_adapter : object
        Adapter instance with fit/predict interface.
    datasets : list of str
        Sensor dataset names.
    horizons : list of int
        Forecast horizons in steps.
    n_folds : int
        Number of expanding window folds.

    Returns
    -------
    list of dict
        Results for each (dataset, horizon, fold) combination.
    """
    all_results = []
    total_experiments = len(datasets) * len(horizons) * n_folds
    experiment_num = 0

    for ds_idx, dataset_name in enumerate(datasets):
        short_name = dataset_name[:40]
        print(f"\n  [{ds_idx+1}/{len(datasets)}] Dataset: {short_name}...")

        result = load_dataset(dataset_name)
        if result is None:
            print(f"    Not found, skipping.")
            continue

        df, feature_cols, ts_col = result

        for horizon in horizons:
            target_col = f"value_t+{horizon}"
            if target_col not in df.columns:
                print(f"    Target {target_col} not found, skipping.")
                continue

            # Prepare data
            X = df[feature_cols].values
            y = df[target_col].values
            y_raw = df["value"].values  # raw time series for foundation models

            # Remove NaN
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X, y, y_raw_clean = X[mask], y[mask], y_raw[mask]

            n = len(X)
            folds = make_folds_idx(n, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, n_folds)

            for fold_idx, (tr0, tr1, v1, te1) in enumerate(folds):
                X_train = X[tr0:tr1]
                y_train = y[tr0:tr1]
                y_raw_train = y_raw_clean[tr0:tr1]
                X_test = X[v1:te1]
                y_test = y[v1:te1]

                try:
                    experiment_num += 1
                    n_test_samples = te1 - v1
                    print(f"    [{experiment_num}/{total_experiments}] H{horizon} fold{fold_idx} "
                          f"(train={tr1}, test={n_test_samples})...", flush=True)

                    t0 = time.time()

                    # Foundation models use raw time series as context
                    # Scale features for Chronos-2 covariates
                    scaler = StandardScaler()
                    X_train_s = scaler.fit_transform(X_train)
                    X_test_s = scaler.transform(X_test)

                    model_adapter.fit(X_train_s, y_raw_train, feature_names=feature_cols)
                    y_pred = model_adapter.predict(X_test_s, horizon=horizon)

                    elapsed = time.time() - t0

                    # Handle prediction length mismatch
                    min_len = min(len(y_test), len(y_pred))
                    metrics = compute_metrics(y_test[:min_len], y_pred[:min_len])

                    print(f"    -> RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f} ({elapsed:.1f}s)")

                    all_results.append({
                        "model": model_name,
                        "dataset": dataset_name,
                        "horizon": f"H{horizon}",
                        "fold": fold_idx,
                        "RMSE": round(metrics["RMSE"], 4) if np.isfinite(metrics["RMSE"]) else None,
                        "MAE": round(metrics["MAE"], 4) if np.isfinite(metrics["MAE"]) else None,
                        "MAPE": round(metrics["MAPE"], 4) if metrics["MAPE"] is not None and np.isfinite(metrics["MAPE"]) else None,
                        "R2": round(metrics["R2"], 4) if np.isfinite(metrics["R2"]) else None,
                        "time_seconds": round(elapsed, 2),
                        "parameter_count": model_adapter.PARAM_COUNT,
                        "type": "Foundation",
                        "status": "SUCCESS",
                    })

                except Exception as e:
                    print(f"    H{horizon} fold{fold_idx}: FAILED - {e}")
                    all_results.append({
                        "model": model_name,
                        "dataset": dataset_name,
                        "horizon": f"H{horizon}",
                        "fold": fold_idx,
                        "status": "FAILED",
                        "error": str(e)[:200],
                        "type": "Foundation",
                        "parameter_count": model_adapter.PARAM_COUNT,
                    })

            # Compute mean across folds for this dataset/horizon
            successful_folds = [r for r in all_results
                                if r.get("status") == "SUCCESS"
                                and r.get("dataset") == dataset_name
                                and r.get("horizon") == f"H{horizon}"]
            if successful_folds:
                avg_rmse = np.mean([r["RMSE"] for r in successful_folds])
                print(f"    H{horizon} mean RMSE: {avg_rmse:.4f}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run foundation model benchmarks")
    parser.add_argument("--models", nargs="+", default=["chronos2", "timesfm"],
                        choices=["chronos2", "timesfm"],
                        help="Which foundation models to run")
    parser.add_argument("--sensors", nargs="+", default=None,
                        help="Specific sensors to run (default: all 10)")
    parser.add_argument("--device", default="cpu",
                        help="Device for inference (cpu, cuda, mps)")
    parser.add_argument("--output", default="foundation_model_results.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    datasets = args.sensors if args.sensors else ALL_SENSORS

    print("=" * 70)
    print("FOUNDATION MODEL BENCHMARK")
    print("=" * 70)
    print(f"Models: {args.models}")
    print(f"Datasets: {len(datasets)}")
    print(f"Horizons: {HORIZONS}")
    print(f"Folds: {N_FOLDS}")
    print(f"Device: {args.device}")
    total_runs = len(args.models) * len(datasets) * len(HORIZONS) * N_FOLDS
    print(f"Total experiments: {total_runs}")
    print("=" * 70)

    all_results = []

    # Run each foundation model sequentially (memory management)
    for model_key in args.models:
        if model_key == "chronos2":
            print(f"\n{'='*70}")
            print("Running Chronos-2...")
            print(f"{'='*70}")

            from common.foundation_models import Chronos2Adapter
            adapter = Chronos2Adapter(device=args.device)

            results = run_foundation_model(
                "Chronos-2", adapter, datasets, HORIZONS, N_FOLDS
            )
            all_results.extend(results)

            # Release model memory before loading next
            adapter.release_model()
            del adapter
            gc.collect()

        elif model_key == "timesfm":
            from common.foundation_models import TimesFMAdapter
            if not TimesFMAdapter.is_available():
                print(f"\n{'='*70}")
                print("SKIPPING TimesFM 2.5 — package not installed.")
                print("TimesFM requires Python >=3.10,<3.12.")
                print(f"{'='*70}")
                continue

            print(f"\n{'='*70}")
            print("Running TimesFM 2.5...")
            print(f"{'='*70}")

            adapter = TimesFMAdapter(device=args.device)

            results = run_foundation_model(
                "TimesFM", adapter, datasets, HORIZONS, N_FOLDS
            )
            all_results.extend(results)

            adapter.release_model()
            del adapter
            gc.collect()

    # Save results
    output_path = BASE_DIR / args.output
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_path, index=False)

    # Summary
    success_count = sum(1 for r in all_results if r.get("status") == "SUCCESS")
    fail_count = sum(1 for r in all_results if r.get("status") == "FAILED")

    print(f"\n{'='*70}")
    print("FOUNDATION MODEL BENCHMARK COMPLETE")
    print(f"{'='*70}")
    print(f"Successful: {success_count}/{len(all_results)}")
    print(f"Failed: {fail_count}/{len(all_results)}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
