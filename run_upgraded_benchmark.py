#!/usr/bin/env python3
"""
Upgraded Benchmark - Master Orchestrator
=========================================
Version: 1.0.0 (2026-02-24)

Runs the full 20-model benchmark (18 existing + Chronos-2 + TimesFM) across
all 10 sensors and 3 horizons, then merges results into a single CSV.

Output: BENCHMARK_RESULTS_FULL.csv with columns:
    model, dataset, horizon, RMSE, MAE, MAPE, R2, time_seconds, parameter_count, type, status

Execution order:
1. run_complete_benchmark.py — ML models (Naive, Ridge, ElasticNet, RF, ExtraTrees,
   CatBoost, LightGBM, XGBoost, NGBoost, KNN, Prophet, SARIMAX, PatchTST, TFT)
2. run_deep_learning_all.py — DL models (LSTM, GRU, BiLSTM)
3. run_foundation_models.py — Foundation models (Chronos-2, TimesFM)
4. Merge all CSV results into BENCHMARK_RESULTS_FULL.csv

Usage:
    # Full benchmark (all 20 models)
    python run_upgraded_benchmark.py

    # Skip stages that already completed (resume after partial run)
    python run_upgraded_benchmark.py --skip-ml --skip-dl

    # Foundation models only
    python run_upgraded_benchmark.py --skip-ml --skip-dl --device cpu

    # Merge only (all CSVs already exist)
    python run_upgraded_benchmark.py --merge-only
"""

import argparse
import glob
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
VENV_PYTHON = str(BASE_DIR / ".venv" / "bin" / "python3")

# Output paths
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV = BASE_DIR / "BENCHMARK_RESULTS_FULL.csv"


def run_script(script_name, args=None, description=""):
    """Run a Python script as subprocess."""
    cmd = [VENV_PYTHON, str(BASE_DIR / script_name)]
    if args:
        cmd.extend(args)

    print(f"\n{'='*70}")
    print(f"RUNNING: {description or script_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"WARNING: {script_name} exited with code {result.returncode}")
    else:
        print(f"OK: {script_name} completed in {elapsed/60:.1f} minutes")

    return result.returncode


def find_latest_results_dir():
    """Find the most recent FINAL_RESULTS_* directory."""
    dirs = sorted(BASE_DIR.glob("FINAL_RESULTS_*"), reverse=True)
    return dirs[0] if dirs else None


def merge_results(skip_ml=False, skip_dl=False, skip_foundation=False):
    """Merge all result CSVs into BENCHMARK_RESULTS_FULL.csv."""
    print(f"\n{'='*70}")
    print("MERGING ALL RESULTS")
    print(f"{'='*70}")

    all_dfs = []

    # 1. ML results from run_complete_benchmark.py
    if not skip_ml:
        results_dir = find_latest_results_dir()
        if results_dir:
            ml_csv = results_dir / "all_results.csv"
            if ml_csv.exists():
                df_ml = pd.read_csv(ml_csv)
                print(f"  ML results: {len(df_ml)} rows from {ml_csv}")
                all_dfs.append(df_ml)
            else:
                print(f"  ML results: NOT FOUND at {ml_csv}")
        else:
            print("  ML results: No FINAL_RESULTS_* directory found")

    # 2. DL results from run_deep_learning_all.py
    if not skip_dl:
        dl_dir = BASE_DIR / "ml_results_deep_learning"
        if dl_dir.exists():
            dl_csvs = [p for p in dl_dir.glob("*.csv") if not p.name.startswith("._")]
            for csv_path in dl_csvs:
                try:
                    df_dl = pd.read_csv(csv_path)
                    if len(df_dl) > 0:
                        # Normalize column names
                        if "type" not in df_dl.columns:
                            df_dl["type"] = "DeepLearning"
                        if "status" not in df_dl.columns:
                            df_dl["status"] = "SUCCESS"
                        print(f"  DL results: {len(df_dl)} rows from {csv_path.name}")
                        all_dfs.append(df_dl)
                except Exception as e:
                    print(f"  DL results: error reading {csv_path.name}: {e}")

        # Also check the JSON results
        dl_json_path = dl_dir / "deep_learning_results.json"
        if dl_json_path.exists() and not dl_csvs:
            import json
            with open(dl_json_path) as f:
                dl_data = json.load(f)
            if isinstance(dl_data, list):
                df_dl = pd.DataFrame(dl_data)
                if "status" not in df_dl.columns:
                    df_dl["status"] = "SUCCESS"
                df_dl["type"] = "DeepLearning"
                print(f"  DL results (JSON): {len(df_dl)} rows")
                all_dfs.append(df_dl)

    # 3. Foundation model results
    if not skip_foundation:
        fm_csv = BASE_DIR / "foundation_model_results.csv"
        if fm_csv.exists():
            df_fm = pd.read_csv(fm_csv)
            print(f"  Foundation results: {len(df_fm)} rows (raw) from {fm_csv}")

            # Average across folds to match ML/DL format (30 rows per model)
            if "fold" in df_fm.columns:
                group_cols = ["model", "dataset", "horizon"]
                metric_cols = ["RMSE", "MAE", "MAPE", "R2", "time_seconds"]
                first_cols = ["parameter_count", "type", "status"]
                agg_dict = {c: "mean" for c in metric_cols if c in df_fm.columns}
                agg_dict.update({c: "first" for c in first_cols if c in df_fm.columns})
                df_fm = df_fm.groupby(group_cols, as_index=False).agg(agg_dict)
                print(f"  Foundation results: {len(df_fm)} rows after fold-averaging")

            all_dfs.append(df_fm)
        else:
            print(f"  Foundation results: NOT FOUND at {fm_csv}")

    # 4. Chronos-1 results (from previous benchmark_chronos.py runs)
    chronos_csv = BASE_DIR / "oos_validation_chronos_results.csv"
    if chronos_csv.exists():
        try:
            df_c1 = pd.read_csv(chronos_csv)
            if len(df_c1) > 0:
                # Rename sensor -> dataset to match other dataframes
                if "sensor" in df_c1.columns and "dataset" not in df_c1.columns:
                    df_c1 = df_c1.rename(columns={"sensor": "dataset"})
                df_c1["type"] = "Foundation"
                df_c1["model"] = "Chronos-1"
                df_c1["parameter_count"] = 200_000_000
                df_c1["status"] = "SUCCESS"
                if "MAPE" not in df_c1.columns:
                    df_c1["MAPE"] = None
                if "time_seconds" not in df_c1.columns:
                    df_c1["time_seconds"] = None
                print(f"  Chronos-1 results: {len(df_c1)} rows")
                all_dfs.append(df_c1)
        except Exception as e:
            print(f"  Chronos-1 results: error reading: {e}")

    # 5. TimesFM results (from run_timesfm_only.py)
    timesfm_csv = BASE_DIR / "timesfm_results.csv"
    if timesfm_csv.exists():
        try:
            df_tfm = pd.read_csv(timesfm_csv)
            if len(df_tfm) > 0:
                print(f"  TimesFM results: {len(df_tfm)} rows (raw) from {timesfm_csv}")

                # Average across folds to match ML/DL format
                if "fold" in df_tfm.columns:
                    group_cols = ["model", "dataset", "horizon"]
                    metric_cols = ["RMSE", "MAE", "MAPE", "R2", "time_seconds"]
                    first_cols = ["parameter_count", "type", "status"]
                    agg_dict = {c: "mean" for c in metric_cols if c in df_tfm.columns}
                    agg_dict.update({c: "first" for c in first_cols if c in df_tfm.columns})
                    df_tfm = df_tfm.groupby(group_cols, as_index=False).agg(agg_dict)
                    print(f"  TimesFM results: {len(df_tfm)} rows after fold-averaging")

                all_dfs.append(df_tfm)
        except Exception as e:
            print(f"  TimesFM results: error reading: {e}")

    # 6. TimesFM 2.5 results (from run_timesfm25_only.py)
    timesfm25_csv = BASE_DIR / "timesfm25_results.csv"
    if timesfm25_csv.exists():
        try:
            df_tfm25 = pd.read_csv(timesfm25_csv)
            if len(df_tfm25) > 0:
                print(f"  TimesFM-2.5 results: {len(df_tfm25)} rows (raw) from {timesfm25_csv}")

                # Average across folds to match ML/DL format
                if "fold" in df_tfm25.columns:
                    group_cols = ["model", "dataset", "horizon"]
                    metric_cols = ["RMSE", "MAE", "MAPE", "R2", "time_seconds"]
                    first_cols = ["parameter_count", "type", "status"]
                    agg_dict = {c: "mean" for c in metric_cols if c in df_tfm25.columns}
                    agg_dict.update({c: "first" for c in first_cols if c in df_tfm25.columns})
                    df_tfm25 = df_tfm25.groupby(group_cols, as_index=False).agg(agg_dict)
                    print(f"  TimesFM-2.5 results: {len(df_tfm25)} rows after fold-averaging")

                all_dfs.append(df_tfm25)
        except Exception as e:
            print(f"  TimesFM-2.5 results: error reading: {e}")

    if not all_dfs:
        print("\n  ERROR: No result files found to merge!")
        return None

    # Merge all DataFrames
    merged = pd.concat(all_dfs, ignore_index=True, sort=False)

    # Standardize columns
    standard_cols = [
        "model", "dataset", "horizon", "RMSE", "MAE", "MAPE", "R2",
        "time_seconds", "parameter_count", "type", "status"
    ]
    # Add any missing standard columns
    for col in standard_cols:
        if col not in merged.columns:
            merged[col] = None

    # Keep only successful results for the main output
    merged_success = merged[merged["status"] == "SUCCESS"].copy()

    # Reorder columns (standard first, then any extras)
    extra_cols = [c for c in merged_success.columns if c not in standard_cols]
    final_cols = [c for c in standard_cols if c in merged_success.columns] + extra_cols
    merged_success = merged_success[final_cols]

    # Save
    merged_success.to_csv(OUTPUT_CSV, index=False)

    # Also save complete results (including failures) for debugging
    merged.to_csv(BASE_DIR / "BENCHMARK_RESULTS_ALL.csv", index=False)

    print(f"\n  Merged {len(merged_success)} successful results")
    print(f"  Saved to: {OUTPUT_CSV}")

    # Summary statistics
    if len(merged_success) > 0:
        print(f"\n  Models: {merged_success['model'].nunique()}")
        print(f"  Datasets: {merged_success['dataset'].nunique()}")
        print(f"  Horizons: {merged_success['horizon'].nunique()}")
        print(f"\n  Models found: {sorted(merged_success['model'].unique())}")

    return merged_success


def print_summary(df):
    """Print benchmark summary table."""
    if df is None or len(df) == 0:
        return

    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY - Mean RMSE by Model")
    print(f"{'='*70}")

    # Clip R2 for summary display only (raw data preserved in CSV)
    # R2 < -10 indicates degenerate fits (e.g., near-zero test variance)
    df_summary = df.copy()
    df_summary["R2_clipped"] = df_summary["R2"].clip(lower=-10.0, upper=1.0)

    summary = df_summary.groupby("model").agg(
        mean_RMSE=("RMSE", "mean"),
        mean_R2=("R2_clipped", "mean"),
        mean_time=("time_seconds", "mean"),
        param_count=("parameter_count", "first"),
        n_experiments=("RMSE", "count"),
    ).sort_values("mean_RMSE")

    for _, row in summary.iterrows():
        params = f"{int(row['param_count']):>12,}" if pd.notna(row['param_count']) else "         N/A"
        print(f"  {row.name:15s} | RMSE={row['mean_RMSE']:.4f} | R2={row['mean_R2']:.4f} | "
              f"Time={row['mean_time']:.1f}s | Params={params} | n={int(row['n_experiments'])}")

    # Computational ROI preview
    print(f"\n{'='*70}")
    print("COMPUTATIONAL ROI PREVIEW (RMSE per 1M parameters)")
    print(f"{'='*70}")

    for _, row in summary.iterrows():
        if pd.notna(row['param_count']) and row['param_count'] > 0:
            roi = row['mean_RMSE'] / (row['param_count'] / 1e6)
            print(f"  {row.name:15s} | ROI={roi:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Run upgraded 20-model benchmark")
    parser.add_argument("--skip-ml", action="store_true",
                        help="Skip ML model training (use existing results)")
    parser.add_argument("--skip-dl", action="store_true",
                        help="Skip DL model training (use existing results)")
    parser.add_argument("--skip-foundation", action="store_true",
                        help="Skip foundation model inference")
    parser.add_argument("--merge-only", action="store_true",
                        help="Only merge existing CSVs, skip all training")
    parser.add_argument("--device", default="cpu",
                        help="Device for foundation models (cpu, cuda, mps)")
    args = parser.parse_args()

    print("=" * 70)
    print("UPGRADED BENCHMARK - 20 Models x 10 Sensors x 3 Horizons")
    print("=" * 70)
    print(f"Timestamp: {TIMESTAMP}")
    print(f"Output: {OUTPUT_CSV}")
    print("=" * 70)

    start_time = time.time()

    if args.merge_only:
        args.skip_ml = True
        args.skip_dl = True
        args.skip_foundation = True

    # Stage 1: ML Models
    if not args.skip_ml:
        rc = run_script("run_complete_benchmark.py",
                        description="Stage 1/3: ML Models (14 models)")
        if rc != 0:
            print("WARNING: ML benchmark had errors, continuing...")

    # Stage 2: Deep Learning Models
    if not args.skip_dl:
        rc = run_script("run_deep_learning_all.py",
                        description="Stage 2/3: Deep Learning (LSTM, GRU, BiLSTM)")
        if rc != 0:
            print("WARNING: DL benchmark had errors, continuing...")

    # Stage 3: Foundation Models
    if not args.skip_foundation:
        rc = run_script("run_foundation_models.py",
                        args=["--device", args.device],
                        description="Stage 3/3: Foundation Models (Chronos-2, TimesFM)")
        if rc != 0:
            print("WARNING: Foundation model benchmark had errors, continuing...")

    # Stage 4: Merge all results
    merged_df = merge_results(
        skip_ml=False,  # Always try to find ML results
        skip_dl=False,
        skip_foundation=False,
    )

    # Summary
    print_summary(merged_df)

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"TOTAL BENCHMARK TIME: {total_time/60:.1f} minutes")
    print(f"OUTPUT: {OUTPUT_CSV}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
