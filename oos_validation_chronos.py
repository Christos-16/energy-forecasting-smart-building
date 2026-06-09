#!/usr/bin/env python3
"""
Out-of-Sample Validation for Chronos (Zero-Shot Foundation Model)
==================================================================
Validates Amazon Chronos on NEW data (after 25/10/2025).

Chronos is a zero-shot model - no training required, just inference.
Uses training data as context for predictions.

Usage:
    python oos_validation_chronos.py
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore', category=FutureWarning)

# Configuration
BASE_DIR = Path(__file__).resolve().parent
NEW_DATA_PATH = BASE_DIR / "measurements_since_2025-10-24.csv"
CLEANED_DATA_DIR = BASE_DIR / "cleaned_data"
TRAIN_CUTOFF = pd.Timestamp("2025-10-25", tz='UTC')

# Chronos settings
MODEL_SIZE = 'small'  # Use small for faster inference
HORIZONS = [1, 5, 12]  # H1=5min, H5=25min, H12=60min
CONTEXT_LENGTH = 512  # How many past steps to use as context
NUM_SAMPLES = 20  # For probabilistic forecast

# Sensors to test
SENSOR_MAPPING = {
    "WavePlug_UPS_Sensor_power": "WavePlug_UPS_Sensor_power",
    "ZWave2_Metered_Wall_Plug_Switch_computers_Sensor_power":
        "ZWave2_Metered_Wall_Plug_Switch_computers_Sensor_power",
}


def load_training_data(sensor_folder):
    """Load training data for context"""
    csv_path = CLEANED_DATA_DIR / sensor_folder / "clean_full.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Training data not found: {csv_path}")

    df = pd.read_csv(csv_path, nrows=1)
    ts_col = 'ts_utc' if 'ts_utc' in df.columns else 'timestamp'
    df = pd.read_csv(csv_path, parse_dates=[ts_col], index_col=ts_col)

    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')

    return df[['value']]


def load_new_data(sensor_name):
    """Load new data for validation"""
    print(f"  Loading new data for: {sensor_name}")
    chunks = []
    for chunk in pd.read_csv(NEW_DATA_PATH, chunksize=500000):
        sensor_data = chunk[chunk['item'] == sensor_name]
        if len(sensor_data) > 0:
            chunks.append(sensor_data)

    if not chunks:
        raise ValueError(f"No data found for sensor: {sensor_name}")

    raw_df = pd.concat(chunks, ignore_index=True)
    raw_df['ts'] = pd.to_datetime(raw_df['ts'], format='mixed', utc=True)
    raw_df = raw_df[raw_df['value'] >= 0]
    raw_df = raw_df.set_index('ts').sort_index()
    new_df = pd.DataFrame({'value': raw_df['value'].resample('5min').mean()}).dropna()

    return new_df


def rolling_forecast_chronos(pipeline, context_values, test_values, horizon, max_predictions=500):
    """
    Rolling forecast using Chronos.

    Args:
        pipeline: Chronos pipeline
        context_values: Historical values for context
        test_values: Test values to predict
        horizon: Steps ahead to predict
        max_predictions: Limit predictions for speed
    """
    n_test = min(len(test_values) - horizon, max_predictions)
    predictions = []
    actuals = []

    print(f"    Making {n_test} predictions for H{horizon}...")
    progress_step = max(1, n_test // 10)

    for i in range(n_test):
        # Use last CONTEXT_LENGTH values as context
        if i > 0:
            current_values = np.concatenate([context_values, test_values[:i]])
        else:
            current_values = context_values

        # Take last CONTEXT_LENGTH points
        context = current_values[-CONTEXT_LENGTH:]
        context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)

        # Forecast
        try:
            forecast = pipeline.predict(
                inputs=context_tensor,
                prediction_length=horizon,
                num_samples=NUM_SAMPLES
            )
            forecast_np = forecast.squeeze().cpu().numpy()

            # Use median as point prediction
            if forecast_np.ndim == 2:
                y_pred = np.median(forecast_np, axis=0)[-1]  # Last step = horizon steps ahead
            else:
                y_pred = forecast_np[-1] if len(forecast_np) >= horizon else forecast_np[0]

            predictions.append(y_pred)
            actuals.append(test_values[i + horizon])

        except Exception as e:
            print(f"    Warning: Prediction failed at step {i}: {e}")
            continue

        if (i + 1) % progress_step == 0:
            print(f"    Progress: {int((i+1)/n_test*100)}%")

    return np.array(predictions), np.array(actuals)


def run_oos_validation(sensor_folder, sensor_raw_name, pipeline):
    """Run OOS validation for a single sensor"""
    print(f"\n{'='*60}")
    print(f"CHRONOS OOS: {sensor_folder}")
    print(f"{'='*60}")

    results = {'sensor': sensor_folder, 'status': 'success', 'metrics': {}}

    try:
        # Load data
        train_df = load_training_data(sensor_folder)
        train_df = train_df[train_df.index <= TRAIN_CUTOFF]
        print(f"  Training context: {len(train_df)} rows")

        new_df = load_new_data(sensor_raw_name)
        print(f"  New data: {len(new_df)} rows")

        if len(new_df) < 100:
            results['status'] = 'insufficient_data'
            return results

        # Get values as numpy arrays
        context_values = train_df['value'].values
        test_values = new_df['value'].values

        # Forecast each horizon
        for h in HORIZONS:
            print(f"\n  Horizon H{h}...")
            y_pred, y_true = rolling_forecast_chronos(
                pipeline, context_values, test_values, h
            )

            if len(y_pred) < 10:
                print(f"    Not enough predictions for H{h}")
                continue

            # Remove NaN/Inf
            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            y_true = y_true[mask]
            y_pred = y_pred[mask]

            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            results['metrics'][f'H{h}'] = {
                'RMSE': round(rmse, 4),
                'MAE': round(mae, 4),
                'R2': round(r2, 4),
                'n_samples': len(y_true)
            }

            print(f"  H{h}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

    except Exception as e:
        print(f"  ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        results['status'] = 'error'
        results['error'] = str(e)

    return results


def main():
    print("=" * 70)
    print("OUT-OF-SAMPLE VALIDATION - CHRONOS (Zero-Shot)")
    print(f"Model: chronos-t5-{MODEL_SIZE}")
    print(f"Training cutoff: {TRAIN_CUTOFF}")
    print("=" * 70)

    if not NEW_DATA_PATH.exists():
        print(f"ERROR: New data file not found: {NEW_DATA_PATH}")
        return

    # Load Chronos model
    print("\nLoading Chronos model...")
    try:
        from chronos import ChronosPipeline

        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"  Device: {device}")

        pipeline = ChronosPipeline.from_pretrained(
            f"amazon/chronos-t5-{MODEL_SIZE}",
            device_map=device,
            torch_dtype=torch.bfloat16
        )
        print("  Model loaded!")
    except ImportError:
        print("ERROR: chronos package not installed")
        print("Install with: pip install chronos-forecasting")
        return
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return

    all_results = []
    for sensor_folder, sensor_raw in SENSOR_MAPPING.items():
        results = run_oos_validation(sensor_folder, sensor_raw, pipeline)
        all_results.append(results)

    # Print summary
    print("\n" + "=" * 70)
    print("CHRONOS OOS SUMMARY")
    print("=" * 70)
    print(f"\n{'Sensor':<50} | H1 R2   | H5 R2   | H12 R2")
    print("-" * 80)

    for r in all_results:
        if r['status'] == 'success':
            h1 = r['metrics'].get('H1', {}).get('R2', 'N/A')
            h5 = r['metrics'].get('H5', {}).get('R2', 'N/A')
            h12 = r['metrics'].get('H12', {}).get('R2', 'N/A')
            print(f"{r['sensor'][:50]:<50} | {h1:>7} | {h5:>7} | {h12:>7}")
        else:
            print(f"{r['sensor'][:50]:<50} | {r['status']}")

    # Save results
    results_df = []
    for r in all_results:
        if r['status'] == 'success':
            for horizon, metrics in r['metrics'].items():
                results_df.append({
                    'sensor': r['sensor'],
                    'model': 'Chronos',
                    'horizon': horizon,
                    **metrics
                })

    if results_df:
        df = pd.DataFrame(results_df)
        output_path = BASE_DIR / "oos_validation_chronos_results.csv"
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
