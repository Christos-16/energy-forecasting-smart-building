#!/usr/bin/env python3
"""
Out-of-Sample Validation Script
================================
Validates Ridge model on NEW data (after 25/10/2025) that was never seen during training.

This script:
1. Loads training data as context buffer (for lag calculations)
2. Loads and preprocesses new raw data
3. Applies IDENTICAL feature engineering as training
4. Trains Ridge on ALL training data
5. Makes predictions on new data using scaler.transform() (NOT fit_transform!)
6. Compares predictions to actual values
7. Reports RMSE, MAE for comparison with training metrics

Usage:
    python oos_validation.py
"""

import os
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
NEW_DATA_PATH = BASE_DIR / "measurements_since_2025-10-24.csv"
CLEANED_DATA_DIR = BASE_DIR / "cleaned_data"

# Training data cutoff (training ended on 25/10/2025)
TRAIN_CUTOFF = pd.Timestamp("2025-10-25", tz='UTC')

# Horizons to test (same as paper)
HORIZONS = [1, 5, 12]  # H1=5min, H5=25min, H12=60min

# Mapping: cleaned folder name -> raw sensor name in new data
SENSOR_MAPPING = {
    "ZWave_Node_042_ZW095_Home_Energy_Meter__Gen_5_Electric_meter_watts":
        "ZWave_Node_042_ZW095_Home_Energy_Meter__Gen_5_Electric_meter_watts",
    "ZWave_Node_042_ZW095_Home_Energy_Meter__Gen_5_Electric_meter_watts_2":
        "ZWave_Node_042_ZW095_Home_Energy_Meter__Gen_5_Electric_meter_watts_2",
    "ZWave2_Metered_Wall_Plug_Switch_computers_Sensor_power":
        "ZWave2_Metered_Wall_Plug_Switch_computers_Sensor_power",
    "WavePlug_UPS_Sensor_power":
        "WavePlug_UPS_Sensor_power",
    "ZWave_Node_016_the_whole_multiplier_over_the_window_Sensor_power":
        "ZWave_Node_016_the_whole_multiplier_over_the_window_Sensor_power",
    "ZWave7_bookcase_Sensor_power":
        "ZWave_7_Metered_Wall_Plug_Switch_bookcase_Sensor_power",
    "ZWave5_network_switch_Sensor_power":
        "ZWave5_Metered_Wall_Plug_Switch_network_switch_Sensor_power",
    "ZWave8_on_desk_Sensor_power":
        "ZWave_8_Metered_Wall_Plug_Switch_on_desk_Sensor_power",
    "Wall_Plug_2_Sensor_power":
        "Wall_Plug_2_Sensor_power",
    "ZWave_Node_031_SERVER_SB_Sensor_power":
        "ZWave_Node_031_SERVER_SB_Sensor_power",
}

# Feature engineering parameters (must match training!)
LAGS = [1, 5, 10, 30, 60, 120, 1440, 10080, 20160]
ROLLING_WINDOWS = [5, 10, 30, 60, 120, 1440]
FOURIER_DAILY_K = 5
FOURIER_WEEKLY_K = 1


# ============================================================================
# FEATURE ENGINEERING (must match training exactly!)
# ============================================================================

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features: hour, dow, is_weekend, month."""
    df = df.copy()
    df['hour'] = df.index.hour
    df['dow'] = df.index.dayofweek
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    df['month'] = df.index.month
    return df


def add_lag_features(df: pd.DataFrame, col: str = 'value') -> pd.DataFrame:
    """Add lag features."""
    df = df.copy()
    for lag in LAGS:
        df[f'{col}_lag{lag}'] = df[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, col: str = 'value') -> pd.DataFrame:
    """Add rolling statistics with CAUSAL masking (shift before rolling)."""
    df = df.copy()
    for window in ROLLING_WINDOWS:
        # CRITICAL: shift(1) ensures we only use PAST values (no data leakage!)
        shifted = df[col].shift(1)
        rolling = shifted.rolling(window, min_periods=max(1, window // 2))

        df[f'{col}_roll{window}_mean'] = rolling.mean()
        df[f'{col}_roll{window}_std'] = rolling.std()
        df[f'{col}_roll{window}_min'] = rolling.min()
        df[f'{col}_roll{window}_max'] = rolling.max()
    return df


def add_fourier_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Fourier features for daily and weekly seasonality."""
    df = df.copy()

    # Daily Fourier (period = 288 for 5-min data = 24h)
    minutes_in_day = df.index.hour * 60 + df.index.minute
    for k in range(1, FOURIER_DAILY_K + 1):
        df[f'fourier_daily_sin{k}'] = np.sin(2 * np.pi * k * minutes_in_day / (24 * 60))
        df[f'fourier_daily_cos{k}'] = np.cos(2 * np.pi * k * minutes_in_day / (24 * 60))

    # Weekly Fourier (period = 7 days)
    day_of_week = df.index.dayofweek + (minutes_in_day / (24 * 60))
    for k in range(1, FOURIER_WEEKLY_K + 1):
        df[f'fourier_weekly_sin{k}'] = np.sin(2 * np.pi * k * day_of_week / 7)
        df[f'fourier_weekly_cos{k}'] = np.cos(2 * np.pi * k * day_of_week / 7)

    return df


def add_targets(df: pd.DataFrame, col: str = 'value') -> pd.DataFrame:
    """Add target columns (future values)."""
    df = df.copy()
    for h in HORIZONS:
        df[f'{col}_t+{h}'] = df[col].shift(-h)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply full feature engineering pipeline."""
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_fourier_features(df)
    df = add_targets(df)
    return df


# ============================================================================
# DATA LOADING
# ============================================================================

def load_training_context(sensor_folder: str) -> pd.DataFrame:
    """Load cleaned training data as context buffer."""
    csv_path = CLEANED_DATA_DIR / sensor_folder / "clean_full.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Training data not found: {csv_path}")

    # Read first to detect timestamp column name
    df = pd.read_csv(csv_path, nrows=1)
    ts_col = 'ts_utc' if 'ts_utc' in df.columns else 'timestamp'

    # Read full file with correct column
    df = pd.read_csv(csv_path, parse_dates=[ts_col], index_col=ts_col)

    # Ensure UTC timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')

    print(f"  Loaded training data: {len(df)} rows, {df.index.min()} to {df.index.max()}")
    return df


def load_new_data(sensor_name: str) -> pd.DataFrame:
    """Load and preprocess new raw data for a specific sensor."""
    print(f"  Loading new data for: {sensor_name}")

    # Read in chunks to handle large file
    chunks = []
    for chunk in pd.read_csv(NEW_DATA_PATH, chunksize=500000):
        sensor_data = chunk[chunk['item'] == sensor_name]
        if len(sensor_data) > 0:
            chunks.append(sensor_data)

    if not chunks:
        raise ValueError(f"No data found for sensor: {sensor_name}")

    raw_df = pd.concat(chunks, ignore_index=True)
    print(f"  Found {len(raw_df)} raw measurements")

    # Parse timestamps (handle mixed formats)
    raw_df['ts'] = pd.to_datetime(raw_df['ts'], format='mixed', utc=True)

    # Filter negative values (impossible for power)
    raw_df = raw_df[raw_df['value'] >= 0]

    # Resample to 5-minute intervals
    raw_df = raw_df.set_index('ts').sort_index()
    resampled = raw_df['value'].resample('5min').mean()

    # Create DataFrame
    df = pd.DataFrame({'value': resampled})
    df = df.dropna()

    print(f"  Resampled to {len(df)} 5-min intervals")
    print(f"  Time range: {df.index.min()} to {df.index.max()}")

    return df


# ============================================================================
# VALIDATION
# ============================================================================

def run_validation_for_sensor(sensor_folder: str, sensor_raw_name: str) -> dict:
    """Run out-of-sample validation for a single sensor."""
    print(f"\n{'='*60}")
    print(f"SENSOR: {sensor_folder}")
    print(f"{'='*60}")

    results = {
        'sensor': sensor_folder,
        'status': 'success',
        'metrics': {}
    }

    try:
        # 1. Load training context
        train_df = load_training_context(sensor_folder)

        # Keep only 'value' column from training (we'll recalculate features)
        train_values = train_df[['value']].copy()

        # 2. Load new data
        new_df = load_new_data(sensor_raw_name)

        # 3. Use ALL training data (not just context buffer)
        # For new data, we need context to calculate lags, so prepend last 3 weeks of training
        context_start = TRAIN_CUTOFF - pd.Timedelta(days=21)
        train_context_for_lags = train_values[train_values.index >= context_start]

        # Combine context + new data for feature engineering
        combined_for_new = pd.concat([train_context_for_lags, new_df])
        combined_for_new = combined_for_new[~combined_for_new.index.duplicated(keep='last')]
        combined_for_new = combined_for_new.sort_index()

        print(f"  Combined (context+new): {len(combined_for_new)} rows")

        # 4. Engineer features separately for train and test
        print("  Engineering features...")

        # For TRAINING: use ALL original training data
        train_features = engineer_features(train_values.copy())
        train_data = train_features[train_features.index <= TRAIN_CUTOFF].dropna()

        # For TESTING: use combined (context + new), then filter to only new data
        test_features = engineer_features(combined_for_new)
        test_data = test_features[test_features.index > TRAIN_CUTOFF].dropna()

        print(f"  Train samples: {len(train_data)}")
        print(f"  Test samples: {len(test_data)}")

        if len(test_data) < 100:
            print(f"  WARNING: Not enough test data!")
            results['status'] = 'insufficient_data'
            return results

        # 6. Get feature columns (exclude targets and value)
        feature_cols = [c for c in train_data.columns
                       if c not in ['value', 'value_t+1', 'value_t+5', 'value_t+12']]

        print(f"  Feature columns: {len(feature_cols)}")

        # 7. For each horizon, train and predict
        for h in HORIZONS:
            target_col = f'value_t+{h}'

            # Prepare train data
            X_train = train_data[feature_cols]
            y_train = train_data[target_col]

            # Prepare test data
            X_test = test_data[feature_cols]
            y_test = test_data[target_col]

            # Fit scaler on training data ONLY
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            # Transform test data (NOT fit_transform!)
            X_test_scaled = scaler.transform(X_test)

            # Train Ridge
            model = Ridge(alpha=1.0)
            model.fit(X_train_scaled, y_train)

            # Predict
            y_pred = model.predict(X_test_scaled)

            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results['metrics'][f'H{h}'] = {
                'RMSE': round(rmse, 4),
                'MAE': round(mae, 4),
                'R2': round(r2, 4),
                'n_samples': len(y_test)
            }

            print(f"  H{h}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

    except Exception as e:
        print(f"  ERROR: {str(e)}")
        results['status'] = 'error'
        results['error'] = str(e)

    return results


def main():
    """Run validation for all sensors."""
    print("=" * 70)
    print("OUT-OF-SAMPLE VALIDATION")
    print(f"Training cutoff: {TRAIN_CUTOFF}")
    print(f"New data file: {NEW_DATA_PATH}")
    print("=" * 70)

    # Check if new data file exists
    if not NEW_DATA_PATH.exists():
        print(f"ERROR: New data file not found: {NEW_DATA_PATH}")
        sys.exit(1)

    all_results = []

    # Run validation for each sensor
    for sensor_folder, sensor_raw in SENSOR_MAPPING.items():
        results = run_validation_for_sensor(sensor_folder, sensor_raw)
        all_results.append(results)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Sensor':<50} | H1 RMSE | H5 RMSE | H12 RMSE")
    print("-" * 80)

    for r in all_results:
        if r['status'] == 'success':
            h1 = r['metrics'].get('H1', {}).get('RMSE', 'N/A')
            h5 = r['metrics'].get('H5', {}).get('RMSE', 'N/A')
            h12 = r['metrics'].get('H12', {}).get('RMSE', 'N/A')
            print(f"{r['sensor'][:50]:<50} | {h1:>7} | {h5:>7} | {h12:>8}")
        else:
            print(f"{r['sensor'][:50]:<50} | {r['status']}")

    # Save results to CSV
    results_df = []
    for r in all_results:
        if r['status'] == 'success':
            for horizon, metrics in r['metrics'].items():
                results_df.append({
                    'sensor': r['sensor'],
                    'horizon': horizon,
                    **metrics
                })

    if results_df:
        df = pd.DataFrame(results_df)
        output_path = BASE_DIR / "oos_validation_results.csv"
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
