#!/usr/bin/env python3
"""
Generate Out-of-Sample Validation Figure for Paper
Shows actual vs predicted values for UPS sensor over one week in January 2026.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from pathlib import Path
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# Configuration
BASE_DIR = Path(__file__).resolve().parent
CLEANED_DATA_DIR = BASE_DIR / "cleaned_data"
NEW_DATA_PATH = BASE_DIR / "measurements_since_2025-10-24.csv"
OUTPUT_DIR = BASE_DIR / "paper_figures"
OUTPUT_DIR.mkdir(exist_ok=True)

TRAIN_CUTOFF = pd.Timestamp("2025-10-25", tz='UTC')

# Feature engineering parameters
LAGS = [1, 5, 10, 30, 60, 120, 1440, 10080, 20160]
ROLLING_WINDOWS = [5, 10, 30, 60, 120, 1440]
FOURIER_DAILY_K = 5
FOURIER_WEEKLY_K = 1


def add_calendar_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dow'] = df.index.dayofweek
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    df['month'] = df.index.month
    return df


def add_lag_features(df, col='value'):
    df = df.copy()
    for lag in LAGS:
        df[f'{col}_lag{lag}'] = df[col].shift(lag)
    return df


def add_rolling_features(df, col='value'):
    df = df.copy()
    for window in ROLLING_WINDOWS:
        shifted = df[col].shift(1)
        rolling = shifted.rolling(window, min_periods=max(1, window // 2))
        df[f'{col}_roll{window}_mean'] = rolling.mean()
        df[f'{col}_roll{window}_std'] = rolling.std()
        df[f'{col}_roll{window}_min'] = rolling.min()
        df[f'{col}_roll{window}_max'] = rolling.max()
    return df


def add_fourier_features(df):
    df = df.copy()
    minutes_in_day = df.index.hour * 60 + df.index.minute
    for k in range(1, FOURIER_DAILY_K + 1):
        df[f'fourier_daily_sin{k}'] = np.sin(2 * np.pi * k * minutes_in_day / (24 * 60))
        df[f'fourier_daily_cos{k}'] = np.cos(2 * np.pi * k * minutes_in_day / (24 * 60))
    day_of_week = df.index.dayofweek + (minutes_in_day / (24 * 60))
    for k in range(1, FOURIER_WEEKLY_K + 1):
        df[f'fourier_weekly_sin{k}'] = np.sin(2 * np.pi * k * day_of_week / 7)
        df[f'fourier_weekly_cos{k}'] = np.cos(2 * np.pi * k * day_of_week / 7)
    return df


def add_targets(df, col='value'):
    df = df.copy()
    for h in [1, 5, 12]:
        df[f'{col}_t+{h}'] = df[col].shift(-h)
    return df


def engineer_features(df):
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_fourier_features(df)
    df = add_targets(df)
    return df


def main():
    print("Generating Out-of-Sample Validation Figure...")

    # Load UPS training data
    csv_path = CLEANED_DATA_DIR / "WavePlug_UPS_Sensor_power" / "clean_full.csv"
    df = pd.read_csv(csv_path, nrows=1)
    ts_col = 'ts_utc' if 'ts_utc' in df.columns else 'timestamp'
    train_df = pd.read_csv(csv_path, parse_dates=[ts_col], index_col=ts_col)

    if train_df.index.tz is None:
        train_df.index = train_df.index.tz_localize('UTC')
    else:
        train_df.index = train_df.index.tz_convert('UTC')

    train_values = train_df[['value']].copy()
    print(f"Training data: {len(train_values)} rows")

    # Load new data for UPS
    print("Loading new data...")
    chunks = []
    for chunk in pd.read_csv(NEW_DATA_PATH, chunksize=500000):
        sensor_data = chunk[chunk['item'] == 'WavePlug_UPS_Sensor_power']
        if len(sensor_data) > 0:
            chunks.append(sensor_data)

    if not chunks:
        raise ValueError("No data found for WavePlug_UPS_Sensor_power in new data file")

    raw_df = pd.concat(chunks, ignore_index=True)
    raw_df['ts'] = pd.to_datetime(raw_df['ts'], format='mixed', utc=True)
    raw_df = raw_df[raw_df['value'] >= 0]
    raw_df = raw_df.set_index('ts').sort_index()
    new_df = pd.DataFrame({'value': raw_df['value'].resample('5min').mean()}).dropna()
    print(f"New data: {len(new_df)} rows")

    # Combine for feature engineering
    context_start = TRAIN_CUTOFF - pd.Timedelta(days=21)
    train_context = train_values[train_values.index >= context_start]
    combined = pd.concat([train_context, new_df])
    combined = combined[~combined.index.duplicated(keep='last')].sort_index()

    # Feature engineering
    print("Engineering features...")
    train_features = engineer_features(train_values.copy())
    train_data = train_features[train_features.index <= TRAIN_CUTOFF].dropna()

    test_features = engineer_features(combined)
    test_data = test_features[test_features.index > TRAIN_CUTOFF].dropna()

    # Get feature columns
    feature_cols = [c for c in train_data.columns
                   if c not in ['value', 'value_t+1', 'value_t+5', 'value_t+12']]

    # Train model (H1 = 5 minutes ahead)
    print("Training Ridge model...")
    X_train = train_data[feature_cols]
    y_train = train_data['value_t+1']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)

    # Predict on new data
    X_test = test_data[feature_cols]
    y_test = test_data['value_t+1']

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    # Create prediction DataFrame
    pred_df = pd.DataFrame({
        'actual': y_test.values,
        'predicted': y_pred
    }, index=test_data.index)

    # Select one week in January 2026
    week_start = pd.Timestamp("2026-01-06", tz='UTC')
    week_end = pd.Timestamp("2026-01-13", tz='UTC')
    week_data = pred_df[(pred_df.index >= week_start) & (pred_df.index < week_end)]

    print(f"Plotting week: {week_start.date()} to {week_end.date()}")
    print(f"Data points: {len(week_data)}")

    if len(week_data) < 10:
        raise ValueError(f"Insufficient data points ({len(week_data)}) for the selected week")

    # Create figure
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot actual and predicted
    ax.plot(week_data.index, week_data['actual'],
            color='#2E86AB', linewidth=1.2, label='Actual', alpha=0.8)
    ax.plot(week_data.index, week_data['predicted'],
            color='#E94F37', linewidth=1.2, label='Predicted (Ridge)', alpha=0.8)

    # Formatting
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Power (W)', fontsize=12)
    ax.set_title('Out-of-Sample Validation: UPS Power Consumption (January 2026)', fontsize=14)
    ax.legend(loc='upper right', fontsize=11)

    # Format x-axis
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %d'))
    ax.tick_params(axis='x', rotation=45)

    # Add R² annotation
    r2 = r2_score(week_data['actual'], week_data['predicted'])
    ax.text(0.02, 0.98, f'$R^2$ = {r2:.3f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save figure
    output_path = OUTPUT_DIR / "fig5_oos_validation.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to: {output_path}")

    # Also save as PNG for preview
    plt.savefig(OUTPUT_DIR / "fig5_oos_validation.png", dpi=150, bbox_inches='tight')

    plt.close()
    print("Done!")


if __name__ == "__main__":
    main()
