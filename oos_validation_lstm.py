#!/usr/bin/env python3
"""
Out-of-Sample Validation for LSTM
==================================
Validates LSTM model on NEW data (after 25/10/2025) that was never seen during training.

This script:
1. Loads training data and trains LSTM on ALL training data
2. Loads new raw data and preprocesses it
3. Makes predictions on new data
4. Reports metrics for comparison with Ridge OOS results

Usage:
    python oos_validation_lstm.py
"""

import warnings
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore', category=FutureWarning)

# Device selection
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Configuration
BASE_DIR = Path(__file__).resolve().parent
NEW_DATA_PATH = BASE_DIR / "measurements_since_2025-10-24.csv"
CLEANED_DATA_DIR = BASE_DIR / "cleaned_data"
TRAIN_CUTOFF = pd.Timestamp("2025-10-25", tz='UTC')

# Feature engineering parameters (must match training!)
LAGS = [1, 5, 10, 30, 60, 120, 1440, 10080, 20160]
ROLLING_WINDOWS = [5, 10, 30, 60, 120, 1440]
FOURIER_DAILY_K = 5
FOURIER_WEEKLY_K = 1
HORIZONS = [1, 5, 12]

# Sensors to test (same as Ridge OOS)
SENSOR_MAPPING = {
    "WavePlug_UPS_Sensor_power": "WavePlug_UPS_Sensor_power",
    "ZWave2_Metered_Wall_Plug_Switch_computers_Sensor_power":
        "ZWave2_Metered_Wall_Plug_Switch_computers_Sensor_power",
}


class LSTMForecaster(nn.Module):
    """LSTM model for multi-horizon energy forecasting"""

    def __init__(self, input_size, hidden_size=64, output_size=3, dropout=0.2):
        super(LSTMForecaster, self).__init__()

        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                             num_layers=1, batch_first=True, dropout=0)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size // 2,
                             num_layers=1, batch_first=True, dropout=0)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


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
    for h in HORIZONS:
        df[f'{col}_t+{h}'] = df[col].shift(-h)
    return df


def engineer_features(df):
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_fourier_features(df)
    df = add_targets(df)
    return df


def load_training_data(sensor_folder):
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

    return df


def load_new_data(sensor_name):
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


def train_lstm(X_train, y_train, epochs=50, batch_size=256, lr=0.001, patience=10):
    """Train LSTM model"""
    X_train_t = torch.FloatTensor(X_train).unsqueeze(1).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)

    model = LSTMForecaster(
        input_size=X_train.shape[1],
        hidden_size=64,
        output_size=y_train.shape[1],
        dropout=0.2
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0

        for i in range(0, len(X_train_t), batch_size):
            batch_X = X_train_t[i:i+batch_size]
            batch_y = y_train_t[i:i+batch_size]

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: loss={avg_loss:.4f}")

    model.load_state_dict(best_state)
    return model


def run_oos_validation(sensor_folder, sensor_raw_name):
    """Run OOS validation for a single sensor"""
    print(f"\n{'='*60}")
    print(f"LSTM OOS: {sensor_folder}")
    print(f"{'='*60}")

    results = {'sensor': sensor_folder, 'status': 'success', 'metrics': {}}

    try:
        # Load data
        train_df = load_training_data(sensor_folder)
        train_values = train_df[['value']].copy()
        print(f"  Training data: {len(train_values)} rows")

        new_df = load_new_data(sensor_raw_name)
        print(f"  New data: {len(new_df)} rows")

        # Combine for feature engineering
        context_start = TRAIN_CUTOFF - pd.Timedelta(days=21)
        train_context = train_values[train_values.index >= context_start]
        combined = pd.concat([train_context, new_df])
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()

        # Feature engineering
        print("  Engineering features...")
        train_features = engineer_features(train_values.copy())
        train_data = train_features[train_features.index <= TRAIN_CUTOFF].dropna()

        test_features = engineer_features(combined)
        test_data = test_features[test_features.index > TRAIN_CUTOFF].dropna()

        print(f"  Train samples: {len(train_data)}")
        print(f"  Test samples: {len(test_data)}")

        if len(test_data) < 100:
            results['status'] = 'insufficient_data'
            return results

        # Prepare features
        target_cols = [f'value_t+{h}' for h in HORIZONS]
        feature_cols = [c for c in train_data.columns
                       if c not in ['value'] + target_cols]

        X_train = train_data[feature_cols].values
        y_train = train_data[target_cols].values

        X_test = test_data[feature_cols].values
        y_test = test_data[target_cols].values

        # Scale features
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train)
        X_test_scaled = scaler_X.transform(X_test)

        # Train LSTM
        print("  Training LSTM...")
        model = train_lstm(X_train_scaled, y_train_scaled)

        # Predict
        model.eval()
        X_test_t = torch.FloatTensor(X_test_scaled).unsqueeze(1).to(device)
        with torch.no_grad():
            y_pred_scaled = model(X_test_t).cpu().numpy()

        y_pred = scaler_y.inverse_transform(y_pred_scaled)

        # Calculate metrics per horizon
        for i, h in enumerate(HORIZONS):
            y_true_h = y_test[:, i]
            y_pred_h = y_pred[:, i]

            mask = np.isfinite(y_true_h) & np.isfinite(y_pred_h)
            y_true_h = y_true_h[mask]
            y_pred_h = y_pred_h[mask]

            rmse = np.sqrt(mean_squared_error(y_true_h, y_pred_h))
            mae = mean_absolute_error(y_true_h, y_pred_h)
            r2 = r2_score(y_true_h, y_pred_h)

            results['metrics'][f'H{h}'] = {
                'RMSE': round(rmse, 4),
                'MAE': round(mae, 4),
                'R2': round(r2, 4),
                'n_samples': len(y_true_h)
            }

            print(f"  H{h}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

    except Exception as e:
        print(f"  ERROR: {str(e)}")
        results['status'] = 'error'
        results['error'] = str(e)

    return results


def main():
    print("=" * 70)
    print("OUT-OF-SAMPLE VALIDATION - LSTM")
    print(f"Training cutoff: {TRAIN_CUTOFF}")
    print("=" * 70)

    if not NEW_DATA_PATH.exists():
        print(f"ERROR: New data file not found: {NEW_DATA_PATH}")
        return

    all_results = []
    for sensor_folder, sensor_raw in SENSOR_MAPPING.items():
        results = run_oos_validation(sensor_folder, sensor_raw)
        all_results.append(results)

    # Print summary
    print("\n" + "=" * 70)
    print("LSTM OOS SUMMARY")
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
                    'model': 'LSTM',
                    'horizon': horizon,
                    **metrics
                })

    if results_df:
        df = pd.DataFrame(results_df)
        output_path = BASE_DIR / "oos_validation_lstm_results.csv"
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
