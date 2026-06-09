#!/usr/bin/env python3
"""
Deep Learning Runner - All Datasets
====================================
Version: 2.0.0 (2025-12-31)

Runs LSTM, GRU, BiLSTM on all cleaned datasets with proper target scaling.
Supports Apple MPS GPU acceleration for faster training.

Features:
- Walk-forward validation (60/10/30 split)
- Early stopping with patience
- Target scaling for stable training
- Multi-horizon forecasting (H1, H5, H12)

Usage:
    python run_deep_learning_all.py
"""

import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path(__file__).resolve().parent
CLEANED_DIR = BASE_DIR / "cleaned_data"
RESULTS_DIR = BASE_DIR / "ml_results_deep_learning"
RESULTS_DIR.mkdir(exist_ok=True)

# Device selection (configurable via environment variable)
# Default: CPU - because MPS has known issues with LSTM/RNN operations
# See: https://github.com/pytorch/pytorch/issues/94691
# To use MPS/CUDA: export PYTORCH_DEVICE=mps (or cuda)
import os
_device_name = os.environ.get('PYTORCH_DEVICE', 'cpu')
device = torch.device(_device_name)
if _device_name == 'mps':
    print(f"⚠️  WARNING: MPS has known LSTM issues (PyTorch #94691)")
print(f"💻 Using device: {device}")

# Datasets (10 total - all cleaned power sensors)
DATASETS = [
    # Original 4
    "ZWave2_Metered_Wall_Plug_Switch_computers_Sensor_power",
    "ZWave_Node_016_the_whole_multiplier_over_the_window_Sensor_power",
    "ZWave_Node_042_ZW095_Home_Energy_Meter__Gen_5_Electric_meter_watts",
    "ZWave_Node_042_ZW095_Home_Energy_Meter__Gen_5_Electric_meter_watts_2",
    # New 6
    "Wall_Plug_2_Sensor_power",
    "WavePlug_UPS_Sensor_power",
    "ZWave5_network_switch_Sensor_power",
    "ZWave7_bookcase_Sensor_power",
    "ZWave8_on_desk_Sensor_power",
    "ZWave_Node_031_SERVER_SB_Sensor_power",
]

# Horizons
HORIZONS = [1, 5, 12]


# ==============================================================================
# MODEL DEFINITIONS
# ==============================================================================

class LSTMModel(nn.Module):
    """LSTM for time series forecasting"""

    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


class GRUModel(nn.Module):
    """GRU for time series forecasting"""

    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=3, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


class BiLSTMModel(nn.Module):
    """Bidirectional LSTM for time series forecasting"""

    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, bidirectional=True,
                           dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


MODELS = {
    "LSTM": LSTMModel,
    "GRU": GRUModel,
    "BiLSTM": BiLSTMModel,
}


# ==============================================================================
# TRAINING FUNCTIONS
# ==============================================================================

def load_and_prepare_data(dataset_name):
    """Load and prepare dataset for training"""

    data_path = CLEANED_DIR / dataset_name / "clean_full.csv"

    if not data_path.exists():
        print(f"  ❌ Dataset not found: {data_path}")
        return None

    df = pd.read_csv(data_path)

    # Find target columns
    target_cols = [f"value_t+{h}" for h in HORIZONS]
    available_targets = [t for t in target_cols if t in df.columns]

    if not available_targets:
        print(f"  ❌ No target columns found in {dataset_name}")
        return None

    # Feature columns (exclude timestamp and targets)
    exclude = ['timestamp', 'value'] + target_cols
    feature_cols = [c for c in df.columns if c not in exclude]

    # Select only numeric columns to avoid isnan errors
    numeric_feature_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    X = df[numeric_feature_cols].values.astype(np.float32)
    y = df[available_targets].values.astype(np.float32)

    # Remove NaN
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y).any(axis=1))
    X, y = X[mask], y[mask]

    # Walk-forward split (60/10/30)
    n = len(X)
    train_end = int(n * 0.6)
    val_end = int(n * 0.7)

    return {
        'X_train': X[:train_end],
        'y_train': y[:train_end],
        'X_val': X[train_end:val_end],
        'y_val': y[train_end:val_end],
        'X_test': X[val_end:],
        'y_test': y[val_end:],
        'n_features': X.shape[1],
        'n_targets': y.shape[1],
    }


def train_model(model_class, data, epochs=50, batch_size=256, lr=0.001, patience=10):
    """Train a deep learning model"""

    # Scale data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(data['X_train'])
    X_val = scaler_X.transform(data['X_val'])
    X_test = scaler_X.transform(data['X_test'])

    y_train = scaler_y.fit_transform(data['y_train'])
    y_val = scaler_y.transform(data['y_val'])

    # Convert to tensors and reshape for RNN: (batch, seq_len=1, features)
    X_train_t = torch.FloatTensor(X_train).unsqueeze(1).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).unsqueeze(1).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    X_test_t = torch.FloatTensor(X_test).unsqueeze(1).to(device)

    # Initialize model
    model = model_class(
        input_size=data['n_features'],
        hidden_size=64,
        output_size=data['n_targets'],
        dropout=0.2
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        n_batches = 0

        for i in range(0, len(X_train_t), batch_size):
            batch_X = X_train_t[i:i+batch_size]
            batch_y = y_train_t[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Load best model
    if best_state:
        model.load_state_dict(best_state)

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_t).cpu().numpy()

    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test = data['y_test']

    # Calculate metrics per horizon
    results = {}
    for i, h in enumerate(HORIZONS[:data['n_targets']]):
        y_true_h = y_test[:, i]
        y_pred_h = y_pred[:, i]

        mask = np.isfinite(y_true_h) & np.isfinite(y_pred_h)
        y_true_h, y_pred_h = y_true_h[mask], y_pred_h[mask]

        results[f"H{h}"] = {
            "RMSE": float(np.sqrt(mean_squared_error(y_true_h, y_pred_h))),
            "MAE": float(mean_absolute_error(y_true_h, y_pred_h)),
            "R2": float(r2_score(y_true_h, y_pred_h)),
        }

    return results


def main():
    """Run all deep learning models on all datasets"""

    print("=" * 70)
    print("DEEP LEARNING TRAINING - ALL DATASETS")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Models: {list(MODELS.keys())}")
    print(f"Datasets: {len(DATASETS)}")
    print(f"Horizons: {HORIZONS}")
    print("=" * 70)

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = RESULTS_DIR / f"session_{session_id}"
    session_dir.mkdir(exist_ok=True)

    all_results = []
    total_runs = len(DATASETS) * len(MODELS)
    current_run = 0

    start_time = time.time()

    for dataset_name in DATASETS:
        short_name = dataset_name[:30]
        print(f"\n{'='*70}")
        print(f"DATASET: {short_name}...")
        print(f"{'='*70}")

        # Load data
        data = load_and_prepare_data(dataset_name)
        if data is None:
            continue

        print(f"  Samples: Train={len(data['X_train'])}, Val={len(data['X_val'])}, Test={len(data['X_test'])}")
        print(f"  Features: {data['n_features']}")

        for model_name, model_class in MODELS.items():
            current_run += 1
            print(f"\n  [{current_run}/{total_runs}] Training {model_name}...")

            try:
                model_start = time.time()
                results = train_model(model_class, data, epochs=50, patience=10)
                model_time = time.time() - model_start

                # Count parameters
                param_count = sum(
                    p.numel() for p in model_class(input_size=data['n_features']).parameters()
                )

                # Print results
                for horizon, metrics in results.items():
                    print(f"    {horizon}: RMSE={metrics['RMSE']:.4f}, R²={metrics['R2']:.4f}")

                print(f"    Time: {model_time:.1f}s, Params: {param_count:,}")

                # Store results
                result_entry = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "status": "SUCCESS",
                    "metrics": results,
                    "time_seconds": round(model_time, 2),
                    "parameter_count": param_count,
                }
                all_results.append(result_entry)

                # Save individual result
                result_file = session_dir / f"{short_name}_{model_name}.json"
                with open(result_file, 'w') as f:
                    json.dump(result_entry, f, indent=2)

            except Exception as e:
                print(f"    ❌ Error: {e}")
                all_results.append({
                    "dataset": dataset_name,
                    "model": model_name,
                    "status": "FAILED",
                    "error": str(e)
                })

    # Save summary
    total_time = time.time() - start_time
    summary = {
        "session_id": session_id,
        "total_time_minutes": round(total_time / 60, 2),
        "device": str(device),
        "datasets": DATASETS,
        "models": list(MODELS.keys()),
        "horizons": HORIZONS,
        "total_runs": total_runs,
        "successful": sum(1 for r in all_results if r.get('status') == 'SUCCESS'),
        "results": all_results
    }

    summary_file = session_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Flatten results to CSV for orchestrator (run_upgraded_benchmark.py)
    flat_rows = []
    for r in all_results:
        if r.get('status') == 'SUCCESS':
            for h_key, metrics in r['metrics'].items():
                flat_rows.append({
                    "model": r['model'],
                    "dataset": r['dataset'],
                    "horizon": h_key,
                    "RMSE": round(metrics.get('RMSE', 0), 4),
                    "MAE": round(metrics.get('MAE', 0), 4),
                    "MAPE": round(metrics.get('MAPE', 0), 4) if metrics.get('MAPE') is not None else None,
                    "R2": round(metrics.get('R2', 0), 4),
                    "time_seconds": r.get('time_seconds'),
                    "parameter_count": r.get('parameter_count'),
                    "type": "DeepLearning",
                    "status": "SUCCESS",
                })
    if flat_rows:
        csv_path = RESULTS_DIR / "deep_learning_results.csv"
        pd.DataFrame(flat_rows).to_csv(csv_path, index=False)
        print(f"\nCSV results saved to: {csv_path}")

    print("\n" + "=" * 70)
    print("DEEP LEARNING TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Successful: {summary['successful']}/{total_runs}")
    print(f"Results saved to: {session_dir}")
    print("=" * 70)

    # Print comparison table
    print("\n📊 RESULTS SUMMARY:")
    print("-" * 70)
    print(f"{'Model':<10} {'Dataset':<25} {'H1 RMSE':<12} {'H5 RMSE':<12} {'H12 RMSE':<12}")
    print("-" * 70)

    for r in all_results:
        if r.get('status') == 'SUCCESS':
            m = r['metrics']
            print(f"{r['model']:<10} {r['dataset'][:24]:<25} "
                  f"{m.get('H1', {}).get('RMSE', 0):<12.4f} "
                  f"{m.get('H5', {}).get('RMSE', 0):<12.4f} "
                  f"{m.get('H12', {}).get('RMSE', 0):<12.4f}")


if __name__ == "__main__":
    main()
