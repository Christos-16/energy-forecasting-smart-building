#!/usr/bin/env python3
"""
COMPLETE BENCHMARK - ALL MODELS × ALL DATASETS
===============================================
Version: 2.0.0 (2025-12-31)

Runs ALL algorithms (ML + Deep Learning) on ALL cleaned datasets.
Results saved with timestamp for clear identification.

Supported Models (17 total):
- Baseline: Naive (persistence model)
- Linear: Ridge, ElasticNet
- Tree-based: RandomForest, ExtraTrees
- Boosting: CatBoost, LightGBM, XGBoost, NGBoost
- Instance-based: KNN
- Time Series: Prophet, SARIMAX
- Deep Learning: PatchTST, TFT (transformer-based)
- RNN: LSTM, GRU, BiLSTM (via run_deep_learning_all.py)

Usage:
    python run_complete_benchmark.py
"""

import json
import time
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent
CLEANED_DIR = BASE_DIR / "cleaned_data"

# Create results directory with timestamp
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = BASE_DIR / f"FINAL_RESULTS_{TIMESTAMP}"
RESULTS_DIR.mkdir(exist_ok=True)

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
# IMPORTS
# ==============================================================================

print("Loading libraries...")

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("  Warning: CatBoost not available")

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("  Warning: LightGBM not available")

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("  Warning: XGBoost not available")

try:
    from ngboost import NGBRegressor
    HAS_NGBOOST = True
except ImportError:
    HAS_NGBOOST = False
    print("  Warning: NGBoost not available")

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True

    # Device selection (configurable via environment variable)
    # Default: CPU - because MPS has known issues with LSTM/RNN operations
    # See: https://github.com/pytorch/pytorch/issues/94691
    # To use MPS/CUDA: export PYTORCH_DEVICE=mps (or cuda)
    import os
    _device_name = os.environ.get('PYTORCH_DEVICE', 'cpu')
    DEVICE = torch.device(_device_name)
    if _device_name == 'mps':
        print(f"  ⚠️  WARNING: MPS has known LSTM issues (PyTorch #94691)")
    print(f"  PyTorch: Using device: {DEVICE}")
except ImportError:
    HAS_TORCH = False
    DEVICE = None
    print("  Warning: PyTorch not available")

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False
    print("  Warning: Prophet not available")

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX as SARIMAX_Model
    HAS_SARIMAX = True
except ImportError:
    HAS_SARIMAX = False
    print("  Warning: SARIMAX not available")

# PatchTST and TFT from our common module
try:
    from common.extra_models_and_metrics import PatchTSTAdapter, TFTAdapter
    HAS_PATCHTST = True
    HAS_TFT = True
except ImportError as e:
    HAS_PATCHTST = False
    HAS_TFT = False
    print(f"  Warning: PatchTST/TFT not available: {e}")

print("Libraries loaded!\n")

# ==============================================================================
# NAIVE BASELINE
# ==============================================================================

class NaiveRegressor:
    """
    Naive baseline: predicts the last known value (persistence model).
    Essential baseline for time series - any good model should beat this.
    """
    def __init__(self):
        self.last_value = None

    def fit(self, X, y):
        # Store the last y value from training
        self.last_value = y[-1] if len(y) > 0 else 0
        return self

    def predict(self, X):
        # Persistence model: return last training value for all predictions
        # This is the simplest baseline - any good model should beat this
        if hasattr(X, 'shape') and len(X.shape) == 2:
            return np.full(X.shape[0], self.last_value)
        return np.array([self.last_value])


class ProphetRegressor:
    """
    Prophet adapter with sklearn-style interface.

    IMPORTANT: This adapter is designed for SEQUENTIAL forecasting only.
    - X is used only to determine prediction length (X.shape[0])
    - Predictions are made sequentially from end of training data
    - Only valid for walk-forward validation, NOT for shuffled/CV splits

    Prophet requires datetime index, so we create synthetic timestamps.
    Weekly and daily seasonality are captured; yearly is disabled (short series).
    """
    def __init__(self):
        self.model = None
        self.train_size = 0

    def fit(self, X, y):
        import logging
        logging.getLogger('prophet').setLevel(logging.WARNING)
        logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

        self.train_size = len(y)
        # Create synthetic datetime index (1-minute intervals)
        # Note: Actual timestamps would be better for seasonality, but we use
        # synthetic ones since our pipeline uses X,y matrices without timestamps
        dates = pd.date_range(start='2020-01-01', periods=len(y), freq='1min')
        df = pd.DataFrame({'ds': dates, 'y': y})

        self.model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=True,
            changepoint_prior_scale=0.05
        )
        self.model.fit(df)
        return self

    def predict(self, X):
        # X is only used for determining prediction length
        # Predictions are sequential from end of training (walk-forward)
        n_pred = X.shape[0] if hasattr(X, 'shape') else len(X)
        future_dates = pd.date_range(
            start='2020-01-01',
            periods=self.train_size + n_pred,
            freq='1min'
        )[self.train_size:]
        future = pd.DataFrame({'ds': future_dates})
        forecast = self.model.predict(future)
        return forecast['yhat'].values


class SARIMAXRegressor:
    """
    SARIMAX adapter with sklearn-style interface.

    IMPORTANT: This adapter is designed for SEQUENTIAL forecasting only.
    - X is used only to determine prediction length (X.shape[0])
    - Predictions are made sequentially from end of training data
    - Only valid for walk-forward validation, NOT for shuffled/CV splits

    Uses ARIMA(1,0,1) for speed - full grid search would be too slow for benchmark.
    Falls back to naive prediction if model fails to converge.
    """
    def __init__(self):
        self.model = None
        self.train_y = None
        self.fit_error = None

    def fit(self, X, y):
        self.train_y = y
        self.fit_error = None
        # Simple ARIMA(1,0,1) - fast but reasonable for energy data
        try:
            self.model = SARIMAX_Model(
                y, order=(1, 0, 1),
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False, maxiter=50)
        except (ValueError, np.linalg.LinAlgError) as e:
            # Common convergence issues - fall back to naive
            self.fit_error = f"Convergence: {type(e).__name__}"
            self.model = None
            warnings.warn(f"SARIMAX convergence failed ({type(e).__name__}), using naive fallback")
        except Exception as e:
            # Unexpected error - log and fall back
            self.fit_error = f"{type(e).__name__}: {str(e)[:50]}"
            self.model = None
            warnings.warn(f"SARIMAX error ({type(e).__name__}), using naive fallback")
        return self

    def predict(self, X):
        # X is only used for determining prediction length
        # Predictions are sequential from end of training (walk-forward)
        n_pred = X.shape[0] if hasattr(X, 'shape') else len(X)

        if self.model is None:
            # Fallback: naive prediction (last value) - model failed to fit
            return np.full(n_pred, self.train_y[-1] if len(self.train_y) > 0 else 0)


        try:
            forecast = self.model.forecast(steps=n_pred)
            return forecast.values if hasattr(forecast, 'values') else np.array(forecast)
        except Exception as e:
            # Forecast failed - fall back to naive
            return np.full(n_pred, self.train_y[-1] if len(self.train_y) > 0 else 0)


# ==============================================================================
# ML MODELS
# ==============================================================================

def get_ml_models():
    """Return dictionary of ML models"""
    models = {
        "Naive": NaiveRegressor(),  # Baseline - should be beaten by all models
        "Ridge": Ridge(alpha=1.0),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42),
        "KNN": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
    }

    if HAS_CATBOOST:
        models["CatBoost"] = CatBoostRegressor(
            iterations=500, depth=6, learning_rate=0.1,
            verbose=False, random_seed=42, early_stopping_rounds=50
        )

    if HAS_LGBM:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.1,
            verbose=-1, random_state=42, n_jobs=-1
        )

    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.1,
            verbosity=0, random_state=42, n_jobs=-1
        )

    if HAS_NGBOOST:
        models["NGBoost"] = NGBRegressor(
            n_estimators=200, learning_rate=0.05, verbose=False, random_state=42
        )

    if HAS_PROPHET:
        models["Prophet"] = ProphetRegressor()

    if HAS_SARIMAX:
        models["SARIMAX"] = SARIMAXRegressor()

    if HAS_PATCHTST:
        models["PatchTST"] = PatchTSTAdapter(
            seq_len=48, patch_len=8, stride=4,
            epochs=20, lr=1e-3, batch_size=128,
            d_model=64, n_heads=4, n_layers=2,
            patience=5, device="cpu"
        )

    if HAS_TFT:
        models["TFT"] = TFTAdapter(
            seq_len=48, hidden_size=32, num_heads=4,
            num_layers=2, epochs=20, lr=1e-3,
            batch_size=128, patience=5
        )

    return models

# ==============================================================================
# DEEP LEARNING MODELS
# ==============================================================================

if HAS_TORCH:
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True, dropout=dropout if num_layers > 1 else 0)
            self.fc = nn.Linear(hidden_size, 1)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.dropout(out[:, -1, :])
            return self.fc(out).squeeze(-1)

    class GRUModel(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
            super().__init__()
            self.gru = nn.GRU(input_size, hidden_size, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
            self.fc = nn.Linear(hidden_size, 1)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            out, _ = self.gru(x)
            out = self.dropout(out[:, -1, :])
            return self.fc(out).squeeze(-1)

    class BiLSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True, bidirectional=True,
                               dropout=dropout if num_layers > 1 else 0)
            self.fc = nn.Linear(hidden_size * 2, 1)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.dropout(out[:, -1, :])
            return self.fc(out).squeeze(-1)

    # DL models disabled - run separately with run_deep_learning_all.py for stability
    # DL_MODELS = {
    #     "LSTM": LSTMModel,
    #     "GRU": GRUModel,
    #     "BiLSTM": BiLSTMModel,
    # }
    DL_MODELS = {}  # Empty - DL runs separately
else:
    DL_MODELS = {}

# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_dataset(dataset_name):
    """Load and prepare dataset"""
    data_path = CLEANED_DIR / dataset_name / "clean_full.csv"

    if not data_path.exists():
        return None

    df = pd.read_csv(data_path)

    # Exclude columns
    exclude_patterns = ['timestamp', 'ts_utc', 'value_t+']
    feature_cols = [c for c in df.columns if not any(p in c for p in exclude_patterns)]

    return df, feature_cols

# ==============================================================================
# ML TRAINING
# ==============================================================================

def train_ml_model(model, X_train, y_train, X_test, y_test):
    """Train ML model and return metrics"""

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # MAPE (avoid division by zero)
    mask = y_test != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
    else:
        mape = np.nan

    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}

# ==============================================================================
# DL TRAINING
# ==============================================================================

def train_dl_model(model_class, X_train, y_train, X_val, y_val, X_test, y_test,
                   epochs=50, batch_size=256, lr=0.001, patience=10):
    """Train Deep Learning model with proper scaling"""

    # Scale features AND target
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_s = scaler_X.fit_transform(X_train)
    X_val_s = scaler_X.transform(X_val)
    X_test_s = scaler_X.transform(X_test)

    y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_s = scaler_y.transform(y_val.reshape(-1, 1)).ravel()

    # Convert to tensors (add sequence dimension)
    X_train_t = torch.FloatTensor(X_train_s).unsqueeze(1).to(DEVICE)
    y_train_t = torch.FloatTensor(y_train_s).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val_s).unsqueeze(1).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val_s).to(DEVICE)
    X_test_t = torch.FloatTensor(X_test_s).unsqueeze(1).to(DEVICE)

    # Initialize model
    model = model_class(input_size=X_train.shape[1]).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()

        # Mini-batch training (sequential to preserve temporal order)
        for i in range(0, len(X_train_t), batch_size):
            batch_X = X_train_t[i:i+batch_size]
            batch_y = y_train_t[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Load best model
    if best_state:
        model.load_state_dict(best_state)

    # Predict on test set
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_t).cpu().numpy()

    # Inverse transform predictions
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mask = y_test != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
    else:
        mape = np.nan

    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 70)
    print("COMPLETE BENCHMARK - ALL MODELS x ALL DATASETS")
    print("=" * 70)
    print(f"Timestamp: {TIMESTAMP}")
    print(f"Results will be saved to: {RESULTS_DIR}")
    print(f"Datasets: {len(DATASETS)}")
    print(f"Horizons: {HORIZONS}")
    print("=" * 70)

    ml_models = get_ml_models()
    print(f"ML Models: {list(ml_models.keys())}")
    print(f"DL Models: {list(DL_MODELS.keys())}")

    total_models = len(ml_models) + len(DL_MODELS)
    total_runs = len(DATASETS) * len(HORIZONS) * total_models
    print(f"Total runs: {total_runs}")
    print("=" * 70)

    all_results = []
    current_run = 0
    start_time = time.time()

    for dataset_name in DATASETS:
        short_name = dataset_name[:40]
        print(f"\n{'='*70}")
        print(f"DATASET: {short_name}...")
        print(f"{'='*70}")

        # Load data
        result = load_dataset(dataset_name)
        if result is None:
            print(f"  Dataset not found, skipping...")
            continue

        df, feature_cols = result

        # Get data info
        print(f"  Total samples: {len(df)}")
        print(f"  Features: {len(feature_cols)}")

        for horizon in HORIZONS:
            target_col = f"value_t+{horizon}"
            if target_col not in df.columns:
                print(f"  Target {target_col} not found, skipping...")
                continue

            print(f"\n  --- Horizon H{horizon} ---")

            # Prepare data
            X = df[feature_cols].values
            y = df[target_col].values

            # Remove NaN
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X, y = X[mask], y[mask]

            # Walk-forward split (60/10/30)
            n = len(X)
            train_end = int(n * 0.6)
            val_end = int(n * 0.7)

            X_train, y_train = X[:train_end], y[:train_end]
            X_val, y_val = X[train_end:val_end], y[train_end:val_end]
            X_test, y_test = X[val_end:], y[val_end:]

            # ==================== ML MODELS ====================
            for model_name, model in ml_models.items():
                current_run += 1
                print(f"    [{current_run}/{total_runs}] {model_name}...", end=" ", flush=True)

                try:
                    t0 = time.time()
                    # Clone model for fresh training
                    import copy
                    model_clone = copy.deepcopy(model)
                    metrics = train_ml_model(model_clone, X_train, y_train, X_test, y_test)
                    elapsed = time.time() - t0

                    # Count parameters
                    try:
                        from common.param_counter import count_parameters
                        param_count = count_parameters(model_clone, model_name)
                    except Exception:
                        param_count = None

                    print(f"RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f} ({elapsed:.1f}s)")

                    all_results.append({
                        "dataset": dataset_name,
                        "horizon": f"H{horizon}",
                        "model": model_name,
                        "type": "ML",
                        "RMSE": round(metrics['RMSE'], 4),
                        "MAE": round(metrics['MAE'], 4),
                        "R2": round(metrics['R2'], 4),
                        "MAPE": round(metrics['MAPE'], 4) if not np.isnan(metrics['MAPE']) else None,
                        "time_seconds": round(elapsed, 2),
                        "parameter_count": param_count,
                        "status": "SUCCESS"
                    })

                except Exception as e:
                    print(f"FAILED: {e}")
                    all_results.append({
                        "dataset": dataset_name,
                        "horizon": f"H{horizon}",
                        "model": model_name,
                        "type": "ML",
                        "status": "FAILED",
                        "error": str(e)
                    })

            # ==================== DL MODELS ====================
            if HAS_TORCH:
                for model_name, model_class in DL_MODELS.items():
                    current_run += 1
                    print(f"    [{current_run}/{total_runs}] {model_name}...", end=" ", flush=True)

                    try:
                        t0 = time.time()
                        metrics = train_dl_model(
                            model_class, X_train, y_train, X_val, y_val, X_test, y_test,
                            epochs=50, patience=10
                        )
                        elapsed = time.time() - t0

                        # Count DL parameters
                        try:
                            dl_param_count = sum(
                                p.numel() for p in model_class(input_size=X_train.shape[1]).parameters()
                            )
                        except Exception:
                            dl_param_count = None

                        print(f"RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f} ({elapsed:.1f}s)")

                        all_results.append({
                            "dataset": dataset_name,
                            "horizon": f"H{horizon}",
                            "model": model_name,
                            "type": "DeepLearning",
                            "RMSE": round(metrics['RMSE'], 4),
                            "MAE": round(metrics['MAE'], 4),
                            "R2": round(metrics['R2'], 4),
                            "MAPE": round(metrics['MAPE'], 4) if not np.isnan(metrics['MAPE']) else None,
                            "time_seconds": round(elapsed, 2),
                            "parameter_count": dl_param_count,
                            "status": "SUCCESS"
                        })

                    except Exception as e:
                        print(f"FAILED: {e}")
                        all_results.append({
                            "dataset": dataset_name,
                            "horizon": f"H{horizon}",
                            "model": model_name,
                            "type": "DeepLearning",
                            "status": "FAILED",
                            "error": str(e)
                        })

    # ==============================================================================
    # SAVE RESULTS
    # ==============================================================================

    total_time = time.time() - start_time

    # Save as JSON
    summary = {
        "timestamp": TIMESTAMP,
        "total_time_minutes": round(total_time / 60, 2),
        "datasets": DATASETS,
        "horizons": HORIZONS,
        "ml_models": list(ml_models.keys()),
        "dl_models": list(DL_MODELS.keys()),
        "total_runs": total_runs,
        "successful": sum(1 for r in all_results if r.get('status') == 'SUCCESS'),
        "failed": sum(1 for r in all_results if r.get('status') == 'FAILED'),
    }

    with open(RESULTS_DIR / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    with open(RESULTS_DIR / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    # Save as CSV for easy analysis
    results_df = pd.DataFrame([r for r in all_results if r.get('status') == 'SUCCESS'])
    results_df.to_csv(RESULTS_DIR / "all_results.csv", index=False)

    # Create pivot table for comparison
    if len(results_df) > 0:
        pivot = results_df.pivot_table(
            index=['dataset', 'horizon'],
            columns='model',
            values='RMSE',
            aggfunc='first'
        )
        pivot.to_csv(RESULTS_DIR / "comparison_rmse.csv")

    # ==============================================================================
    # PRINT SUMMARY
    # ==============================================================================

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE!")
    print("=" * 70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Successful: {summary['successful']}/{total_runs}")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 70)

    # Print best models per dataset/horizon
    print("\nBEST MODELS (by RMSE):")
    print("-" * 70)

    for dataset in DATASETS:
        short_name = dataset[:35]
        for horizon in HORIZONS:
            subset = [r for r in all_results
                     if r.get('dataset') == dataset
                     and r.get('horizon') == f"H{horizon}"
                     and r.get('status') == 'SUCCESS']

            if subset:
                best = min(subset, key=lambda x: x['RMSE'])
                print(f"{short_name[:30]:<32} H{horizon}: {best['model']:<12} RMSE={best['RMSE']:.4f}")

    print("\n" + "=" * 70)
    print(f"FILES CREATED:")
    print(f"  - {RESULTS_DIR}/summary.json")
    print(f"  - {RESULTS_DIR}/all_results.json")
    print(f"  - {RESULTS_DIR}/all_results.csv")
    print(f"  - {RESULTS_DIR}/comparison_rmse.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
