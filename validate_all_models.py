#!/usr/bin/env python3
"""
Validate All 18 Models - Smoke Test
====================================
Quick validation that all 18 existing models can train and predict on the
re-cleaned data without NaN crashes or errors.

Uses a small subset of one sensor for speed (~5000 train, ~1000 test rows).
"""

import copy
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
CLEANED_DIR = BASE_DIR / "cleaned_data"

# Use a re-cleaned sensor (one of the 4 with guaranteed leak-safe data)
TEST_SENSOR = "ZWave2_Metered_Wall_Plug_Switch_computers_Sensor_power"
TARGET_COL = "value_t+1"  # horizon 1

# Subset sizes for quick validation
TRAIN_MAX = 5000
TEST_MAX = 1000


def load_data():
    """Load and prepare data for validation."""
    data_path = CLEANED_DIR / TEST_SENSOR / "clean_full.csv"
    if not data_path.exists():
        print(f"ERROR: {data_path} not found")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows from {TEST_SENSOR}")

    # Exclude timestamp and target columns from features
    exclude_patterns = ["timestamp", "ts_utc", "value_t+"]
    feature_cols = [c for c in df.columns if not any(p in c for p in exclude_patterns)]

    X = df[feature_cols].values
    y = df[TARGET_COL].values

    # Remove NaN rows
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[mask], y[mask]

    # 60/10/30 split
    n = len(X)
    train_end = int(n * 0.6)
    val_end = int(n * 0.7)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    # Subsample for speed
    if len(X_train) > TRAIN_MAX:
        X_train, y_train = X_train[:TRAIN_MAX], y_train[:TRAIN_MAX]
    if len(X_val) > TRAIN_MAX // 5:
        X_val, y_val = X_val[:TRAIN_MAX // 5], y_val[:TRAIN_MAX // 5]
    if len(X_test) > TEST_MAX:
        X_test, y_test = X_test[:TEST_MAX], y_test[:TEST_MAX]

    print(f"Shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def validate_predictions(y_pred, model_name):
    """Check predictions are valid."""
    issues = []
    if y_pred is None:
        issues.append("predictions are None")
    elif len(y_pred) == 0:
        issues.append("empty predictions")
    else:
        if np.any(np.isnan(y_pred)):
            nan_pct = np.isnan(y_pred).mean() * 100
            issues.append(f"{nan_pct:.1f}% NaN")
        if np.any(np.isinf(y_pred)):
            issues.append("contains Inf")
    return issues


def main():
    print("=" * 70)
    print("VALIDATE ALL 18 MODELS - SMOKE TEST")
    print("=" * 70)

    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_data()

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    results = {}

    # =========================================================================
    # 1. ML Models (from run_complete_benchmark.py get_ml_models())
    # =========================================================================
    print("\n--- ML Models ---")

    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
    from sklearn.neighbors import KNeighborsRegressor

    # NaiveRegressor
    class NaiveRegressor:
        def fit(self, X, y):
            self.last_val = y[-1]
            return self
        def predict(self, X):
            return np.full(len(X), self.last_val)

    ml_models = {
        "Naive": NaiveRegressor(),
        "Ridge": Ridge(alpha=1.0),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42),
        "KNN": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
    }

    # Boosting models
    try:
        from catboost import CatBoostRegressor
        ml_models["CatBoost"] = CatBoostRegressor(
            iterations=500, depth=6, learning_rate=0.1,
            verbose=False, random_seed=42, early_stopping_rounds=50
        )
    except ImportError:
        print("  [SKIP] CatBoost not installed")

    try:
        from lightgbm import LGBMRegressor
        ml_models["LightGBM"] = LGBMRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.1,
            verbose=-1, random_state=42, n_jobs=-1
        )
    except ImportError:
        print("  [SKIP] LightGBM not installed")

    try:
        from xgboost import XGBRegressor
        ml_models["XGBoost"] = XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.1,
            verbosity=0, random_state=42, n_jobs=-1
        )
    except ImportError:
        print("  [SKIP] XGBoost not installed")

    try:
        from ngboost import NGBRegressor
        ml_models["NGBoost"] = NGBRegressor(
            n_estimators=200, learning_rate=0.05, verbose=False, random_state=42
        )
    except ImportError:
        print("  [SKIP] NGBoost not installed")

    # Train and validate ML models
    for name, model in ml_models.items():
        try:
            t0 = time.time()
            m = copy.deepcopy(model)
            # CatBoost needs eval_set for early stopping
            if name == "CatBoost":
                m.fit(X_train_s, y_train, eval_set=(X_val_s, y_val), verbose=False)
            else:
                m.fit(X_train_s, y_train)
            y_pred = m.predict(X_test_s)
            elapsed = time.time() - t0

            issues = validate_predictions(y_pred, name)
            if issues:
                results[name] = f"WARN ({', '.join(issues)})"
                print(f"  WARN  {name:15s} | {', '.join(issues)} | {elapsed:.1f}s")
            else:
                r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
                results[name] = "PASS"
                print(f"  PASS  {name:15s} | R2={r2:.4f} | {elapsed:.1f}s")
        except Exception as e:
            results[name] = f"FAIL ({e})"
            print(f"  FAIL  {name:15s} | {e}")

    # =========================================================================
    # 2. Statistical Models (Prophet, SARIMAX)
    # =========================================================================
    print("\n--- Statistical Models ---")

    # Prophet
    try:
        from prophet import Prophet
        t0 = time.time()
        # Prophet needs a DataFrame with ds, y columns
        dates = pd.date_range("2024-01-01", periods=len(y_train), freq="5min")
        train_df = pd.DataFrame({"ds": dates, "y": y_train})
        m = Prophet(yearly_seasonality=False, weekly_seasonality=True,
                    daily_seasonality=True, changepoint_prior_scale=0.05)
        m.fit(train_df)
        future_dates = pd.date_range(dates[-1], periods=len(y_test) + 1, freq="5min")[1:]
        future_df = pd.DataFrame({"ds": future_dates})
        forecast = m.predict(future_df)
        y_pred = forecast["yhat"].values
        elapsed = time.time() - t0

        issues = validate_predictions(y_pred, "Prophet")
        if issues:
            results["Prophet"] = f"WARN ({', '.join(issues)})"
            print(f"  WARN  {'Prophet':15s} | {', '.join(issues)} | {elapsed:.1f}s")
        else:
            r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
            results["Prophet"] = "PASS"
            print(f"  PASS  {'Prophet':15s} | R2={r2:.4f} | {elapsed:.1f}s")
    except ImportError:
        print("  [SKIP] Prophet not installed")
    except Exception as e:
        results["Prophet"] = f"FAIL ({e})"
        print(f"  FAIL  {'Prophet':15s} | {e}")

    # SARIMAX
    try:
        import statsmodels.api as sm
        t0 = time.time()
        # Use only last 500 training points for SARIMAX speed
        sarimax_y = y_train[-500:]
        model_sm = sm.tsa.SARIMAX(
            sarimax_y, order=(1, 0, 1),
            enforce_stationarity=False, enforce_invertibility=False
        ).fit(disp=False, maxiter=50)
        y_pred = model_sm.forecast(steps=min(100, len(y_test)))
        y_pred = y_pred.values if hasattr(y_pred, "values") else np.array(y_pred)
        elapsed = time.time() - t0

        issues = validate_predictions(y_pred, "SARIMAX")
        if issues:
            results["SARIMAX"] = f"WARN ({', '.join(issues)})"
            print(f"  WARN  {'SARIMAX':15s} | {', '.join(issues)} | {elapsed:.1f}s")
        else:
            y_test_sub = y_test[:len(y_pred)]
            r2 = 1 - np.sum((y_test_sub - y_pred) ** 2) / np.sum((y_test_sub - y_test_sub.mean()) ** 2)
            results["SARIMAX"] = "PASS"
            print(f"  PASS  {'SARIMAX':15s} | R2={r2:.4f} | {elapsed:.1f}s")
    except ImportError:
        print("  [SKIP] statsmodels not installed")
    except Exception as e:
        results["SARIMAX"] = f"FAIL ({e})"
        print(f"  FAIL  {'SARIMAX':15s} | {e}")

    # =========================================================================
    # 3. Deep Learning Models (LSTM, GRU, BiLSTM)
    # =========================================================================
    print("\n--- Deep Learning Models ---")

    try:
        import torch
        import torch.nn as nn

        device = torch.device("cpu")

        # Target scaling for DL
        from sklearn.preprocessing import StandardScaler as SS
        scaler_y = SS()
        y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

        # Tensors with sequence dimension
        X_tr_t = torch.FloatTensor(X_train_s).unsqueeze(1).to(device)
        y_tr_t = torch.FloatTensor(y_train_s).to(device)
        X_te_t = torch.FloatTensor(X_test_s).unsqueeze(1).to(device)

        input_size = X_train_s.shape[1]

        dl_models = {
            "LSTM": type("LSTMModel", (nn.Module,), {
                "__init__": lambda self, inp: (
                    super(type(self), self).__init__(),
                    setattr(self, "lstm", nn.LSTM(inp, 64, 2, batch_first=True, dropout=0.2)),
                    setattr(self, "fc", nn.Linear(64, 1)),
                    setattr(self, "drop", nn.Dropout(0.2)),
                )[-1],
                "forward": lambda self, x: self.fc(self.drop(self.lstm(x)[0][:, -1, :])).squeeze(-1),
            }),
            "GRU": type("GRUModel", (nn.Module,), {
                "__init__": lambda self, inp: (
                    super(type(self), self).__init__(),
                    setattr(self, "gru", nn.GRU(inp, 64, 2, batch_first=True, dropout=0.2)),
                    setattr(self, "fc", nn.Linear(64, 1)),
                    setattr(self, "drop", nn.Dropout(0.2)),
                )[-1],
                "forward": lambda self, x: self.fc(self.drop(self.gru(x)[0][:, -1, :])).squeeze(-1),
            }),
            "BiLSTM": type("BiLSTMModel", (nn.Module,), {
                "__init__": lambda self, inp: (
                    super(type(self), self).__init__(),
                    setattr(self, "lstm", nn.LSTM(inp, 64, 2, batch_first=True, bidirectional=True, dropout=0.2)),
                    setattr(self, "fc", nn.Linear(128, 1)),
                    setattr(self, "drop", nn.Dropout(0.2)),
                )[-1],
                "forward": lambda self, x: self.fc(self.drop(self.lstm(x)[0][:, -1, :])).squeeze(-1),
            }),
        }

        for name, model_cls in dl_models.items():
            try:
                t0 = time.time()
                model = model_cls(input_size).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                criterion = nn.MSELoss()

                # Quick train (5 epochs)
                model.train()
                for epoch in range(5):
                    for i in range(0, len(X_tr_t), 256):
                        bx = X_tr_t[i:i + 256]
                        by = y_tr_t[i:i + 256]
                        optimizer.zero_grad()
                        loss = criterion(model(bx), by)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                model.eval()
                with torch.no_grad():
                    y_pred_s = model(X_te_t).cpu().numpy()
                y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
                elapsed = time.time() - t0

                issues = validate_predictions(y_pred, name)
                if issues:
                    results[name] = f"WARN ({', '.join(issues)})"
                    print(f"  WARN  {name:15s} | {', '.join(issues)} | {elapsed:.1f}s")
                else:
                    r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
                    results[name] = "PASS"
                    print(f"  PASS  {name:15s} | R2={r2:.4f} | {elapsed:.1f}s")
            except Exception as e:
                results[name] = f"FAIL ({e})"
                print(f"  FAIL  {name:15s} | {e}")

    except ImportError:
        print("  [SKIP] PyTorch not installed")

    # =========================================================================
    # 4. Transformer Models (PatchTST, TFT)
    # =========================================================================
    print("\n--- Transformer Models ---")

    try:
        from common.extra_models_and_metrics import PatchTSTAdapter, TFTAdapter

        for name, adapter in [
            ("PatchTST", PatchTSTAdapter(
                seq_len=48, patch_len=8, stride=4,
                epochs=3, lr=1e-3, batch_size=128,
                d_model=64, n_heads=4, n_layers=2,
                patience=5, device="cpu"
            )),
            ("TFT", TFTAdapter(
                seq_len=48, hidden_size=32, num_heads=4,
                num_layers=2, epochs=3, lr=1e-3,
                batch_size=128, patience=5
            )),
        ]:
            try:
                t0 = time.time()
                adapter.fit(X_train_s, y_train)
                y_pred = adapter.predict(X_test_s)
                elapsed = time.time() - t0

                issues = validate_predictions(y_pred, name)
                if issues:
                    results[name] = f"WARN ({', '.join(issues)})"
                    print(f"  WARN  {name:15s} | {', '.join(issues)} | {elapsed:.1f}s")
                else:
                    r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
                    results[name] = "PASS"
                    print(f"  PASS  {name:15s} | R2={r2:.4f} | {elapsed:.1f}s")
            except Exception as e:
                results[name] = f"FAIL ({e})"
                print(f"  FAIL  {name:15s} | {e}")

    except ImportError as e:
        print(f"  [SKIP] extra_models_and_metrics import failed: {e}")

    # =========================================================================
    # 5. Chronos-1 (existing foundation model)
    # =========================================================================
    print("\n--- Foundation Models ---")

    try:
        from chronos import ChronosPipeline
        import torch as _torch
        t0 = time.time()
        pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-base",
            device_map="cpu",
            torch_dtype=_torch.bfloat16
        )
        context = _torch.tensor(y_train[-512:]).unsqueeze(0)
        forecast = pipeline.predict(context, prediction_length=min(64, len(y_test)))
        y_pred = forecast.median(dim=1).values.numpy().flatten()
        elapsed = time.time() - t0

        issues = validate_predictions(y_pred, "Chronos-1")
        if issues:
            results["Chronos-1"] = f"WARN ({', '.join(issues)})"
            print(f"  WARN  {'Chronos-1':15s} | {', '.join(issues)} | {elapsed:.1f}s")
        else:
            y_test_sub = y_test[:len(y_pred)]
            r2 = 1 - np.sum((y_test_sub - y_pred) ** 2) / np.sum((y_test_sub - y_test_sub.mean()) ** 2)
            results["Chronos-1"] = "PASS"
            print(f"  PASS  {'Chronos-1':15s} | R2={r2:.4f} | {elapsed:.1f}s")
    except ImportError:
        print("  [SKIP] Chronos not installed")
    except Exception as e:
        results["Chronos-1"] = f"FAIL ({e})"
        print(f"  FAIL  {'Chronos-1':15s} | {e}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v == "PASS")
    warned = sum(1 for v in results.values() if v.startswith("WARN"))
    failed = sum(1 for v in results.values() if v.startswith("FAIL"))
    total = len(results)

    for name, status in results.items():
        symbol = "OK" if status == "PASS" else ("!!" if status.startswith("WARN") else "XX")
        print(f"  [{symbol}] {name:15s} | {status}")

    print(f"\nTotal: {total} | Passed: {passed} | Warnings: {warned} | Failed: {failed}")

    if failed > 0:
        print("\nFAILED models need investigation before proceeding.")
        sys.exit(1)
    else:
        print("\nAll models validated successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
