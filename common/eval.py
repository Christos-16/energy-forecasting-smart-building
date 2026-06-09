# common/eval.py
from __future__ import annotations
import numpy as np
import pandas as pd

def mae(y_true, y_pred): return float(np.mean(np.abs(y_true - y_pred)))
def rmse(y_true, y_pred): return float(np.sqrt(np.mean((y_true - y_pred)**2)))
def mape(y_true, y_pred):
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    mask = np.abs(y_true) > 1e-8
    return float(np.mean(np.abs((y_true[mask]-y_pred[mask]) / y_true[mask])) * 100)

def residual_anomaly_scores(resid: pd.Series) -> pd.Series:
    """Z-score για γρήγορα spikes."""
    r = (resid - resid.mean()) / (resid.std() + 1e-8)
    return r.abs()