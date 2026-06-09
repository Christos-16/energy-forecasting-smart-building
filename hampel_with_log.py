"""
Hampel Filter with Logging for Outlier Detection
=================================================
Version: 2.0.0 (2025-12-31)

Provides hampel_filter_with_log(series, window_size, n_sigmas)
which returns (cleaned_series, outliers_df).

The Hampel filter uses median absolute deviation (MAD) to detect outliers
in a rolling window, making it robust to extreme values.

Output DataFrame columns: index, original, replaced, median, mad, threshold.
"""

import numpy as np
import pandas as pd

def hampel_filter_with_log(series: pd.Series, window_size: int = 7, n_sigmas: float = 3.0):
    """
    Apply Hampel filter to detect and replace outliers.

    Uses center=False (causal) to prevent look-ahead bias in time series.
    Only past values are used for computing median and MAD.
    """
    x = series.copy().astype(float)
    k = 1.4826  # scale for Gaussian
    # IMPORTANT: center=False ensures only past values are used (causal, no look-ahead)
    med = x.rolling(window_size, center=False, min_periods=1).median()
    mad = k * (x - med).abs().rolling(window_size, center=False, min_periods=1).median()
    diff = (x - med).abs()
    thr = n_sigmas * mad
    mask = diff > thr
    replaced = x.where(~mask, med)
    outliers = pd.DataFrame({
        "index": x.index[mask],
        "original": x[mask].values,
        "replaced": med[mask].values,
        "median": med[mask].values,
        "mad": mad[mask].values,
        "threshold": thr[mask].values,
    })
    return replaced, outliers
