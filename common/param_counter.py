"""
Parameter Counter for All Model Types
======================================
Version: 1.0.0 (2026-02-24)

Provides count_parameters(model, model_name) that returns the total number
of learnable/stored parameters for any of the 20 benchmark models.

Used by run_upgraded_benchmark.py to populate the 'parameter_count' column
in BENCHMARK_RESULTS_FULL.csv for the Computational ROI analysis.
"""

from __future__ import annotations

import warnings
from typing import Any


def count_parameters(model: Any, model_name: str) -> int:
    """Count parameters for any model type.

    Parameters
    ----------
    model : Any
        The fitted model object.
    model_name : str
        Model name string (e.g., 'Ridge', 'CatBoost', 'LSTM', 'Chronos-2').

    Returns
    -------
    int
        Estimated total parameter count.
    """
    name_lower = model_name.lower().replace("-", "").replace("_", "").replace(" ", "")

    # Foundation models — published parameter counts
    if name_lower in ("chronos1", "chronost5base"):
        return 200_000_000
    if name_lower in ("chronos2",):
        return 120_000_000
    if name_lower in ("timesfm",):
        return 200_000_000

    # Naive baseline
    if name_lower == "naive":
        return 1

    # Try specific counting strategies
    try:
        return _count_specific(model, name_lower)
    except Exception:
        pass

    # Generic fallback strategies
    return _count_generic(model)


def _count_specific(model: Any, name_lower: str) -> int:
    """Model-specific parameter counting."""

    # ---- Linear models (sklearn) ----
    if name_lower in ("ridge", "elasticnet", "lasso", "linearregression"):
        n = 0
        if hasattr(model, "coef_"):
            import numpy as np
            n += np.size(model.coef_)
        if hasattr(model, "intercept_"):
            import numpy as np
            n += np.size(model.intercept_)
        return max(n, 1)

    # ---- Tree ensembles (sklearn) ----
    if name_lower in ("randomforest", "extratrees"):
        return _count_tree_ensemble(model)

    # ---- Boosting models ----
    if name_lower == "catboost":
        return _count_catboost(model)
    if name_lower in ("lightgbm", "lgbm"):
        return _count_lightgbm(model)
    if name_lower in ("xgboost", "xgb"):
        return _count_xgboost(model)
    if name_lower == "ngboost":
        return _count_ngboost(model)

    # ---- KNN ----
    if name_lower == "knn":
        if hasattr(model, "_fit_X"):
            return model._fit_X.size
        return 0

    # ---- Statistical models ----
    if name_lower == "prophet":
        return _count_prophet(model)
    if name_lower == "sarimax":
        return _count_sarimax(model)

    # ---- PyTorch models ----
    if name_lower in ("lstm", "gru", "bilstm", "patchtst", "tft"):
        return _count_pytorch(model)

    raise ValueError(f"No specific counter for {name_lower}")


def _count_generic(model: Any) -> int:
    """Generic fallback: try common attribute patterns."""
    # Try PyTorch .parameters()
    if hasattr(model, "parameters") and callable(model.parameters):
        try:
            return sum(p.numel() for p in model.parameters())
        except Exception:
            pass

    # Try sklearn-style .coef_
    if hasattr(model, "coef_"):
        import numpy as np
        n = np.size(model.coef_)
        if hasattr(model, "intercept_"):
            n += np.size(model.intercept_)
        return max(n, 1)

    # Try .model attribute (adapter pattern)
    if hasattr(model, "model") and model.model is not None:
        return count_parameters(model.model, "unknown")

    # Try ._model attribute
    if hasattr(model, "_model") and model._model is not None:
        return count_parameters(model._model, "unknown")

    # Foundation model adapters with PARAM_COUNT class attribute
    if hasattr(model, "PARAM_COUNT"):
        return model.PARAM_COUNT

    warnings.warn(f"Could not count parameters for {type(model).__name__}, returning 0")
    return 0


# =============================================================================
# Model-specific counters
# =============================================================================


def _count_tree_ensemble(model) -> int:
    """Count parameters in sklearn tree ensembles (RF, ExtraTrees)."""
    total = 0
    if hasattr(model, "estimators_"):
        for est in model.estimators_:
            tree = est.tree_
            # Each internal node stores: feature index, threshold, left/right child
            # Each leaf stores: value
            total += tree.node_count * 2  # threshold + value per node
    return max(total, 1)


def _count_catboost(model) -> int:
    """Count parameters in CatBoost model."""
    try:
        # CatBoost: tree_count * avg_leaves * 2
        tree_count = model.tree_count_
        # Estimate: depth=6 means up to 64 leaves per tree
        depth = model.get_param("depth") or 6
        avg_leaves = 2 ** depth
        return tree_count * avg_leaves * 2
    except Exception:
        # Fallback: estimate from model dump size
        try:
            dump = model.save_model(None, format="json")
            return len(dump) // 8  # rough estimate
        except Exception:
            return 500 * 64 * 2  # default: 500 trees, depth 6


def _count_lightgbm(model) -> int:
    """Count parameters in LightGBM model."""
    try:
        booster = model.booster_
        dump = booster.dump_model()
        total = 0
        for tree_info in dump.get("tree_info", []):
            total += tree_info.get("num_leaves", 31) * 2
        return max(total, 1)
    except Exception:
        try:
            return model.n_estimators_ * 62  # 31 leaves * 2 per tree
        except Exception:
            return 500 * 62


def _count_xgboost(model) -> int:
    """Count parameters in XGBoost model."""
    try:
        booster = model.get_booster()
        dump = booster.get_dump()
        total = 0
        for tree_str in dump:
            # Count lines = nodes; each node has a split value or leaf value
            total += tree_str.count("\n") + 1
        return max(total, 1)
    except Exception:
        try:
            return model.n_estimators * 64 * 2  # estimate
        except Exception:
            return 500 * 128


def _count_ngboost(model) -> int:
    """Count parameters in NGBoost model."""
    total = 0
    try:
        for learner in model.learners:
            if hasattr(learner, "tree_"):
                total += learner.tree_.node_count * 2
            elif hasattr(learner, "estimators_"):
                for est in learner.estimators_:
                    total += est.tree_.node_count * 2
            else:
                total += 64  # estimate per learner
    except Exception:
        total = 200 * 64  # fallback
    return max(total, 1)


def _count_prophet(model) -> int:
    """Count parameters in Prophet model.

    Prophet parameters:
    - trend: k (growth rate) + m (offset) + changepoint deltas
    - seasonality: Fourier coefficients for each seasonality
    - Total typically ~51 for default config
    """
    n = 2  # k, m
    try:
        if hasattr(model, "changepoints"):
            n += len(model.changepoints)  # delta per changepoint (~25)
        if hasattr(model, "seasonalities"):
            for name, params in model.seasonalities.items():
                n += params.get("fourier_order", 3) * 2  # sin + cos
        return max(n, 51)
    except Exception:
        return 51  # typical default


def _count_sarimax(model) -> int:
    """Count parameters in SARIMAX model.

    For ARIMA(1,0,1): AR(1) + MA(1) + intercept + sigma2 = 4 parameters.
    """
    try:
        if hasattr(model, "params"):
            return len(model.params)
        if hasattr(model, "model") and hasattr(model.model, "params"):
            return len(model.model.params)
    except Exception:
        pass
    return 4  # typical ARIMA(1,0,1)


def _count_pytorch(model) -> int:
    """Count parameters in PyTorch model.

    Handles both raw nn.Module and adapter classes that wrap a model.
    """
    import torch.nn as nn

    # Direct nn.Module
    if isinstance(model, nn.Module):
        return sum(p.numel() for p in model.parameters())

    # Adapter pattern: look for .model, ._model, .net, ._net
    for attr in ("model", "_model", "net", "_net", "network"):
        inner = getattr(model, attr, None)
        if inner is not None and isinstance(inner, nn.Module):
            return sum(p.numel() for p in inner.parameters())

    # PatchTSTAdapter / TFTAdapter from extra_models_and_metrics.py
    # These store the model differently
    for attr in ("_patch_model", "_tft_model"):
        inner = getattr(model, attr, None)
        if inner is not None and isinstance(inner, nn.Module):
            return sum(p.numel() for p in inner.parameters())

    return 0
