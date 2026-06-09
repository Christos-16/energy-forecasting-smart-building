# common/seed.py
"""
Centralized random seed management for reproducibility.

Usage:
    from common.seed import set_global_seed, GLOBAL_SEED
    set_global_seed()  # Sets all random generators to GLOBAL_SEED
"""
from __future__ import annotations

import os
import random
import numpy as np

# Global seed for all experiments
GLOBAL_SEED = 42


def set_global_seed(seed: int = GLOBAL_SEED) -> None:
    """
    Set random seed for all libraries used in the project.

    This ensures reproducibility across:
    - Python's random module
    - NumPy
    - PyTorch (if available)
    - TensorFlow/Keras (if available)
    - Environment variable for hash seed

    Args:
        seed: The random seed to use (default: GLOBAL_SEED = 42)
    """
    # Python's built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Environment variable for Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    # PyTorch (if available)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    # TensorFlow (if available)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def get_seed() -> int:
    """Return the global seed value."""
    return GLOBAL_SEED
