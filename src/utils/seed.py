"""
Reproducibility utilities — pin all sources of randomness.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set seed for all random number generators.

    Covers: Python stdlib, NumPy, PyTorch (CPU + CUDA),
    and HuggingFace via env var.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CuDNN deterministic (slight perf cost)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # HuggingFace Transformers reads this
    os.environ["PYTHONHASHSEED"] = str(seed)