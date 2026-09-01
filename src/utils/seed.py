"""Reproducible random-seed setup for Python, NumPy, and optional PyTorch."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


_MAX_NUMPY_SEED = 2**32 - 1


@dataclass(frozen=True)
class SeedReport:
    """Auditable description of the random generators configured for one run."""

    seed: int
    torch_available: bool
    deterministic_algorithms: bool


def set_global_seed(seed: int, *, deterministic_algorithms: bool = True) -> SeedReport:
    """Set every supported random source before model initialization or splitting."""

    _validate_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        return SeedReport(seed, torch_available=False, deterministic_algorithms=False)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return SeedReport(seed, torch_available=True, deterministic_algorithms=deterministic_algorithms)


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= _MAX_NUMPY_SEED:
        raise ValueError(f"seed must be an integer in [0, {_MAX_NUMPY_SEED}]")
