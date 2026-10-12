"""The single seed manager (research R7)."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from hyperdime.contracts.validation import ContractError


def seed_everything(seed: int, *, deterministic: bool = True) -> dict[str, str | bool | None]:
    """Seed ``random``, NumPy, and torch (CPU and CUDA); return what was set, for the manifest.

    ``PYTHONHASHSEED`` is recorded, not set: it only takes effect when the interpreter starts.
    """
    if seed < 0:
        raise ContractError(f"seed must be >= 0, got {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if deterministic:
        torch.backends.cudnn.benchmark = False
    return {
        "seed": str(seed),
        "deterministic": deterministic,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
    }
