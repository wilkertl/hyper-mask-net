"""Descriptor group A: per-dimension marginal statistics of a corpus (protocol P0).

Only document embeddings go in; no queries and no labels.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

MARGINAL_FEATURES: tuple[str, ...] = (
    "mean",
    "std",
    "mean_abs",
    "rms",
    "q05",
    "q25",
    "q50",
    "q75",
    "q95",
)
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def marginal_stats(documents: Tensor) -> Tensor:
    """``[n_docs, D] -> [D, 9]``: one row per dimension, columns in ``MARGINAL_FEATURES`` order.

    The standard deviation is the population one; quantiles interpolate linearly. Quantiles use
    numpy because ``torch.quantile`` rejects inputs above 16M elements (full corpora).
    """
    if documents.ndim != 2 or documents.shape[0] == 0:
        raise ValueError("documents must have shape [n_docs, D] with n_docs >= 1.")
    values = documents.double()
    moments = torch.stack(
        [
            values.mean(dim=0),
            values.std(dim=0, correction=0),
            values.abs().mean(dim=0),
            values.square().mean(dim=0).sqrt(),
        ],
        dim=1,
    )
    quantiles = torch.from_numpy(np.quantile(values.numpy(), QUANTILES, axis=0).T)
    return torch.cat([moments, quantiles], dim=1).float()
