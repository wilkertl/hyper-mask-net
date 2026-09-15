"""Fixed top-k and fixed-ratio selection with query-only masking."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def resolve_k(k: int | float, dim: int) -> int:
    """Resolve an absolute count, or a ratio rho in (0, 1] as floor(rho * dim)."""
    if isinstance(k, bool):
        raise TypeError("k must be an int count or a float ratio.")
    # The epsilon keeps ratios such as 0.29 * 100 from flooring to 28.
    count = math.floor(k * dim + 1e-9) if isinstance(k, float) else k
    if not 1 <= count <= dim:
        raise ValueError(f"k must resolve to an integer in [1, {dim}], got {k}.")
    return count


def top_k_mask(importance: Tensor, k: int | float) -> Tensor:
    """Boolean mask marking each query's k most important dimensions."""
    if importance.ndim != 2:
        raise ValueError("importance must have shape [batch, D].")
    indices = importance.topk(resolve_k(k, importance.shape[-1]), dim=-1).indices
    mask = torch.zeros_like(importance, dtype=torch.bool)
    return mask.scatter(1, indices, torch.ones_like(indices, dtype=torch.bool))


def apply_mask(query_embeddings: Tensor, mask: Tensor) -> Tensor:
    """Zero unselected coordinates; the masked query is deliberately not re-normalized."""
    if query_embeddings.shape != mask.shape:
        raise ValueError("query_embeddings and mask must share a shape.")
    return query_embeddings * mask.to(query_embeddings.dtype)
