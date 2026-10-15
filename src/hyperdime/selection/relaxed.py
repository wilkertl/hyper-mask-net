"""Relaxed top-k mask for training selectors on masked retrieval (F5 research D3)."""

from __future__ import annotations

import torch
from torch import Tensor

from hyperdime.selection.topk import top_k_mask


def soft_top_k(scores: Tensor, k: int, beta: float) -> Tensor:
    """``sigmoid((z - t) / beta)`` with ``z`` the per-query standardized scores.

    ``t`` is the midpoint between the k-th and (k+1)-th largest ``z`` and carries no gradient, so
    values above 0.5 are exactly the hard top-k, and the mask tends to it as ``beta -> 0``.
    Standardizing makes ``beta`` independent of the scores' scale. With ``k`` equal to the
    dimension every coordinate is kept.
    """
    if scores.ndim != 2:
        raise ValueError("scores must have shape [batch, D].")
    dim = scores.shape[-1]
    if not 1 <= k <= dim:
        raise ValueError(f"k must be in [1, {dim}], got {k}.")
    if k == dim:
        return torch.ones_like(scores)
    z = (scores - scores.mean(-1, keepdim=True)) / (
        scores.std(-1, correction=0, keepdim=True) + 1e-12
    )
    top = z.detach().topk(k + 1, dim=-1).values
    threshold = (top[:, k - 1] + top[:, k]).unsqueeze(-1) / 2
    return torch.sigmoid((z - threshold) / beta)


def soft_mass_on_topk(mask: Tensor, scores: Tensor, k: int) -> Tensor:
    """Per query, the share of the soft mask's mass that falls on the hard top-k of ``scores``."""
    hard = top_k_mask(scores, k).to(mask.dtype)
    return (mask * hard).sum(-1) / mask.sum(-1)
