"""Measures of how differently two selectors weight and select dimensions."""

from __future__ import annotations

import torch
from torch import Tensor


def mean_importance(log_importance: Tensor) -> Tensor:
    """Average predicted distribution over a query battery, shape ``[D]``, summing to 1."""
    return log_importance.float().exp().mean(dim=0)


def overlap_at_k(mask_a: Tensor, mask_b: Tensor) -> Tensor:
    """Per-query fraction of ``mask_a``'s selected dimensions also selected by ``mask_b``."""
    if mask_a.shape != mask_b.shape or mask_a.ndim != 2:
        raise ValueError("masks must share shape [batch, D].")
    return (mask_a & mask_b).sum(dim=-1).float() / mask_a.sum(dim=-1).clamp(min=1).float()


def js_divergence(p: Tensor, q: Tensor) -> float:
    """Jensen-Shannon divergence in nats between two distributions over dimensions."""
    p, q = p.double() / p.sum(), q.double() / q.sum()
    m = (p + q) / 2
    return float(0.5 * (_kl(p, m) + _kl(q, m)))


def spearman(a: Tensor, b: Tensor) -> float:
    """Spearman rank correlation of two vectors (ties broken by position)."""
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be vectors of equal length.")
    ranks_a, ranks_b = _ranks(a), _ranks(b)
    return float(torch.corrcoef(torch.stack([ranks_a, ranks_b]))[0, 1])


def _kl(p: Tensor, q: Tensor) -> Tensor:
    support = p > 0
    return (p[support] * (p[support] / q[support]).log()).sum()


def _ranks(values: Tensor) -> Tensor:
    ranks = torch.empty(values.shape[0], dtype=torch.float64)
    ranks[values.argsort()] = torch.arange(values.shape[0], dtype=torch.float64)
    return ranks
