"""Measures of how differently two selectors weight and select dimensions."""

from __future__ import annotations

import torch
from torch import Tensor

from hyperdime.selection.topk import top_k_mask


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
    """Spearman rank correlation of two vectors; tied values share their average rank."""
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be vectors of equal length.")
    ranks_a, ranks_b = _ranks(a), _ranks(b)
    return float(torch.corrcoef(torch.stack([ranks_a, ranks_b]))[0, 1])


def selection_frequency(scores: Tensor, k: int) -> Tensor:
    """Fraction of queries selecting each dimension in their top k, shape ``[D]``, summing to k."""
    return top_k_mask(scores, k).float().mean(dim=0)


def half_split_profiles(scores: Tensor, k: int, draws: int, seed: int) -> tuple[Tensor, Tensor]:
    """Selection profiles of two disjoint random halves of the queries, one pair per draw.

    Returns two ``[draws, D]`` tensors; with an odd number of queries the last one is unused.
    """
    masks = top_k_mask(scores, k).float()
    half = masks.shape[0] // 2
    if half == 0:
        raise ValueError("at least two queries are needed to split them in half.")
    generator = torch.Generator().manual_seed(seed)
    first, second = [], []
    for _ in range(draws):
        order = torch.randperm(masks.shape[0], generator=generator)
        first.append(masks[order[:half]].mean(dim=0))
        second.append(masks[order[half : 2 * half]].mean(dim=0))
    return torch.stack(first), torch.stack(second)


def split_half_spearman(scores: Tensor, k: int, draws: int, seed: int) -> float:
    """Mean Spearman between the selection profiles of random query halves (reliability)."""
    first, second = half_split_profiles(scores, k, draws, seed)
    return sum(spearman(a, b) for a, b in zip(first, second, strict=True)) / draws


def profile_permutation_test(a: Tensor, b: Tensor, k: int, n: int, seed: int) -> float:
    """Permutation p-value that two query sets share one selection profile.

    The statistic is ``1 - Spearman`` of the two sets' profiles; shuffles keep the group sizes.
    Returns ``(hits + 1) / (n + 1)``.
    """
    masks = top_k_mask(torch.cat([a, b]), k).float()
    size = a.shape[0]

    def statistic(rows: Tensor) -> float:
        return 1.0 - spearman(masks[rows[:size]].mean(dim=0), masks[rows[size:]].mean(dim=0))

    observed = statistic(torch.arange(masks.shape[0]))
    generator = torch.Generator().manual_seed(seed)
    hits = sum(
        statistic(torch.randperm(masks.shape[0], generator=generator)) >= observed - 1e-12
        for _ in range(n)
    )
    return (hits + 1) / (n + 1)


def _kl(p: Tensor, q: Tensor) -> Tensor:
    support = p > 0
    return (p[support] * (p[support] / q[support]).log()).sum()


def _ranks(values: Tensor) -> Tensor:
    """0-based ranks; selection frequencies tie often, and ties must not borrow an order."""
    unique: tuple[Tensor, Tensor, Tensor] = torch.unique(
        values, return_inverse=True, return_counts=True
    )
    _, inverse, counts = unique
    ends = counts.cumsum(dim=0).double()
    average = ends - (counts.double() + 1) / 2
    return average[inverse]
