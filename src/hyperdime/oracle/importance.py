"""Learning-to-Select oracle: r_q = e_q * (p - n) and pi_q = softmax(r_q / tau)."""

from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F


def oracle_scores(query: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
    """Per-dimension contribution of each coordinate to the positive-negative margin."""
    if query.ndim != 2 or not query.shape == positive.shape == negative.shape:
        raise ValueError("query, positive, and negative must share shape [batch, D].")
    return query * (positive - negative)


def oracle_importance(
    query: Tensor, positive: Tensor, negative: Tensor, temperature: float
) -> Tensor:
    """Target importance distribution over dimensions, detached from the graph."""
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    scores = oracle_scores(query, positive, negative).float()
    return F.softmax(scores / temperature, dim=-1).detach()
