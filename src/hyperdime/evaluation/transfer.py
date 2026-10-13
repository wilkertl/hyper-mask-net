"""Transfer of selectors across environments and their fit to targets (F4 research D2–D5).

Seeds are averaged per query before any paired test, so each ordered (train, test) pair is one
test over the test environment's queries. Nothing here reads a judgment: callers pass per-query
metrics, oracle scores, or query ids.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from hyperdime.evaluation.selector_shift import overlap_at_k
from hyperdime.selection.topk import top_k_mask

ALPHA = 0.05


def fit_gain(
    selector_scores: Sequence[Tensor], global_scores: Tensor, oracle_scores: Tensor, k: int
) -> Tensor:
    """Per-query gain of a selector's seeds over global importance at matching the oracle mask.

    ``gain(q) = mean_s overlap@k(selector_s, oracle) - overlap@k(global, oracle)`` (research D2).
    """
    oracle = top_k_mask(oracle_scores, k)
    seeds = torch.stack([overlap_at_k(top_k_mask(s, k), oracle) for s in selector_scores])
    return seeds.mean(dim=0) - overlap_at_k(top_k_mask(global_scores, k), oracle)


def fits(mean_gain: float, p_holm: float, alpha: float = ALPHA) -> bool:
    """The ranking-fit rule: a positive mean gain with a Holm-adjusted p at or below alpha."""
    return mean_gain > 0 and p_holm <= alpha
