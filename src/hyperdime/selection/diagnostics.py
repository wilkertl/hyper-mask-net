"""How much of a trained selector's ranking still comes from the geometric prior (F9 research D5).

A learned prior weight alone does not say how much the prior still decides: scaling ``a``, ``W``,
and ``b`` together by ``c > 0`` scales every score by ``c``, which neither the hard top-k nor the
standardized relaxed mask can see. ``prior_share`` compares the spread of the two terms per query
instead, which that scaling leaves unchanged. Computed on validation queries only.
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor

from hyperdime.baselines.learning_to_select import GeometricSelector
from hyperdime.baselines.methods import variance_importance
from hyperdime.evaluation.selector_shift import overlap_at_k
from hyperdime.selection.topk import top_k_mask


@torch.no_grad()
def prior_share(model: GeometricSelector, queries: Tensor, k: int | None = None) -> float:
    """Mean over queries of ``σ(a_k π) / (σ(a_k π) + σ(W q + b))``, with ``σ`` the standard
    deviation over dimensions: 1 when the correction is constant (as at step 0), 0 when the
    prior's weight is 0."""
    prior, correction = model.terms(queries, k)
    spread_prior = prior.std(dim=-1, correction=0)
    spread_correction = correction.std(dim=-1, correction=0)
    return float((spread_prior / (spread_prior + spread_correction)).mean())


@torch.no_grad()
def overlap_with_variance(model: GeometricSelector, queries: Tensor, k: int) -> float:
    """Mean top-k overlap with the ``variance`` method (the selector with ``W = b = 0``)."""
    reference = variance_importance(queries, model.variance)
    return _overlap(model(queries, k), reference, k)


@torch.no_grad()
def overlap_with_correction(model: GeometricSelector, queries: Tensor, k: int) -> float:
    """Mean top-k overlap with the correction ``W q + b`` alone (every prior weight 0)."""
    return _overlap(model(queries, k), model.terms(queries, k)[1], k)


def without_prior(model: GeometricSelector) -> GeometricSelector:
    """A copy with every prior weight set to 0, for the prior-removal ablation."""
    ablated = copy.deepcopy(model)
    with torch.no_grad():
        for name, weight in ablated.named_parameters():
            if name in ("prior_weight", "prior_weights"):
                weight.zero_()
    if not any(n in ("prior_weight", "prior_weights") for n, _ in ablated.named_parameters()):
        raise ValueError("the selector has no learned prior weight to remove")
    return ablated.eval()


def _overlap(scores: Tensor, reference: Tensor, k: int) -> float:
    return float(overlap_at_k(top_k_mask(scores, k), top_k_mask(reference, k)).mean())
