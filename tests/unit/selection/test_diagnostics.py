"""The prior's share of a selector's score, its overlaps, and prior removal (F9 research D5)."""

import pytest
import torch

from hyperdime.baselines.learning_to_select import BudgetGeometricSelector, GeometricSelector
from hyperdime.selection.diagnostics import (
    overlap_with_correction,
    overlap_with_variance,
    prior_share,
    without_prior,
)

# D = 4, variance 1, two queries. The prior term is a_k · log(q²) (ε is negligible here).
QUERIES = torch.tensor([[1.0, 0.5, 0.25, 0.125], [0.5, 1.0, 0.5, 0.25]])


def _fixture(weights: tuple[float, float] = (1.0, 0.5)) -> BudgetGeometricSelector:
    model = BudgetGeometricSelector(torch.ones(4), budgets=(1, 2))
    with torch.no_grad():
        model.prior_weights.copy_(torch.tensor(weights))
        model.projection.weight.zero_()
        model.projection.bias.copy_(torch.tensor([0.0, 0.0, 4.0, 4.0]))
    return model


def test_an_untrained_selector_is_all_prior() -> None:
    assert prior_share(BudgetGeometricSelector(torch.ones(4), budgets=(1, 2)), QUERIES, 2) == 1.0


def test_a_zero_weight_leaves_no_prior_share() -> None:
    assert prior_share(_fixture((0.0, 1.0)), QUERIES, 1) == 0.0


def test_the_share_is_the_hand_computed_ratio_of_spreads() -> None:
    # σ of (0, 0, 4, 4) is 2 for both queries. log q² for query 0 is 2·log(1, 1/2, 1/4, 1/8) =
    # -2 ln 2 · (0, 1, 2, 3), σ = 2 ln 2 · sqrt(5/4); at k = 2 with a = 0.5 it is halved.
    # Query 1's log q² is -2 ln 2 · (1, 0, 1, 2), σ = 2 ln 2 · sqrt(1/2); halved as well.
    ln2 = torch.log(torch.tensor(2.0)).item()
    spreads = [0.5 * 2 * ln2 * (5 / 4) ** 0.5, 0.5 * 2 * ln2 * 0.5**0.5]
    expected = sum(s / (s + 2.0) for s in spreads) / 2
    assert prior_share(_fixture(), QUERIES, 2) == pytest.approx(expected, rel=1e-5)


def test_the_share_does_not_change_when_every_term_is_scaled_together() -> None:
    model = _fixture()
    scaled = _fixture((3.0, 1.5))
    with torch.no_grad():
        scaled.projection.bias.mul_(3)
    assert prior_share(scaled, QUERIES, 2) == pytest.approx(prior_share(model, QUERIES, 2))
    assert torch.equal(scaled(QUERIES, 2).topk(2).indices, model(QUERIES, 2).topk(2).indices)


def test_the_overlaps_with_the_variance_method_and_with_the_correction_alone() -> None:
    model = _fixture()
    # k = 2, query 0: scores 0.5·log q² + b = (0, -0.69, 2.61, 1.92): keeps {2, 3}; variance
    # keeps {0, 1} (overlap 0) and the correction keeps {2, 3} (overlap 1). Query 1: (−0.69, 0,
    # 3.31, 2.61): keeps {2, 3}; variance keeps {1, 0} or {1, 2} (ties), correction {2, 3}.
    assert overlap_with_correction(model, QUERIES, 2) == 1.0
    assert overlap_with_variance(model, QUERIES[:1], 2) == 0.0
    untrained = BudgetGeometricSelector(torch.ones(4), budgets=(1, 2))
    assert overlap_with_variance(untrained, QUERIES, 2) == 1.0


def test_prior_removal_zeroes_every_weight_and_leaves_the_original_alone() -> None:
    model = _fixture()
    ablated = without_prior(model)
    assert ablated.prior_weights.tolist() == [0.0, 0.0]
    assert model.prior_weights.tolist() == [1.0, 0.5]
    weighted = without_prior(GeometricSelector(torch.ones(4), learn_weight=True))
    assert weighted.prior_weight is not None and weighted.prior_weight.item() == 0.0
    with pytest.raises(ValueError, match="no learned prior weight"):
        without_prior(GeometricSelector(torch.ones(4)))
