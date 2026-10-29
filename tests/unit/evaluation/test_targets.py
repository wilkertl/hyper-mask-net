"""Per-dimension selector targets for E2, on hand-countable selectors (F10 research D6)."""

import numpy as np
import pytest
import torch

from hyperdime.baselines.learning_to_select import BudgetGeometricSelector, LinearSelector
from hyperdime.evaluation import targets

BUDGETS = (1, 2)
# Three queries over D = 4; an identity selector ranks dimensions by the query's own values.
QUERIES = torch.tensor([[4.0, 3.0, 2.0, 1.0], [1.0, 4.0, 3.0, 2.0], [4.0, 1.0, 3.0, 2.0]])


def _identity(bias: list[float] | None = None) -> LinearSelector:
    model = LinearSelector(dim=4)
    with torch.no_grad():
        model.projection.weight.copy_(torch.eye(4))
        model.projection.bias.copy_(torch.tensor(bias or [0.0] * 4))
    return model.eval()


def _budget(weight: torch.Tensor) -> BudgetGeometricSelector:
    model = BudgetGeometricSelector(torch.ones(4), budgets=BUDGETS)
    with torch.no_grad():
        model.projection.weight.copy_(weight)
        model.prior_weights.fill_(100.0)  # a large prior must not reach the correction profile
    return model.eval()


def test_frequencies_match_a_hand_count_and_each_budget_sums_to_one() -> None:
    profiles = targets.frequency_profiles(_identity(), QUERIES, BUDGETS)
    # Top 1 per query: dims 0, 1, 0. Top 2: {0, 1}, {1, 2}, {0, 2}.
    assert profiles[1] == pytest.approx([2 / 3, 1 / 3, 0.0, 0.0])
    assert profiles[2] == pytest.approx(np.array([2, 2, 2, 0]) / 3 / 2)
    assert all(p.sum() == pytest.approx(1.0) for p in profiles.values())


def test_batched_frequencies_equal_one_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    whole = targets.frequency_profiles(_identity(), QUERIES, BUDGETS)
    monkeypatch.setattr(targets, "BATCH", 2)
    batched = targets.frequency_profiles(_identity(), QUERIES, BUDGETS)
    assert all(np.allclose(whole[k], batched[k]) for k in BUDGETS)


def test_target_profiles_hold_every_key_of_the_data_model() -> None:
    s1 = {0: _identity([0.0, 0.0, 0.0, 9.0]), 1: _identity()}
    s4 = {0: _budget(torch.eye(4)), 1: _budget(-torch.eye(4))}
    found = targets.target_profiles(s1, s4, torch.tensor([1.0, 1.0, 1.0, 100.0]), QUERIES, BUDGETS)
    seed1 = targets.frequency_profiles(_identity(), QUERIES, BUDGETS)
    assert found["s1/seed-1/budget-1"] == pytest.approx(seed1[1])
    assert found["s1/seed-1/mean"] == pytest.approx((seed1[1] + seed1[2]) / 2)
    # Seed 0's bias of 9 on dimension 3 makes it always first: top 1 is dim 3 for every query.
    assert found["s1/seed-0/budget-1"] == pytest.approx([0.0, 0.0, 0.0, 1.0])
    seeds = (found["s1/seed-0/mean"] + found["s1/seed-1/mean"]) / 2
    assert found["s1/mean"] == pytest.approx(seeds)
    assert found["s1/bias"] == pytest.approx([0.0, 0.0, 0.0, 4.5])
    log_softmax = torch.log_softmax(QUERIES, dim=-1).double().mean(dim=0).numpy()
    assert found["s1/seed-1/log_importance"] == pytest.approx(log_softmax)
    # The correction alone: +identity ranks as the query, -identity in reverse; the prior
    # (weight 100) would have ranked every query by q² instead.
    assert found["s4_correction/seed-0/mean"] == pytest.approx(found["s1/seed-1/mean"])
    reverse = targets.frequency_profiles(lambda q: -q, QUERIES, BUDGETS)
    assert found["s4_correction/seed-1/mean"] == pytest.approx((reverse[1] + reverse[2]) / 2)
    # variance: q² · Var with Var = (1, 1, 1, 100) gives [16, 9, 4, 100], [1, 16, 9, 400],
    # [16, 1, 9, 400]: top 1 is dim 3 for all; top 2 adds dims 0, 1, 0. Budget 1 profile
    # [0, 0, 0, 1], budget 2 [2, 1, 0, 3] / 3 / 2; their mean:
    assert found["variance/mean"] == pytest.approx([1 / 6, 1 / 12, 0.0, 3 / 4])


def test_the_validation_sample_is_seeded_capped_and_order_free() -> None:
    ids = [f"q{i}" for i in range(50)]
    sample = targets.validation_sample(ids, size=10)
    assert len(sample) == 10 and set(sample) <= set(ids)
    assert targets.validation_sample(list(reversed(ids)), size=10) == sample
    assert sorted(targets.validation_sample(ids, size=100)) == sorted(ids)


def test_seed_profiles_reads_one_targets_per_seed_means() -> None:
    keys = ["s1/seed-0/mean", "s1/seed-2/mean", "s1/mean", "s1/seed-0/budget-1",
            "s4_correction/seed-0/mean"]  # fmt: skip
    profiles = {key: np.ones(2) for key in keys}
    assert sorted(targets.seed_profiles(profiles)) == [0, 2]
    assert sorted(targets.seed_profiles(profiles, "s4_correction")) == [0]
