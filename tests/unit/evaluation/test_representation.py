"""P3's arithmetic on hand-computable pairs (ADR-0014)."""

import itertools

import numpy as np
import pytest
import torch

from hyperdime.evaluation.representation import (
    aggregation_diagnostics,
    average_ranks,
    budget_masks,
    chance_jaccard,
    contribution_order,
    contributions,
    cumulative_scores,
    gain_weights,
    pair_statistics,
    query_agreement,
    query_balanced,
    select_queries,
    split_assignments,
)
from hyperdime.oracle.targets import aggregate_pairs

QUERY = np.array([1.0, -2.0, 0.5])
DOCUMENT = np.array([[2.0, 1.0, 4.0]])  # c = (2, -2, 2), dot = 2


def _statistics() -> dict[str, np.ndarray]:
    return pair_statistics(QUERY, DOCUMENT, budgets=[1, 2, 3], curve_k=[1, 2, 3])


def test_contributions_sum_exactly_to_the_inner_product() -> None:
    values = contributions(QUERY, DOCUMENT)
    assert values.tolist() == [[2.0, -2.0, 2.0]]
    assert values.sum() == float(DOCUMENT[0] @ QUERY) == 2.0


def test_pair_statistics_split_signed_mass_on_a_hand_case() -> None:
    stats = _statistics()
    assert stats["score"].tolist() == [2.0]
    assert stats["positive_mass"].tolist() == [4.0]
    assert stats["negative_mass"].tolist() == [2.0]
    assert stats["negative_share"].tolist() == pytest.approx([1 / 3])
    assert stats["sign_agreement"].tolist() == pytest.approx([2 / 3])
    assert (stats["n_positive"][0], stats["n_negative"][0], stats["n_zero"][0]) == (2, 1, 0)
    assert stats["effective_positive_dims"].tolist() == pytest.approx([2.0])  # 16 / (4 + 4)
    assert stats["dims_for_50"].tolist() == [1.0]
    assert stats["dims_for_99"].tolist() == [2.0]
    assert stats["k_return"].tolist() == [1.0]  # the first contribution already reaches 2


def test_pair_statistics_record_what_each_budget_removes() -> None:
    stats = _statistics()
    # Ordering (0, 2, 1): curve 2, 4, 2. The last dimension in is the negative one.
    assert stats["max_score"].tolist() == [[2.0, 4.0, 2.0]]
    assert stats["removed_positive_mass"].tolist() == [[2.0, 0.0, 0.0]]
    assert stats["removed_negative_mass"].tolist() == [[2.0, 2.0, 0.0]]
    assert stats["preservation_error"][0].tolist() == pytest.approx([0.0, 2 / 6, 0.0])
    assert stats["gain_over_full"][0].tolist() == pytest.approx([0.0, 2 / 6, 0.0])
    assert stats["curve"].tolist() == [[0.5, 1.0, 0.5]]


def test_removed_mass_accounts_for_the_change_from_the_full_score() -> None:
    rng = np.random.default_rng(0)
    q, docs = rng.normal(size=16), rng.normal(size=(4, 16))
    stats = pair_statistics(q, docs, budgets=[3, 8, 15], curve_k=[16])
    change = stats["max_score"] - stats["score"][:, None]
    assert change == pytest.approx(stats["removed_negative_mass"] - stats["removed_positive_mass"])


def test_magnitude_shares_are_exact_and_skip_zero_coordinates() -> None:
    rng = np.random.default_rng(1)
    q, docs = rng.normal(size=32), rng.normal(size=(3, 32))
    q[0], docs[1, 5] = 0.0, 0.0
    stats = pair_statistics(q, docs, budgets=[4], curve_k=[32])
    total = (
        stats["log_variance_query_share"]
        + stats["log_variance_document_share"]
        + stats["log_variance_covariance_share"]
    )
    assert total == pytest.approx(np.ones(3))


def test_zero_positive_mass_gives_undefined_ratios_rather_than_infinities() -> None:
    stats = pair_statistics(np.array([1.0, 1.0]), np.array([[-1.0, -2.0]]), [1], [2])
    assert stats["positive_mass"].tolist() == [0.0]
    assert np.isnan(stats["dims_for_50"]).all() and np.isnan(stats["top_1_share"]).all()
    assert stats["max_score"].tolist() == [[-1.0]]
    assert stats["negative_share"].tolist() == [1.0]


@pytest.mark.parametrize("seed", range(5))
def test_top_k_of_contributions_is_the_exhaustive_optimum_at_every_k(seed: int) -> None:
    rng = np.random.default_rng(seed)
    q, d = rng.normal(size=8), rng.normal(size=(1, 8))
    values = contributions(q, d)
    curve = cumulative_scores(values, contribution_order(values))[0]
    for k in range(1, 9):
        best = max(
            sum(values[0, j] for j in subset) for subset in itertools.combinations(range(8), k)
        )
        assert curve[k - 1] == pytest.approx(best)


def test_exact_cardinality_can_force_a_negative_contribution() -> None:
    values = np.array([[3.0, -1.0, -2.0]])
    curve = cumulative_scores(values, contribution_order(values))[0]
    assert curve.tolist() == [3.0, 2.0, 0.0]  # no mask of 2 dimensions does better than 3 - 1


def test_masks_hold_exactly_k_dimensions_and_ties_go_to_the_lower_index() -> None:
    values = np.array([[1.0, 1.0, 1.0, 0.0], [0.0, 2.0, 2.0, 2.0]])
    order = contribution_order(values)
    assert order.tolist() == [[0, 1, 2, 3], [1, 2, 3, 0]]
    masks = budget_masks(order, 2)
    assert masks.sum(axis=1).tolist() == [2, 2]
    assert masks.tolist() == [[True, True, False, False], [False, True, True, False]]


def test_negative_zero_ties_with_zero_deterministically() -> None:
    assert contribution_order(np.array([[-0.0, 0.0, -0.0]])).tolist() == [[0, 1, 2]]


def test_average_ranks_share_ties() -> None:
    assert average_ranks(np.array([[1.0, 1.0, 2.0, 0.0]])).tolist() == [[1.5, 1.5, 3.0, 0.0]]


def test_chance_jaccard_on_a_hand_case() -> None:
    assert chance_jaccard(256) == pytest.approx(1 / 7)


def _agreement(values: np.ndarray, grades: list[int], budgets: list[int]) -> dict[str, np.ndarray]:
    return query_agreement(values, contribution_order(values), np.array(grades), budgets)


def test_agreement_on_a_hand_computed_query() -> None:
    values = np.array(
        [[3.0, 2.0, 1.0, 1.0], [3.0, 2.0, 1.0, 1.0], [3.0, 1.0, 2.0, -1.0]],
    )
    found = _agreement(values, [1, 1, 1], [2])
    # Masks {0,1}, {0,1}, {0,2}: Jaccards 1, 1/3, 1/3.
    assert found["mean_jaccard"].tolist() == pytest.approx([5 / 9])
    assert found["min_jaccard"].tolist() == pytest.approx([1 / 3])
    # Support (3, 2, 1, 0).
    assert found["core_dims"].tolist() == [1.0]
    assert found["majority_dims"].tolist() == [1.0]
    assert found["minority_dims"].tolist() == [1.0]
    assert found["union_dims"].tolist() == [3.0]
    assert float(found["always_helpful_dims"]) == 3.0
    assert float(found["always_harmful_dims"]) == 0.0
    assert float(found["sign_conflict_dims"]) == 1.0
    assert float(found["sign_conflict_share"]) == 0.25


def test_ordering_agreement_is_one_when_identical_and_minus_one_when_reversed() -> None:
    same = _agreement(np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]), [1, 1], [1])
    reversed_ = _agreement(np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]), [1, 1], [1])
    assert float(same["mean_spearman"]) == pytest.approx(1.0)
    assert float(reversed_["mean_spearman"]) == pytest.approx(-1.0)


def test_agreement_separates_same_grade_and_cross_grade_pairs() -> None:
    values = np.array([[3.0, 2.0, 1.0, 0.0], [3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0]])
    found = _agreement(values, [1, 1, 2], [2])
    assert found["same_grade_jaccard"].tolist() == pytest.approx([1.0])  # masks {0,1}, {0,1}
    assert found["cross_grade_jaccard"].tolist() == pytest.approx([0.0])  # {0,1} against {2,3}


def test_agreement_is_undefined_for_a_single_positive() -> None:
    found = _agreement(np.array([[3.0, 2.0, 1.0]]), [1], [1])
    assert np.isnan(found["mean_jaccard"]).all() and np.isnan(found["mean_spearman"])


def _aggregation(query: np.ndarray, docs: np.ndarray, grades: list[int], k: list[int]):  # type: ignore[no-untyped-def]
    values = contributions(query, docs)
    return aggregation_diagnostics(
        query, docs, values, contribution_order(values), np.array(grades), k
    )


def test_aggregation_on_a_hand_case_shows_what_the_mean_loses() -> None:
    query = np.array([1.0, 1.0])
    docs = np.array([[1.0, -1.0], [-1.0, 1.0]])  # c = docs; gains 3 and 1, weights 3/4 and 1/4
    found = _aggregation(query, docs, [2, 1], [1])
    # Mean c = (1/2, -1/2). Absolute mass 2, with 1 left: half of it cancelled.
    assert float(found["cancelled_share"]) == pytest.approx(0.5)
    # Between: 1/2. Within, per dimension: 3/4 * (1/2)² + 1/4 * (3/2)² = 3/4.
    assert float(found["within_positive_variance_share"]) == pytest.approx(1.5 / 2.0)
    assert float(found["flipped_dims"]) == 2.0  # each dimension is opposed by one positive
    # The mean keeps dimension 0: the first positive's own choice, the second's worse one.
    assert found["mean_overlap"].tolist() == pytest.approx([0.75])
    assert found["min_overlap"].tolist() == [0.0]
    assert found["conflicted_in_mask"].tolist() == [1.0]
    # The second positive goes from its maximum 1 to -1, over its mass 2.
    assert found["worst_regret"].tolist() == pytest.approx([1.0])
    assert found["mean_regret"].tolist() == pytest.approx([0.25])
    # Each dimension is selected by one of two positives; the mean's mask drops one of them.
    assert found["minority_dropped"].tolist() == pytest.approx([0.5])


def test_weighted_mean_identity_matches_the_repository_oracle() -> None:
    rng = np.random.default_rng(2)
    query = rng.normal(size=16)
    docs = rng.normal(size=(5, 16))
    grades = [1, 2, 3, 1, 2]
    weights = gain_weights(grades)
    assert (weights @ contributions(query, docs)) == pytest.approx(query * (weights @ docs))
    found = _aggregation(query, docs, grades, [4])
    assert float(found["identity_error"]) < 1e-14
    aggregated, _ = aggregate_pairs(
        torch.from_numpy(query[None, :]).float(),
        torch.from_numpy(docs).float(),
        [dict(enumerate(grades))],
        [[0]],
    )
    assert aggregated[0].numpy() == pytest.approx((weights @ docs).astype(np.float32), abs=1e-5)


def test_gain_weights_refuse_a_nonrelevant_document() -> None:
    assert gain_weights([2, 1]).tolist() == pytest.approx([0.75, 0.25])
    with pytest.raises(ValueError, match="grade 0"):
        gain_weights([1, 0])


def test_query_balancing_counts_each_query_once() -> None:
    values = np.array([1.0, 1.0, 1.0, 3.0])
    balanced = query_balanced(values, np.array([0, 0, 0, 1]), 2)
    assert balanced.tolist() == [1.0, 3.0]
    assert balanced.mean() == 2.0 and values.mean() == 1.5  # the pair-weighted mean differs


def test_query_balancing_refuses_a_query_with_no_row() -> None:
    with pytest.raises(ValueError, match="every query"):
        query_balanced(np.array([1.0]), np.array([0]), 2)


def test_split_assignments_are_balanced_and_deterministic() -> None:
    first = split_assignments(10, seed=3, draws=4)
    assert (first == split_assignments(10, seed=3, draws=4)).all()
    assert first.sum(axis=1).tolist() == [5, 5, 5, 5]
    assert not (first == split_assignments(10, seed=4, draws=4)).all()


def test_query_selection_is_seeded_nested_and_sorted() -> None:
    ids = [f"q{i}" for i in range(20)]
    small, large = select_queries(ids, 0, 3), select_queries(ids, 0, 6)
    assert small == sorted(small) and set(small) < set(large)
    assert select_queries(reversed(ids), 0, 3) == small
    assert select_queries(ids, 0, None) == sorted(ids)
    assert select_queries(ids, 1, 3) != small
