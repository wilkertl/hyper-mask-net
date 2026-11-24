"""Exact score matching and its coverage summaries on hand-computable cases (ADR-0014)."""

from itertools import product
from typing import Any

import numpy as np
import pytest

from hyperdime.evaluation.representation_matching import (
    COMPLETE,
    CONTENTION,
    NO_DOCUMENT_IN_CALIPER,
    TOO_FEW_DOCUMENTS,
    allocate_strata,
    exact_matching,
    failure_reason,
    largest_matching_size,
    nearest_unused_size,
    select_stratified,
    stratified_mean,
    summarize,
)


def _a(*values: float) -> np.ndarray:
    return np.array(values, dtype=np.float64)


def _augmenting_size(positives: np.ndarray, candidates: np.ndarray, caliper: float) -> int:
    """Kuhn's maximum bipartite matching, independent of the sorted structure."""
    owner: dict[int, int] = {}

    def attempt(row: int, seen: set[int]) -> bool:
        for column, value in enumerate(candidates):
            if abs(value - positives[row]) <= caliper and column not in seen:
                seen.add(column)
                if column not in owner or attempt(owner[column], seen):
                    owner[column] = row
                    return True
        return False

    return sum(attempt(row, set()) for row in range(positives.size))


def test_nearest_unused_can_miss_a_complete_matching() -> None:
    # Taken first, 0.50 grabs 0.49, its nearest by the lower-index tie, and 0.48 is left nothing.
    assert nearest_unused_size(_a(0.50, 0.48), _a(0.49, 0.51), 0.02) == 1


def test_exact_matching_finds_that_complete_matching() -> None:
    assert largest_matching_size(_a(0.48, 0.50), _a(0.49, 0.51), 0.02) == 2
    rows, columns = exact_matching(_a(0.48, 0.50), _a(0.49, 0.51), 0.02)
    assert rows.tolist() == [0, 1] and columns.tolist() == [0, 1]


def test_matching_size_equals_kuhn_and_the_dynamic_program_on_random_lists() -> None:
    rng = np.random.default_rng(0)
    for _ in range(300):
        positives = np.sort(rng.uniform(0, 1, rng.integers(1, 7)))
        candidates = np.sort(rng.uniform(0, 1, rng.integers(0, 12)))
        caliper = float(rng.choice([0.01, 0.05, 0.2]))
        expected = _augmenting_size(positives, candidates, caliper)
        assert largest_matching_size(positives, candidates, caliper) == expected
        rows, columns = exact_matching(positives, candidates, caliper)
        assert rows.size == expected and np.unique(columns).size == expected
        assert (np.abs(positives[rows] - candidates[columns]) <= caliper).all()


def test_matching_has_the_least_residual_on_a_hand_case() -> None:
    # Residuals 0.01 + 0.01 either way; the lower positions win the tie, and 0.30 is out of reach.
    rows, columns = exact_matching(_a(0.10, 0.12), _a(0.09, 0.11, 0.13, 0.30), 0.05)
    assert rows.tolist() == [0, 1] and columns.tolist() == [0, 1]


def _least_residual(positives: np.ndarray, candidates: np.ndarray, caliper: float) -> float:
    """Brute force over every assignment: the least residual among those of the largest size."""
    best = (0, 0.0)
    for picks in product([None, *range(candidates.size)], repeat=positives.size):
        taken = [(row, column) for row, column in enumerate(picks) if column is not None]
        columns = [column for _, column in taken]
        residuals = [abs(candidates[column] - positives[row]) for row, column in taken]
        if len(set(columns)) == len(columns) and all(r <= caliper for r in residuals):
            total = float(sum(residuals))
            if len(taken) > best[0] or (len(taken) == best[0] and total < best[1]):
                best = (len(taken), total)
    return best[1]


@pytest.mark.parametrize("seed", range(40))
def test_matching_residual_equals_brute_force_on_small_lists(seed: int) -> None:
    rng = np.random.default_rng(seed)
    positives = np.sort(rng.uniform(0, 0.3, rng.integers(1, 5)))
    candidates = np.sort(rng.uniform(0, 0.3, rng.integers(1, 7)))
    rows, columns = exact_matching(positives, candidates, 0.05)
    found = float(np.abs(positives[rows] - candidates[columns]).sum())
    assert found == pytest.approx(_least_residual(positives, candidates, 0.05))


def test_nearest_candidates_suffice_when_many_lie_in_the_caliper() -> None:
    candidates = np.linspace(0.0, 1.0, 2001)
    positives = _a(0.5, 0.5001, 0.5002)
    rows, columns = exact_matching(positives, candidates, 0.02)
    assert rows.size == 3
    assert np.abs(candidates[columns] - positives).max() < 0.001


def test_tied_candidates_resolve_to_the_lowest_position() -> None:
    _, columns = exact_matching(_a(0.5), _a(0.5, 0.5, 0.5), 0.02)
    assert columns.tolist() == [0]


def test_a_document_exactly_at_the_caliper_matches() -> None:
    assert largest_matching_size(_a(0.5), _a(0.75), 0.25) == 1
    assert largest_matching_size(_a(0.5), _a(0.75), 0.125) == 0


def test_failures_say_why_and_none_blames_the_algorithm() -> None:
    assert failure_reason(_a(0.5), _a(0.5), 0.02, 1) == COMPLETE
    assert failure_reason(_a(0.5, 0.6), _a(0.5), 0.02, 1) == TOO_FEW_DOCUMENTS
    assert failure_reason(_a(0.5, 0.9), _a(0.5, 0.51), 0.02, 1) == NO_DOCUMENT_IN_CALIPER
    assert failure_reason(_a(0.5, 0.51), _a(0.505, 0.9), 0.02, 1) == CONTENTION


def test_allocation_keeps_small_strata_and_never_exceeds_a_stratum() -> None:
    assert allocate_strata([69, 0, 0], 500) == [69, 0, 0]
    assert allocate_strata([2317, 4, 0], 500) == [496, 4, 0]
    found = allocate_strata([399, 688, 1075], 500)
    assert sum(found) == 500 and min(found) >= 50
    assert allocate_strata([5, 5, 5], 3) == [1, 1, 1]


def test_selection_reports_weights_and_is_deterministic() -> None:
    counts = {f"q{i}": 2 for i in range(20)} | {f"r{i}": 8 for i in range(2)}
    ids, strata, weights, table = select_stratified(counts, 0, 7)
    assert len(ids) == 7 and (ids, strata.tolist()) == (
        select_stratified(counts, 0, 7)[0],
        select_stratified(counts, 0, 7)[1].tolist(),
    )
    assert table[1] == {
        "positives": "6-20", "eligible_queries": 2, "selected_queries": 2, "sampling_weight": 1.0
    }  # fmt: skip
    assert weights[strata == 0][0] == pytest.approx(20 / table[0]["selected_queries"])


def test_stratified_mean_weights_by_eligible_queries_not_selection() -> None:
    table = [
        {"eligible_queries": 90, "selected_queries": 2},
        {"eligible_queries": 10, "selected_queries": 2},
    ]
    found = stratified_mean(_a(0.0, 0.0, 1.0, 1.0), np.array([0, 0, 1, 1]), table)
    assert found["estimate"] == pytest.approx(0.1)
    assert found["standard_error"] == 0.0


def _summary() -> dict[str, Any]:
    # Two queries, one repeat. Query 0 has 2 positives, both matched; query 1 has 4, 1 matched.
    table = [
        {"positives": "2-5", "eligible_queries": 40, "selected_queries": 2, "sampling_weight": 20.0}
    ]
    nothing = np.array([[np.nan], [np.nan]])
    return summarize(
        np.array([2, 4]),
        np.array([0, 0]),
        np.array([20.0, 20.0]),
        table,
        np.array([[2], [1]]),
        np.array([[COMPLETE], [NO_DOCUMENT_IN_CALIPER]]),
        np.array([[1], [1]]),
        _a(0.8, 0.6),
        np.array([[0.5], [np.nan]]),
        nothing,
        np.array([[0.1], [0.2]]),
        {"positive": _a(0.4, 0.5, 0.6), "reference": _a(0.41, 0.49, 0.6), "unmatched": _a(0.9)},
    )


def test_summary_separates_query_and_pair_rates() -> None:
    found = _summary()
    assert found["sample"]["complete_rate_unweighted"] == 0.5
    assert found["sample"]["pair_rate_unweighted"] == 0.5  # 3 of 6
    assert found["weighted"]["pair_rate"] == 0.5
    assert found["weighted"]["complete_rate"]["estimate"] == 0.5


def test_summary_counts_reasons_and_greedy_false_failures() -> None:
    found = _summary()
    assert found["reasons"]["complete"] == 1
    assert found["nearest_unused"]["query_repeats_greedy_falsely_failed"] == 1
    assert found["nearest_unused"]["greedy_complete_query_repeats"] == 0


def test_summary_reports_residuals_and_which_queries_matched() -> None:
    found = _summary()
    assert found["scores"]["largest_absolute_residual"] == pytest.approx(0.01)
    assert found["scores"]["residual_reference_minus_positive"]["mean"] == pytest.approx(
        0.0, abs=1e-12
    )
    groups = {row["group"]: row for row in found["selection"]}
    assert groups["complete in every repeat"]["queries"] == 1
    assert groups["complete in every repeat"]["positive_cosine"]["mean"] == 0.8
    assert groups["complete in no repeat"]["weighted_share"] == 0.5
    assert found["cosine"]["positive_minus_matched"]["mean"] == pytest.approx(0.3)
