"""P3's draws, reference curves, and summaries on hand-computable cases (ADR-0014)."""

import numpy as np
import pytest

from hyperdime.evaluation.representation import (
    rank_correlation,
)
from hyperdime.evaluation.representation_controls import (
    _curve_distribution,
    chance_corrected,
    chance_curve,
    coordinate_scale,
    floor_curve,
    pair_indices,
    query_stream,
    reference_rows,
    row_spearman,
    total_variation,
)


def test_pairs_are_exhaustive_under_the_limit_and_sampled_above_it() -> None:
    rng = np.random.default_rng(0)
    first, second = pair_indices(4, 6, rng)
    assert sorted(zip(first.tolist(), second.tolist(), strict=True)) == [
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
    ]  # fmt: skip
    first, second = pair_indices(5, 3, rng)
    assert first.size == 3 and (first != second).all()


def test_cardinality_alone_forces_overlap_above_half_the_dimensions() -> None:
    assert floor_curve(4).tolist() == [0.0, 0.0, 0.5, 1.0]
    assert chance_curve(4)[-1] == 1.0
    assert (floor_curve(1024) <= chance_curve(1024) + 1e-12).all()


def test_chance_correction_is_undefined_where_chance_is_one() -> None:
    corrected = chance_corrected(np.array([0.5, 1.0]), np.array([0.0, 1.0]))
    assert corrected[0] == 0.5 and np.isnan(corrected[1])


def test_reference_rows_are_distinct_avoid_exclusions_and_are_repeatable() -> None:
    drawn = reference_rows(1000, {1, 2, 3}, 20, query_stream(0, 100, "q"))
    assert drawn.tolist() == sorted(set(drawn.tolist())) and drawn.size == 20
    assert not {1, 2, 3} & set(drawn.tolist())
    again = reference_rows(1000, {1, 2, 3}, 20, query_stream(0, 100, "q"))
    assert (drawn == again).all()
    assert (drawn != reference_rows(1000, {1, 2, 3}, 20, query_stream(0, 101, "q"))).any()


def test_reference_rows_handle_a_corpus_barely_larger_than_the_group() -> None:
    assert reference_rows(5, {0, 1}, 3, query_stream(0, 1, "q")).tolist() == [2, 3, 4]
    assert reference_rows(5, {0, 1}, 9, query_stream(0, 1, "q")).tolist() == [2, 3, 4]
    assert reference_rows(10, {0}, 8, query_stream(0, 1, "q")).size == 8


def test_a_query_draw_does_not_depend_on_which_other_queries_ran() -> None:
    first = query_stream(0, 100, "q1").integers(0, 10**9, 4)
    query_stream(0, 100, "q0").integers(0, 10**9, 4)
    assert (first == query_stream(0, 100, "q1").integers(0, 10**9, 4)).all()


def test_coordinate_scale_is_the_product_of_mean_magnitudes() -> None:
    queries = np.array([[1.0, -3.0, 0.0], [3.0, 1.0, 0.0]])
    documents = np.array([[2.0, -2.0, 4.0], [0.0, 2.0, 4.0], [9.0, 9.0, 9.0]])
    scale = coordinate_scale(queries, documents, np.array([0, 1]))
    assert scale.tolist() == [2.0, 4.0, 1.0]  # mean|q| (2, 2, 0), mean|d| (1, 2, 4); zero keeps 1


def test_curve_summary_gives_each_cardinality_its_own_error_and_query_count() -> None:
    found = _curve_distribution(np.array([[1.0, np.nan], [3.0, 5.0]]))
    assert found["queries"] == [2, 1]
    assert found["mean"] == [2.0, 5.0]
    assert found["se"][0] == pytest.approx(1.0)  # sd sqrt(2), over sqrt(2)
    assert np.isnan(found["se"][1])


def test_row_spearman_matches_the_repository_definition_and_refuses_a_constant_row() -> None:
    rng = np.random.default_rng(4)
    a, b = rng.normal(size=(2, 32))
    assert row_spearman(a[None], b[None])[0] == pytest.approx(rank_correlation(a, b))
    assert np.isnan(row_spearman(np.ones((1, 8)), rng.normal(size=(1, 8)))).all()


def test_rounding_noise_does_not_invent_an_order_among_equal_frequencies() -> None:
    noisy = np.ones((1, 8)) + 1e-15 * np.arange(8)
    assert np.isnan(row_spearman(noisy, noisy)).all()


def test_total_variation_runs_from_equal_to_disjoint() -> None:
    same = np.array([[1.0, 0.0], [1.0, 1.0]])
    assert total_variation(same, same).tolist() == [0.0, 0.0]
    assert total_variation(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])).tolist() == [1.0]
