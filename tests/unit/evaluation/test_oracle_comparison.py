"""Exact positive masks beside L2S (ADR-0017): each rule on a case computable by hand."""

import numpy as np
import torch

from hyperdime.cli.analyze_oracle_l2s import select
from hyperdime.evaluation.oracle_comparison import (
    Comparison,
    boundary_ties,
    compare_masks,
    one_one_partitions,
    oracle_order,
    pair_order,
    plan_query,
    relationship,
)
from hyperdime.evaluation.representation_engine import Clock


def _vector(*values: float) -> np.ndarray:
    return np.array([*values, *np.zeros(1024 - len(values))], dtype=np.float16)


def test_pair_mask_keeps_the_largest_signed_contributions() -> None:
    query = _vector(1.0, 1.0, 1.0, 1.0)
    document = _vector(-1.0, 0.5, 2.0, 0.0)
    assert pair_order(query, document)[:2].tolist() == [2, 1]


def test_oracle_is_the_top_of_the_relevance_weighted_mean() -> None:
    query = _vector(1.0, 1.0, 1.0)
    docs = np.stack([_vector(1.0, 0.0, 0.5), _vector(0.0, 3.0, 0.5)])
    # Gains 1 and 3 for grades 1 and 2: weighted sums are 1.0, 9.0, 2.0.
    assert oracle_order(query, docs, np.array([1, 2]))[:3].tolist() == [1, 2, 0]


def test_oracle_ties_go_to_the_lower_dimension() -> None:
    docs = np.stack([_vector(1.0, 1.0)])
    assert oracle_order(_vector(1.0, 1.0), docs, np.array([1]))[:2].tolist() == [0, 1]


def test_a_lone_positive_needs_no_mask_of_its_own() -> None:
    query = _vector(1.0, -1.0, 0.5)
    docs = _vector(0.5, 0.5, 0.5)[None]
    sources, splits, identical = plan_query(
        3, "q", query, docs, np.array([1]), np.zeros(1024, np.float32), 0
    )
    assert identical
    assert splits == []
    assert {s.partition for s in sources} == {-1}
    assert len(sources) == 4


def test_a_held_out_mask_is_built_only_from_its_construction_positive() -> None:
    query = _vector(1.0, 1.0, 1.0)
    docs = np.stack([_vector(1.0, 0.0, 0.0), _vector(0.0, 1.0, 0.0)])
    sources, splits, _ = plan_query(
        0, "q", query, docs, np.array([1, 1]), np.zeros(1024, np.float32), 0
    )
    held = [s for s in sources if s.partition >= 0]
    for source, (construction, evaluated) in zip(held, splits, strict=True):
        assert construction.tolist() != evaluated.tolist()
        assert source.order is not None
        assert int(source.order[0]) == int(construction[0])


def test_partitions_are_the_disjoint_one_one_cells() -> None:
    splits = one_one_partitions(3, 0, "q")
    assert len(splits) == 6
    for construction, evaluated in splits:
        assert not set(construction.tolist()) & set(evaluated.tolist())


def test_sign_consensus_counts_only_coordinates_the_oracle_keeps() -> None:
    query = _vector(1.0, 1.0, 1.0)
    docs = np.stack([_vector(1.0, 1.0, -1.0), _vector(1.0, -0.25, -1.0)])
    splits = one_one_partitions(2, 0, "q")
    found = relationship(query, docs, np.array([1, 1]), splits, [1, 2])
    # Sums are 2.0, 0.75, -2.0: dim 0 (+, +) and then dim 1 (+, -).
    assert found["sign_consensus"].tolist() == [1.0, 0.5]
    assert found["positive_consensus"].tolist() == [1.0, 0.5]


def test_oracle_regret_is_each_positive_against_its_own_best() -> None:
    query = _vector(1.0, 1.0)
    docs = np.stack([_vector(4.0, 0.0), _vector(0.0, 5.0)])
    found = relationship(query, docs, np.array([1, 1]), one_one_partitions(2, 0, "q"), [1])
    # The mean of the positives keeps dim 1. Doc 0 loses its 4 of mass 4, doc 1 loses nothing.
    assert found["oracle_regret"].tolist() == [0.5]


def test_boundary_ties_counts_equal_importances_across_the_cut() -> None:
    importance = torch.zeros((1, 1024))
    importance[0, :3] = torch.tensor([3.0, 2.0, 2.0])
    assert boundary_ties(importance, [2, 3]) == 1
    assert boundary_ties(importance, [1024]) == 0


COHORTS = {f"s{i}": 1 for i in range(80)} | {f"m{i}": 3 for i in range(20)}


def test_a_bound_is_split_evenly_between_the_cohorts() -> None:
    _, cohort, weights, _ = select(COHORTS, 0, 10)
    assert cohort.tolist().count(1) == 5
    assert cohort.tolist().count(2) == 5
    assert np.allclose(weights[cohort == 1], 16.0)
    assert np.allclose(weights[cohort == 2], 4.0)


def test_a_small_cohort_hands_its_surplus_to_the_other() -> None:
    counts = {"s0": 1} | {f"m{i}": 3 for i in range(20)}
    _, cohort, _, _ = select(counts, 0, 10)
    assert cohort.tolist().count(1) == 1
    assert cohort.tolist().count(2) == 9


def test_multi_positive_queries_keep_their_count_stratum() -> None:
    counts = {f"a{i}": 3 for i in range(10)} | {f"b{i}": 30 for i in range(10)}
    ids, cohort, _, sampled = select(counts, 0, None)
    assert set(cohort.tolist()) == {2}
    assert len(set(sampled.tolist())) == 2
    assert len(ids) == 20


def test_no_bound_keeps_every_query_at_unit_weight() -> None:
    ids, _, weights, _ = select(COHORTS, 0, None)
    assert len(ids) == 100
    assert np.allclose(weights, 1.0)


def _collision_run(score: float) -> Comparison:
    # Corpus rows: the same-id passage, the positive, an unrelated document.
    documents = np.stack([_vector(score), _vector(0.5), _vector(0.25)])
    return compare_masks(
        ["q"], lambda _: _vector(1.0), documents, ["q", "pos", "other"],
        {"q": np.array([1])}, {"q": np.array([1])}, {"q": {"pos": 1}},
        np.zeros((1, 1024), np.float32),
        seed=0, budgets=(512, 1024), device="cpu", clock=Clock("cpu"), chunk_docs=2,
    )  # fmt: skip


def test_a_passage_with_the_query_id_competes_and_is_charged() -> None:
    found = _collision_run(0.75)
    assert found.checks["documents_excluded_by_id"] == 0
    assert found.checks["same_id_list_inconsistencies"] == 0
    assert found.tracked["rank"].min() == 2
    assert found.tracked["collision_charged"].all()


def test_a_tie_with_the_same_id_passage_goes_to_the_lower_row() -> None:
    found = _collision_run(0.5)
    assert found.tracked["rank"].min() == 2
    assert found.tracked["collision_charged"].all()


def test_a_same_id_passage_below_the_positive_charges_nothing() -> None:
    found = _collision_run(0.125)
    assert found.tracked["rank"].max() == 1
    assert not found.tracked["collision_charged"].any()


def test_every_arm_at_the_full_budget_is_the_full_query() -> None:
    assert _collision_run(0.75).checks["k1024_rank_mismatches"] == 0
    assert _collision_run(0.75).checks["k1024_list_mismatches"] == 0
