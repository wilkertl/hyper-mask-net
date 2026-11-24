"""Exact full-corpus ranks under masks (ADR-0016): checked against exhaustive sorting."""

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch

from hyperdime.evaluation.metrics import query_metrics
from hyperdime.evaluation.representation_controls import query_stream
from hyperdime.evaluation.representation_engine import Clock
from hyperdime.evaluation.representation_ranking import (
    DIM,
    ROWS,
    MaskPlan,
    ScanBatch,
    Source,
    analyze_rankings,
    expand,
    held_measures,
    masked_vector,
    per_query_summary,
    plan_query,
    sample_queries,
    scan_corpus,
    tracked_scores,
)
from hyperdime.evaluation.representation_shared import (
    PARTITIONS,
    TRANSFER_CELLS,
    partitions,
    shared_order,
)


def _vectors(rows: int, dim: int, seed: int) -> np.ndarray:
    random = np.random.default_rng(seed).normal(size=(rows, dim))
    return (random / np.linalg.norm(random, axis=1, keepdims=True)).astype(np.float16)


def _exhaustive(
    query: np.ndarray, documents: np.ndarray, excluded: int
) -> tuple[list[int], dict[int, int]]:
    scores = [
        (sum(Fraction(float(a)) * Fraction(float(b)) for a, b in zip(query, d, strict=True)), -row)
        for row, d in enumerate(documents)
        if row != excluded
    ]
    ordered = [-row for _, row in sorted(scores, reverse=True)]
    return ordered, {row: rank + 1 for rank, row in enumerate(ordered)}


def _scan(
    masked: np.ndarray,
    tracked: np.ndarray,
    documents: np.ndarray,
    excluded: np.ndarray,
    depth: int,
    chunk: int,
) -> dict[str, np.ndarray]:
    batch = ScanBatch(
        masked, tracked, tracked_scores(masked, tracked, documents, "cpu"), excluded, depth
    )
    scan_corpus(documents, [batch], "cpu", chunk)
    return batch.finish()


def test_float64_scores_equal_exact_rational_arithmetic() -> None:
    documents = _vectors(40, 64, 0)
    query = _vectors(1, 64, 1)
    found = torch.from_numpy(query).double() @ torch.from_numpy(documents).double().T
    for row, doc in enumerate(documents):
        pairs = zip(query[0], doc, strict=True)
        exact = sum(Fraction(float(a)) * Fraction(float(b)) for a, b in pairs)
        assert Fraction(float(found[0, row])) == exact


@pytest.mark.parametrize("chunk", [1, 7, 11, 100])
def test_ranks_and_lists_match_exhaustive_sort_with_duplicates_and_exclusion(chunk: int) -> None:
    documents = _vectors(60, 32, 2)
    documents[10] = documents[3]  # an exact tie at every mask
    documents[11] = documents[3]
    queries = _vectors(4, 32, 3)
    queries[1, 5:] = 0
    tracked = np.array([[3, 10, 11, -1], [3, 40, -1, -1], [59, 0, 20, 30], [10, -1, -1, -1]])
    excluded = np.array([-1, 40, 20, -1])
    found = _scan(queries, tracked, documents, excluded, 5, chunk)
    for mask in range(4):
        ordered, ranks = _exhaustive(queries[mask], documents, int(excluded[mask]))
        assert found["top_rows"][mask].tolist() == ordered[:5]
        for slot, row in enumerate(tracked[mask]):
            if row < 0 or row == excluded[mask]:
                assert found["rank"][mask, slot] == 0
            else:
                assert found["rank"][mask, slot] == ranks[int(row)]


def test_duplicate_documents_rank_by_row_and_report_ties() -> None:
    documents = _vectors(20, 16, 4)
    documents[5] = documents[2]
    queries = _vectors(1, 16, 5)
    tracked = np.array([[2, 5]])
    found = _scan(queries, tracked, documents, np.array([-1]), 3, 6)
    assert found["rank"][0, 1] == found["rank"][0, 0] + 1
    assert found["ties"][0].tolist() == [1, 1]


def test_scan_is_independent_of_chunk_size_and_mask_batch() -> None:
    documents = _vectors(50, 16, 6)
    queries = _vectors(6, 16, 7)
    tracked = np.tile(np.array([[1, 2]]), (6, 1))
    whole = _scan(queries, tracked, documents, np.full(6, -1), 4, 50)
    first = _scan(queries[:2], tracked[:2], documents, np.full(2, -1), 4, 3)
    assert np.array_equal(whole["rank"][:2], first["rank"])
    assert np.array_equal(whole["top_rows"][:2], first["top_rows"])
    assert np.allclose(whole["corpus_mean"][:2], first["corpus_mean"], atol=1e-15)


def test_rejects_a_tracked_document_that_never_meets_its_own_score() -> None:
    documents = _vectors(10, 8, 8)
    queries = _vectors(1, 8, 9)
    tracked = np.array([[1]])
    wrong = np.array([[123.0]])
    batch = ScanBatch(queries, tracked, wrong, np.array([-1]), 2)
    scan_corpus(documents, [batch], "cpu", 4)
    with pytest.raises(RuntimeError):
        batch.finish()


def test_masked_query_keeps_values_and_is_not_renormalized() -> None:
    query = _vectors(1, DIM, 10)[0]
    order = np.argsort(-np.abs(query))
    masked = masked_vector(query, order, 10)
    kept = np.flatnonzero(masked)
    assert kept.size == 10
    assert set(kept) == set(order[:10].tolist())
    assert np.array_equal(masked[kept], query[kept])
    assert np.array_equal(masked_vector(query, order, DIM), query)


def _query(count: int, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    return _vectors(1, DIM, seed)[0], _vectors(count, DIM, seed + 1).astype(np.float64)


def test_construction_masks_use_stored_partitions_and_never_held_out_documents() -> None:
    query, docs = _query(6)
    grades = np.ones(6, dtype=np.int64)
    sources = plan_query(0, "q", query, docs, grades, seed=3, device="cpu", own=False)
    cell = [s for s in sources if s.row == ROWS.index("2-2")]
    position = next(i for i, c in enumerate(TRANSFER_CELLS) if c[0] == "2-2")
    expected = partitions(6, 2, 2, query_stream(3, 400 + position, "q"), PARTITIONS)
    assert len(cell) == len(expected)
    for source, (kept, held) in zip(cell, expected, strict=True):
        assert np.array_equal(source.construction, kept)
        assert np.array_equal(source.held, held)
        assert not set(kept.tolist()) & set(held.tolist())
    # Changing a held-out document must leave the mask of its partition untouched.
    held_row = int(cell[0].held[0])
    changed = docs.copy()
    changed[held_row] = -changed[held_row]
    again = plan_query(0, "q", query, changed, grades, seed=3, device="cpu", own=False)
    other = [s for s in again if s.row == ROWS.index("2-2")]
    assert held_row not in cell[0].construction
    assert np.array_equal(cell[0].order, other[0].order)


def test_shared_row_uses_the_gain_weighted_mean_of_all_positives() -> None:
    query, docs = _query(3)
    grades = np.array([1, 2, 1])
    sources = plan_query(0, "q", query, docs, grades, seed=0, device="cpu", own=False)
    shared = next(s for s in sources if s.row == ROWS.index("shared"))
    values = torch.from_numpy(query.astype(np.float64) * docs)
    expected = shared_order(values[None], torch.tensor([[1.0, 3.0, 1.0]], dtype=torch.float64))
    assert np.array_equal(shared.order, expected[0].numpy().astype(np.int16))


def test_two_positives_are_used_in_both_directions() -> None:
    query, docs = _query(2)
    sources = plan_query(0, "q", query, docs, np.ones(2, np.int64), seed=0, device="cpu", own=False)
    cell = [s for s in sources if s.row == ROWS.index("1-1")]
    assert [(int(s.construction[0]), int(s.held[0])) for s in cell] == [(0, 1), (1, 0)]


def test_expansion_gives_the_full_query_once_and_every_budget_to_the_rest() -> None:
    sources = [Source(0, ROWS.index("full")), Source(0, ROWS.index("own"), positive=1)]
    plan = expand(sources, [10, 20], np.array([3]))
    assert plan.source.tolist() == [0, 1, 1]
    assert plan.budget.tolist() == [-1, 0, 1]
    assert plan.tracked.tolist() == [3, 1, 1]


def test_secondary_rank_skips_only_construction_positives_above() -> None:
    found = held_measures(np.array([3, 8]), np.array([1, 5, 20]), np.array([1, 1]))
    assert found["secondary_rr"] == pytest.approx((1 / 2 + 1 / 6) / 2)
    assert found["secondary_recall10"] == pytest.approx(1.0)


def test_restricted_held_ndcg_keeps_construction_positions_with_no_gain() -> None:
    # Held documents at ranks 2 and 4; ideal is ranks 1 and 2 with the same gains.
    found = held_measures(np.array([2, 4]), np.array([1, 3]), np.array([1, 2]))
    gains_at = 1 / np.log2(3) + 2 / np.log2(5)
    ideal = 2 / np.log2(2) + 1 / np.log2(3)
    assert found["held_ndcg10"] == pytest.approx(gains_at / ideal)


def test_summary_pairs_each_arm_with_the_same_documents_under_the_full_query() -> None:
    held = np.array([1])
    built = np.array([0])
    sources = [
        Source(0, ROWS.index("full")),
        Source(0, ROWS.index("1-1"), partition=0, construction=built, held=held),
    ]
    plan = expand(sources, [10], np.array([2]))
    offsets = np.array([0, 2, 4])
    results = {
        "position": np.array([0, 1, 0, 1]),
        "rank": np.array([5, 40, 3, 20]),
        "score": np.array([0.5, 0.2, 0.6, 0.4]),
        "corpus_mean": np.array([0.1, 0.05]),
        "corpus_var": np.array([0.01, 0.01]),
        "score_at": np.array([[0.7, 0.3, 0.2], [0.8, 0.35, 0.25]]),
        **{m: np.array([0.2, 0.4]) for m in ("ndcg@10", "mrr@10", "mrr@100", "recall@100")},
    }
    found = per_query_summary(plan, results, offsets, 1, [10], [np.array([1, 1])])
    row = ROWS.index("1-1")
    assert found["rr"][row, 0, 0] == pytest.approx(1 / 20)
    assert found["base_rr"][row, 0, 0] == pytest.approx(1 / 40)
    assert found["improved"][row, 0, 0] == 1.0
    assert found["worsened"][row, 0, 0] == 0.0
    assert found["ndcg@10"][row, 0, 0] == pytest.approx(0.4)
    assert found["ndcg@10"][ROWS.index("full"), 1, 0] == pytest.approx(0.2)
    assert found["score"][row, 0, 0] == pytest.approx(0.4)
    assert found["base_score"][row, 0, 0] == pytest.approx(0.2)
    assert found["rank10_gain"][row, 0, 0] == pytest.approx(0.05)
    assert found["margin10"][row, 0, 0] == pytest.approx(0.05)
    assert found["base_margin10"][row, 0, 0] == pytest.approx(-0.1)
    assert np.isnan(found["rr"][ROWS.index("shared"), 0, 0])


def test_partitions_of_a_query_average_before_the_query_counts_once() -> None:
    sources = [
        Source(0, ROWS.index("full")),
        Source(0, ROWS.index("1-1"), partition=0, construction=np.array([0]), held=np.array([1])),
        Source(0, ROWS.index("1-1"), partition=1, construction=np.array([1]), held=np.array([0])),
    ]
    plan = MaskPlan(sources, np.array([0, 1, 2]), np.array([-1, 0, 0]), np.array([2, 2, 2]))
    results = {
        "position": np.array([0, 1, 0, 1, 0, 1]),
        "rank": np.array([4, 2, 1, 8, 4, 2]),
        "score": np.zeros(6),
        "corpus_mean": np.zeros(3),
        "corpus_var": np.ones(3),
        "score_at": np.zeros((3, 3)),
        **{m: np.zeros(3) for m in ("ndcg@10", "mrr@10", "mrr@100", "recall@100")},
    }
    found = per_query_summary(plan, results, np.array([0, 2, 4, 6]), 1, [10], [np.ones(2)])
    assert found["rr"][ROWS.index("1-1"), 0, 0] == pytest.approx((1 / 8 + 1 / 4) / 2)
    assert found["improved"][ROWS.index("1-1"), 0, 0] == pytest.approx(0.0)
    assert found["worsened"][ROWS.index("1-1"), 0, 0] == pytest.approx(0.5)


def test_retrieved_lists_feed_the_repository_metrics_unchanged() -> None:
    documents = _vectors(30, 16, 13)
    query = _vectors(1, 16, 14)
    found = _scan(query, np.array([[1, 2]]), documents, np.array([-1]), 10, 4)
    ids = [f"d{i}" for i in range(30)]
    judged = {"d1": 1, "d2": 1, f"d{found['top_rows'][0, 0]}": 2}
    ranking = [ids[r] for r in found["top_rows"][0]]
    assert query_metrics(ranking, judged, (10,))["mrr@10"] == 1.0


def test_sample_is_deterministic_and_bounded() -> None:
    ids = [f"q{i}" for i in range(30)]
    assert sample_queries(ids, 0) == sample_queries(ids, 0)
    assert len(sample_queries(ids, 0)) == 10


def test_every_arm_breaks_exact_ties_by_the_lower_corpus_row(tmp_path: Path) -> None:
    documents = _vectors(40, DIM, 20)
    documents[7] = documents[3]  # identical under any mask
    positives = {"q": np.array([3, 7, 12])}
    summary = analyze_rankings(
        ["q"],
        lambda _: _vectors(1, DIM, 21)[0],
        documents,
        [f"d{i}" for i in range(40)],
        positives,
        {"q": np.ones(3, dtype=np.int64)},
        {"q": {"d3": 1, "d7": 1, "d12": 1}},
        seed=0,
        budgets=[64, 512],
        device="cpu",
        clock=Clock("cpu"),
        path=lambda name: tmp_path / name,
        log=lambda _: None,
    )
    assert summary["checks"]["tied_tracked_documents"] > 0
    with np.load(tmp_path / "tracked.npz") as found:
        offsets, rank, ties = found["offsets"], found["rank"], found["ties"]
    with np.load(tmp_path / "masks.npz") as found:
        rows = found["row"]
    arms = {int(row) for row in rows if row != ROWS.index("own")}
    assert {ROWS.index("full"), ROWS.index("prefix"), ROWS.index("shared")} <= arms
    for mask, row in enumerate(rows):
        if row == ROWS.index("own"):
            continue
        at = offsets[mask]
        assert rank[at + 1] == rank[at] + 1
        assert ties[at] >= 1


def test_construction_positives_are_measured_apart_from_held_out_ones() -> None:
    sources = [
        Source(0, ROWS.index("full")),
        Source(0, ROWS.index("1-1"), partition=0, construction=np.array([0]), held=np.array([1])),
    ]
    plan = expand(sources, [10], np.array([2]))
    results = {
        "position": np.array([0, 1, 0, 1]),
        "rank": np.array([50, 40, 1, 60]),
        "score": np.zeros(4),
        "corpus_mean": np.zeros(2),
        "corpus_var": np.ones(2),
        "score_at": np.zeros((2, 3)),
        **{m: np.zeros(2) for m in ("ndcg@10", "mrr@10", "mrr@100", "recall@100")},
    }
    found = per_query_summary(plan, results, np.array([0, 2, 4]), 1, [10], [np.ones(2)])
    row = ROWS.index("1-1")
    assert found["construction_rr"][row, 0, 0] == pytest.approx(1.0)
    assert found["base_construction_rr"][row, 0, 0] == pytest.approx(1 / 50)
    assert found["construction_recall10"][row, 0, 0] == 1.0
    assert found["rr"][row, 0, 0] == pytest.approx(1 / 60)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gpu_scan_is_bit_identical_to_cpu() -> None:
    documents = _vectors(200, 64, 15)
    documents[17] = documents[4]
    queries = _vectors(8, 64, 16)
    tracked = np.tile(np.array([[4, 17, 100]]), (8, 1))
    excluded = np.full(8, -1)
    cpu = _scan(queries, tracked, documents, excluded, 20, 37)
    batch = ScanBatch(
        queries, tracked, tracked_scores(queries, tracked, documents, "cuda"), excluded, 20
    )
    scan_corpus(documents, [batch], "cuda", 37)
    gpu = batch.finish()
    assert np.array_equal(cpu["rank"], gpu["rank"])
    assert np.array_equal(cpu["top_rows"], gpu["top_rows"])
    assert np.array_equal(cpu["top_scores"], gpu["top_scores"])
