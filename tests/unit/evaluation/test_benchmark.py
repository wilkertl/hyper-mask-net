"""F12 exact evaluator: ranks with lower-row ties and metrics from positive ranks (T012)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from hyperdime.evaluation.benchmark import (
    AUTHORS_GRID,
    DECILES,
    CorpusScorer,
    masked_query,
    masked_scores,
    positive_ranks,
    query_measures,
)
from hyperdime.evaluation.metrics import query_metrics


def test_grids() -> None:
    assert DECILES == (102, 204, 307, 409, 512, 614, 716, 819, 921, 1024)
    assert AUTHORS_GRID == (103, 205, 308, 410, 512, 615, 717, 820, 922, 1024)


def test_ranks_count_higher_scores_and_tied_lower_rows() -> None:
    scores = torch.tensor([0.5, 0.9, 0.5, 0.1, 0.5], dtype=torch.float64)
    assert positive_ranks(scores, torch.tensor([2, 0, 4])).tolist() == [3, 2, 4]


def test_measures_match_the_project_metric_on_a_hand_ranking() -> None:
    scores = torch.tensor([0.1 * i for i in range(30)], dtype=torch.float64)
    ranking = [f"d{i}" for i in reversed(range(30))]
    judgments = {"d29": 1, "d20": 2, "d0": 1}
    rows = torch.tensor([29, 20, 0])
    measures = query_measures(
        positive_ranks(scores, rows), torch.tensor([1.0, 2.0, 1.0], dtype=torch.float64)
    )
    expected = query_metrics(ranking, judgments, [10, 100])
    assert math.isclose(measures["ndcg@10"], expected["ndcg@10"], rel_tol=1e-12)
    assert math.isclose(measures["ndcg@100"], expected["ndcg@100"], rel_tol=1e-12)
    assert measures["mrr@10"] == 1.0 and measures["recall@100"] == 1.0


def test_the_full_mask_equals_the_full_query() -> None:
    gen = torch.Generator().manual_seed(0)
    docs = torch.randn(40, 1024, generator=gen).half().double()
    query = torch.randn(1024, generator=gen).half().double()
    assert torch.equal(
        masked_scores(docs, query, torch.arange(1024)), masked_scores(docs, query, None)
    )


# --- chunked exact scorer (MS MARCO path) ------------------------------------------------------


def _tied_corpus() -> np.ndarray:
    rng = np.random.default_rng(3)
    docs = rng.standard_normal((257, 16)).astype(np.float16)
    docs[40] = docs[7]  # exact ties: a lower and a higher duplicate of a positive
    docs[200] = docs[7]
    docs[100] = docs[150]
    return docs


@pytest.mark.parametrize("resident", [True, False])
def test_corpus_scorer_ranks_equal_the_full_matrix_ranks(resident: bool) -> None:
    docs = _tied_corpus()
    full = torch.from_numpy(docs.astype(np.float64))
    scorer = CorpusScorer(docs, "cpu", chunk_rows=37, resident=resident, max_elements=50)
    rng = np.random.default_rng(4)
    query = torch.from_numpy(rng.standard_normal(16).astype(np.float16).astype(np.float64))
    masks = [None, torch.tensor([0, 3, 5, 9]), torch.arange(8)]
    positives = [torch.tensor([7, 40]), torch.tensor([150]), torch.tensor([200, 0, 100])]
    batch = torch.stack([masked_query(query, m) for m in masks])
    got = scorer.positive_ranks(batch, positives)
    for mask, pos, ranks in zip(masks, positives, got, strict=True):
        assert torch.equal(ranks, positive_ranks(masked_scores(full, query, mask), pos))


def test_corpus_scorer_top_keeps_lower_rows_first_and_skips_excluded() -> None:
    docs = _tied_corpus()
    full = torch.from_numpy(docs.astype(np.float64))
    scorer = CorpusScorer(docs, "cpu", chunk_rows=50)
    query = full[7].clone()
    expected = torch.sort(-(full @ query), stable=True).indices.tolist()
    assert scorer.top(query, 20) == expected[:20]
    assert scorer.top(query, 5, exclude=[7]) == [r for r in expected if r != 7][:5]


def test_corpus_scorer_top_many_equals_one_query_at_a_time() -> None:
    docs = _tied_corpus()
    scorer = CorpusScorer(docs, "cpu", chunk_rows=40)
    full = torch.from_numpy(docs.astype(np.float64))
    queries = torch.stack([full[7], full[3], full[150]])
    excludes = [[7], [], [150, 100]]
    many = scorer.top_many(queries, 12, excludes)
    assert many == [scorer.top(q, 12, exclude=e) for q, e in zip(queries, excludes, strict=True)]


def test_corpus_scorer_variance_matches_the_population_variance() -> None:
    docs = _tied_corpus()
    expected = torch.from_numpy(docs.astype(np.float64)).var(0, unbiased=False)
    assert torch.allclose(CorpusScorer(docs, "cpu", chunk_rows=40).variance(), expected, rtol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (the failing case)")
def test_streaming_scorer_accepts_a_gpu_index() -> None:
    """Regression: MS MARCO S1 passed GPU row indices to the streaming (non-resident) scorer."""
    docs = _tied_corpus()
    scorer = CorpusScorer(docs, "cuda", chunk_rows=37, resident=False)
    rows = torch.tensor([7, 40, 150], device="cuda")
    expected = torch.from_numpy(docs[[7, 40, 150]].astype(np.float64))
    assert torch.equal(scorer.rows_f64(rows).cpu(), expected)
