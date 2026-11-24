"""F12 teachers: masks, GDR, and the exact rank-guided search (T021)."""

from __future__ import annotations

import itertools
import math

import numpy as np
import torch

from hyperdime.evaluation.benchmark import CorpusScorer, masked_scores, positive_ranks
from hyperdime.evaluation.teachers import (
    better,
    contrastive_mask,
    gdr_weights,
    make_objective,
    positive_mask,
    positive_mean,
    rank_guided_search,
    shortlist_search,
    top_k_dims,
)


def test_top_k_breaks_ties_to_the_lower_index() -> None:
    assert top_k_dims(np.array([1.0, 3.0, 3.0, 2.0, 3.0]), 2).tolist() == [1, 2]


def test_positive_and_contrastive_masks_by_hand() -> None:
    q = np.array([1.0, -1.0, 2.0, 0.5])
    positives = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]])
    p_bar = positive_mean(positives, np.array([2.0, 1.0]))  # weights 3/4, 1/4
    assert np.allclose(p_bar, [0.75, 0.25, 0.25, 0.75])
    assert positive_mask(q, p_bar, 2).tolist() == [0, 2]
    assert contrastive_mask(q, p_bar, np.array([0.75, 0.0, 0.0, 0.0]), 2).tolist() == [2, 3]


def test_gdr_by_hand() -> None:
    objective = make_objective(torch.tensor([1.0, 2.0]), torch.device("cpu"))
    assert torch.allclose(
        gdr_weights(torch.tensor([1.0, 2.0])), torch.tensor([0.25, 0.75], dtype=torch.float64)
    )
    ndcg, gdr = objective(torch.tensor([1, 3]))
    assert math.isclose(float(gdr), 0.25 / math.log2(2) + 0.75 / math.log2(4))
    ideal = 2.0 + 1.0 / math.log2(3)
    assert math.isclose(float(ndcg), (1.0 + 2.0 / math.log2(4)) / ideal)


def _brute(
    docs: torch.Tensor, q: torch.Tensor, rows: torch.Tensor, objective, k: int
) -> tuple[float, float]:
    best = (-1.0, -1.0)
    for dims in itertools.combinations(range(docs.shape[1]), k):
        scores = docs[:, list(dims)] @ q[list(dims)]
        value = tuple(float(v) for v in objective(positive_ranks(scores, rows)))
        best = value if better(value, best) else best
    return best


def test_search_never_ends_below_its_start_and_reaches_the_brute_force_optimum() -> None:
    reached = 0
    for seed in range(12):
        gen = torch.Generator().manual_seed(seed)
        docs = torch.randn(60, 10, generator=gen).half().double()
        q = torch.randn(10, generator=gen).half().double()
        rows = torch.tensor([3, 17])
        objective = make_objective(torch.tensor([1.0, 1.0]), torch.device("cpu"))
        start = np.arange(4)
        result = rank_guided_search(docs, q, rows, objective, start, cap=50)
        assert len(result.dims) == 4 and not better(
            (result.start_ndcg, result.start_gdr), (result.ndcg, result.gdr)
        )
        reached += (result.ndcg, result.gdr) == _brute(docs, q, rows, objective, 4)
    assert reached >= 8  # local search; most small cases reach the optimum


def test_search_respects_the_cap() -> None:
    gen = torch.Generator().manual_seed(1)
    docs = torch.randn(200, 12, generator=gen).half().double()
    q = torch.randn(12, generator=gen).half().double()
    objective = make_objective(torch.tensor([1.0]), torch.device("cpu"))
    result = rank_guided_search(docs, q, torch.tensor([150]), objective, np.arange(6), cap=1)
    assert result.steps <= 1 and (result.hit_cap == (result.steps == 1))


def _search_case(seed: int) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, np.ndarray]:
    rng = np.random.default_rng(seed)
    docs = rng.standard_normal((120, 12)).astype(np.float16)
    query = torch.from_numpy(rng.standard_normal(12).astype(np.float16).astype(np.float64))
    positives = torch.tensor([5, 77])
    start = top_k_dims(rng.standard_normal(12), 4)
    return docs, query, positives, start


def test_shortlist_covering_the_corpus_equals_the_exact_search() -> None:
    for seed in range(6):
        docs, query, positives, start = _search_case(seed)
        objective = make_objective(torch.tensor([1.0, 1.0]), torch.device("cpu"))
        full = torch.from_numpy(docs.astype(np.float64))
        exact = rank_guided_search(full, query, positives, objective, start, cap=6)
        scorer = CorpusScorer(docs, "cpu", chunk_rows=33)
        short = shortlist_search(scorer, query, positives, objective, start, cap=6, shortlist=200)
        assert short.dims.tolist() == sorted(exact.dims.tolist())
        assert (short.ndcg, short.gdr, short.steps) == (exact.ndcg, exact.gdr, exact.steps)
        assert short.rejected == 0


def test_short_shortlist_never_ends_below_the_start_and_reports_exact_values() -> None:
    for seed in range(6):
        docs, query, positives, start = _search_case(seed)
        objective = make_objective(torch.tensor([1.0, 1.0]), torch.device("cpu"))
        scorer = CorpusScorer(docs, "cpu", chunk_rows=33)
        short = shortlist_search(scorer, query, positives, objective, start, cap=6, shortlist=5)
        full = torch.from_numpy(docs.astype(np.float64))
        ranks = positive_ranks(masked_scores(full, query, torch.as_tensor(short.dims)), positives)
        assert (short.ndcg, short.gdr) == tuple(float(v) for v in objective(ranks))
        assert not better((short.start_ndcg, short.start_gdr), (short.ndcg, short.gdr))


def test_acceptance_is_strictly_lexicographic_so_ties_keep_the_positive_start() -> None:
    """``teachers`` picks the contrastive search only when ``better(contrastive, positive)``; an
    exact tie is not better, so the positive start wins it (US2, T055)."""
    assert not better((0.5, 0.2), (0.5, 0.2))  # tie: positive kept
    assert better((0.6, 0.0), (0.5, 0.9))  # nDCG@10 first
    assert better((0.5, 0.3), (0.5, 0.2))  # then GDR
    assert not better((0.4, 0.9), (0.5, 0.0))
