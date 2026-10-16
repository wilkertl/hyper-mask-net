"""Stacked masked evaluation (F3 US1)."""

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from hyperdime.evaluation.runs import QuerySet, evaluate_methods, ratio_budgets
from hyperdime.retrieval.exact import exact_search

DOCS = F.normalize(torch.randn(60, 1024, generator=torch.Generator().manual_seed(0)), dim=-1)
DOC_IDS = [f"d{i}" for i in range(60)]
QUERIES = F.normalize(
    DOCS[:8] + 0.3 * torch.randn(8, 1024, generator=torch.Generator().manual_seed(1)), dim=-1
)
QIDS = [f"q{i}" for i in range(8)]
JUDGMENTS = {f"q{i}": {f"d{i}": 2, f"d{i + 10}": 1} for i in range(7)}  # q7 has no judgment


def _set(threshold: int = 1) -> QuerySet:
    return QuerySet("test", QIDS, QUERIES, JUDGMENTS, relevance_threshold=threshold, depth=20)


def _importance(seed: int) -> torch.Tensor:
    return torch.rand(8, 1024, generator=torch.Generator().manual_seed(seed))


def test_k_1024_equals_the_unmasked_rankings() -> None:
    evaluation = evaluate_methods(
        _set(), {"full": None, "random": _importance(0)}, [1024], DOCS.numpy(), DOC_IDS,
        environment="toy", keep_k=(1024,), chunk_size=7,
    )  # fmt: skip
    _, expected = exact_search(QUERIES[:7], DOCS, 20)
    assert evaluation.rankings[("full", 1024)] == [
        [DOC_IDS[i] for i in row] for row in expected.tolist()
    ]
    assert evaluation.rankings[("random", 1024)] == evaluation.rankings[("full", 1024)]


def test_stacked_evaluation_equals_one_pair_at_a_time() -> None:
    methods = {"a": _importance(1), "b": _importance(2)}
    together = evaluate_methods(
        _set(), methods, [64, 512], DOCS.numpy(), DOC_IDS, environment="toy"
    )
    for result in together.results:
        alone = evaluate_methods(
            _set(), {result.selector: methods[result.selector]}, [result.k], DOCS.numpy(), DOC_IDS,
            environment="toy",
        )  # fmt: skip
        assert alone.results[0].per_query == result.per_query


def test_queries_without_relevant_documents_are_excluded() -> None:
    evaluation = evaluate_methods(
        _set(threshold=2), {"full": None}, [1024], DOCS.numpy(), DOC_IDS, environment="toy"
    )
    assert evaluation.excluded_queries == 1 and evaluation.results[0].num_queries == 7
    assert evaluation.results[0].split == "test" and evaluation.results[0].scoring == "exact"


def test_ratio_budgets() -> None:
    assert ratio_budgets() == [102, 204, 307, 409, 512, 614, 716, 819, 921, 1024]
    assert ratio_budgets((0.3,)) == [307]
    with pytest.raises(ValueError):
        ratio_budgets((0.0,))
    assert np.isclose(307 / 1024, 0.2998, atol=1e-3)


def test_a_method_can_give_each_budget_its_own_scores() -> None:
    per_budget = {64: _importance(3), 512: _importance(4)}
    together = evaluate_methods(
        _set(), {"fixed": per_budget}, [64, 512], DOCS.numpy(), DOC_IDS, environment="toy"
    )
    alone = [
        evaluate_methods(
            _set(), {"fixed": scores}, [k], DOCS.numpy(), DOC_IDS, environment="toy"
        ).results[0]
        for k, scores in per_budget.items()
    ]
    assert [r.per_query for r in together.results] == [r.per_query for r in alone]
