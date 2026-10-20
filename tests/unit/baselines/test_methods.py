"""E0 importance methods (F3 US3)."""

import torch
from torch.nn import functional as F

from hyperdime.baselines.learning_to_select import BudgetGeometricSelector, LinearSelector
from hyperdime.baselines.methods import (
    budget_importance,
    global_importance,
    oracle_scores,
    prefix_importance,
    random_importance,
    selector_importance,
)
from hyperdime.evaluation.runs import QuerySet, evaluate_methods
from hyperdime.selection.topk import top_k_mask

QUERIES = torch.randn(4, 1024, generator=torch.Generator().manual_seed(0))


def test_every_method_selects_exactly_k() -> None:
    mean = torch.softmax(torch.randn(1024), 0)
    for scores in (
        prefix_importance(QUERIES),
        random_importance(QUERIES),
        global_importance(QUERIES, mean),
        selector_importance(QUERIES, LinearSelector()),
        oracle_scores(QUERIES, torch.randn(4, 1024), torch.randn(4, 1024)),
    ):
        assert top_k_mask(scores, 307).sum(1).tolist() == [307] * 4


def test_prefix_keeps_the_first_k_dimensions() -> None:
    assert top_k_mask(prefix_importance(QUERIES), 3)[0].nonzero().flatten().tolist() == [0, 1, 2]


def test_random_is_reproducible_by_seed() -> None:
    assert torch.equal(random_importance(QUERIES, 1), random_importance(QUERIES, 1))
    assert not torch.equal(random_importance(QUERIES, 1), random_importance(QUERIES, 2))


def test_global_importance_is_shared_by_all_queries() -> None:
    mean = torch.softmax(torch.randn(1024), 0)
    masks = top_k_mask(global_importance(QUERIES, mean), 10)
    assert all(torch.equal(masks[0], row) for row in masks)


def _budget_selector() -> BudgetGeometricSelector:
    model = BudgetGeometricSelector(torch.rand(1024, generator=torch.Generator().manual_seed(1)))
    with torch.no_grad():
        model.prior_weights.copy_(torch.linspace(0.0, 2.0, 9))
        torch.nn.init.normal_(model.projection.weight, generator=torch.Generator().manual_seed(2))
    return model


def test_budget_importance_gives_each_budget_its_own_scores() -> None:
    model = _budget_selector()
    found = budget_importance(QUERIES, model, batch=3)
    assert list(found) == model.budgets.tolist()
    for k, scores in found.items():
        assert scores.shape == (4, 1024) and torch.allclose(scores, model(QUERIES, k), atol=1e-4)


def test_evaluating_budget_importance_equals_each_budgets_scores_alone() -> None:
    docs = F.normalize(torch.randn(30, 1024, generator=torch.Generator().manual_seed(3)), dim=-1)
    doc_ids = [f"d{i}" for i in range(30)]
    queries = F.normalize(docs[:4] + 0.3 * QUERIES, dim=-1)
    query_set = QuerySet("test", ["q0", "q1", "q2", "q3"], queries,
                         {f"q{i}": {f"d{i}": 1} for i in range(4)}, depth=10)  # fmt: skip
    found = budget_importance(queries, _budget_selector())
    budgets = list(found)[:3]
    together = evaluate_methods(
        query_set, {"budget": found}, budgets, docs.numpy(), doc_ids, environment="toy"
    )
    alone = [
        evaluate_methods(
            query_set, {"budget": found[k]}, [k], docs.numpy(), doc_ids, environment="toy"
        ).results[0]
        for k in budgets
    ]
    assert [r.metrics for r in together.results] == [r.metrics for r in alone]
