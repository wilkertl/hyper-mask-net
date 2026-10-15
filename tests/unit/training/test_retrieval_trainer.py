"""Training the linear selector on masked retrieval (F5 research D4–D6)."""

import pytest
import torch
from torch.nn import functional as F

from hyperdime.training.examples import RetrievalExamples
from hyperdime.training.retrieval_trainer import (
    BUDGETS,
    RetrievalConfig,
    beta_at,
    budget_schedule,
    rerank_ndcg10,
    train_retrieval_selector,
)

DIM, PLANTED = 32, 8


def _planted(n: int, seed: int) -> RetrievalExamples:
    """Positives match the query on dims 0..7 only, negatives on dims 8..31 only, so masked
    retrieval needs exactly the planted dimensions."""
    gen = torch.Generator().manual_seed(seed)
    queries = F.normalize(torch.randn(n, DIM, generator=gen), dim=-1)
    noise = lambda: 0.1 * torch.randn(n, DIM, generator=gen)  # noqa: E731
    planted = torch.zeros(DIM)
    planted[:PLANTED] = 1.0
    positives = queries * planted + noise() * (1 - planted)
    negatives = [2 * queries * (1 - planted) + noise() * planted for _ in range(4)]
    documents = torch.cat([positives, *negatives])
    return RetrievalExamples(
        queries,
        documents,
        torch.arange(n)[:, None],
        torch.ones(n, 1),
        torch.stack([torch.arange(n) + (j + 1) * n for j in range(4)], dim=1),
        skipped=0,
    )


TRAIN, VALIDATION = _planted(256, 0), _planted(64, 1)
CONFIG = RetrievalConfig(
    budgets=(PLANTED,),
    eval_budgets=(PLANTED,),
    lr=1e-2,
    steps=300,
    batch_size=16,
    eval_every=50,
    score_temperature=0.05,
)


@pytest.fixture(scope="module")
def trained() -> tuple[torch.nn.Module, list[dict[str, float]]]:
    return train_retrieval_selector(TRAIN, VALIDATION, CONFIG)


def test_the_selector_recovers_the_planted_dimensions(trained) -> None:  # type: ignore[no-untyped-def]
    model, _ = trained
    top = model(VALIDATION.queries).topk(PLANTED, dim=-1).indices
    recovered = (top < PLANTED).sum(-1)
    assert (recovered >= PLANTED - 1).float().mean() >= 0.9


def test_training_improves_masked_reranking(trained) -> None:  # type: ignore[no-untyped-def]
    _, history = trained
    assert history[0]["step"] == 0
    assert max(h["val_rerank_ndcg10"] for h in history) > history[0]["val_rerank_ndcg10"] + 0.2


def test_the_kept_checkpoint_is_the_best_on_validation(trained) -> None:  # type: ignore[no-untyped-def]
    model, history = trained
    best = max(h["val_rerank_ndcg10"] for h in history)
    assert rerank_ndcg10(model, VALIDATION, (PLANTED,)) == pytest.approx(best)


def test_the_same_seed_gives_the_same_selector() -> None:
    config = RetrievalConfig(**{**CONFIG.__dict__, "steps": 20, "eval_every": 10})
    a, _ = train_retrieval_selector(TRAIN, VALIDATION, config)
    b, _ = train_retrieval_selector(TRAIN, VALIDATION, config)
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())


def test_training_can_start_from_a_given_selector() -> None:
    config = RetrievalConfig(**{**CONFIG.__dict__, "steps": 5, "eval_every": 5, "lr": 0.0})
    start = {"projection.weight": torch.eye(DIM), "projection.bias": torch.zeros(DIM)}
    model, _ = train_retrieval_selector(TRAIN, VALIDATION, config, start=start)
    assert torch.equal(model.state_dict()["projection.weight"], torch.eye(DIM))


def test_the_mask_sharpens_geometrically_from_one_to_its_end() -> None:
    assert beta_at(0, 101, 0.01) == pytest.approx(1.0)
    assert beta_at(50, 101, 0.01) == pytest.approx(0.1)
    assert beta_at(100, 101, 0.01) == pytest.approx(0.01)


def test_the_budget_schedule_draws_every_10_percent_budget_reproducibly() -> None:
    schedule = budget_schedule(200, BUDGETS, seed=3)
    assert set(schedule) == set(BUDGETS) and 1024 not in schedule
    assert schedule == budget_schedule(200, BUDGETS, seed=3)
    assert schedule != budget_schedule(200, BUDGETS, seed=4)


@pytest.mark.parametrize("k", [PLANTED, 2 * PLANTED])
def test_one_multi_budget_selector_matches_a_selector_trained_at_each_budget(k: int) -> None:
    budgets = (PLANTED, 2 * PLANTED)
    multi, _ = train_retrieval_selector(
        TRAIN,
        VALIDATION,
        RetrievalConfig(**{**CONFIG.__dict__, "budgets": budgets, "eval_budgets": budgets}),
    )
    fixed, _ = train_retrieval_selector(
        TRAIN,
        VALIDATION,
        RetrievalConfig(**{**CONFIG.__dict__, "budgets": (k,), "eval_budgets": (k,)}),
    )
    assert rerank_ndcg10(multi, VALIDATION, (k,)) >= rerank_ndcg10(fixed, VALIDATION, (k,)) - 0.05
