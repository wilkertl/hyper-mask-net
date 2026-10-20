"""Training the linear selector on masked retrieval (F5 research D4–D6)."""

import pytest
import torch
from torch.nn import functional as F

from hyperdime.baselines.learning_to_select import BudgetGeometricSelector, GeometricSelector
from hyperdime.training.examples import RetrievalExamples
from hyperdime.training.retrieval_trainer import (
    BUDGETS,
    RetrievalConfig,
    beta_at,
    budget_schedule,
    project_prior_weights,
    rerank_ndcg10,
    train_retrieval_candidates,
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


def test_padding_the_positive_lists_does_not_change_the_selector() -> None:
    config = RetrievalConfig(**{**CONFIG.__dict__, "steps": 20, "eval_every": 10})
    pad = lambda e: RetrievalExamples(  # noqa: E731
        e.queries, e.documents,
        torch.cat([e.positives, torch.full((len(e), 4), -1)], dim=1),
        torch.cat([e.weights, torch.zeros(len(e), 4)], dim=1),
        e.negatives, e.skipped,
    )  # fmt: skip
    a, history_a = train_retrieval_selector(TRAIN, VALIDATION, config)
    b, history_b = train_retrieval_selector(pad(TRAIN), pad(VALIDATION), config)
    assert all(torch.allclose(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())
    assert [h["val_rerank_ndcg10"] for h in history_a] == [
        h["val_rerank_ndcg10"] for h in history_b
    ]


def test_a_prior_selector_starts_as_the_variance_method_and_learns_past_a_bad_prior() -> None:
    # Negatives vary most on the irrelevant dims 8..31, so the prior first favors them; the
    # zero-started correction needs more steps and a larger rate to outweigh it.
    prior = TRAIN.documents.var(0, correction=0)
    config = RetrievalConfig(**{**CONFIG.__dict__, "steps": 1000, "lr": 5e-2})
    model, history = train_retrieval_selector(TRAIN, VALIDATION, config, prior=prior)
    assert history[0]["val_rerank_ndcg10"] == pytest.approx(
        rerank_ndcg10(GeometricSelector(prior), VALIDATION, (PLANTED,))
    )
    top = model(VALIDATION.queries).topk(PLANTED, dim=-1).indices
    assert ((top < PLANTED).sum(-1) >= PLANTED - 1).float().mean() >= 0.9


def test_a_learned_prior_weight_drops_on_a_misleading_prior_and_is_recorded() -> None:
    prior = TRAIN.documents.var(0, correction=0)
    config = RetrievalConfig(**{**CONFIG.__dict__, "steps": 300, "eval_every": 100, "lr": 5e-2})
    model, history = train_retrieval_selector(
        TRAIN, VALIDATION, config, prior=prior, learn_prior_weight=True
    )
    assert [h["prior_weight"] for h in history][0] == 1.0
    assert model.prior_weight.item() < 0.5


def test_unweighted_histories_carry_no_prior_weight() -> None:
    config = RetrievalConfig(**{**CONFIG.__dict__, "steps": 10, "eval_every": 10})
    _, history = train_retrieval_selector(
        TRAIN, VALIDATION, config, prior=TRAIN.documents.var(0, correction=0)
    )
    assert all("prior_weight" not in h for h in history)


MISLEADING = TRAIN.documents.var(0, correction=0)
# A large rate drives a prior weight below zero on the misleading prior within a few steps.
STEEP = RetrievalConfig(**{**CONFIG.__dict__, "steps": 60, "eval_every": 20, "lr": 0.5})


@pytest.mark.parametrize("per_budget", [False, True])
def test_projection_keeps_prior_weights_nonnegative_and_counts_its_steps(per_budget: bool) -> None:
    kwargs = {"prior": MISLEADING, "learn_prior_weight": not per_budget,
              "per_budget_prior": per_budget}  # fmt: skip
    _, free = train_retrieval_selector(TRAIN, VALIDATION, STEEP, **kwargs)  # type: ignore[arg-type]
    weights = lambda h: [v for key, v in h.items() if key.startswith("prior_weight")]  # noqa: E731
    assert min(min(weights(h)) for h in free) < 0
    projected = RetrievalConfig(**{**STEEP.__dict__, "nonnegative_prior": True})
    model, history = train_retrieval_selector(TRAIN, VALIDATION, projected, **kwargs)  # type: ignore[arg-type]
    assert all(min(weights(h)) >= 0 for h in history)
    assert history[0]["projected_steps"] == 0 and history[-1]["projected_steps"] > 0
    parameter = model.prior_weights if per_budget else model.prior_weight
    assert bool((parameter >= 0).all())


def test_projection_clamps_only_negative_prior_weights() -> None:
    model = BudgetGeometricSelector(torch.ones(4), budgets=(1, 2, 3))
    with torch.no_grad():
        model.prior_weights.copy_(torch.tensor([-0.5, 0.0, 0.7]))
    assert project_prior_weights(model)
    assert model.prior_weights.tolist() == [0.0, 0.0, pytest.approx(0.7)]
    assert not project_prior_weights(model)


def test_the_budget_selector_is_trained_on_each_batchs_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[int | None] = []
    forward = BudgetGeometricSelector.forward

    def spy(self: BudgetGeometricSelector, queries: torch.Tensor, k: int | None = None):  # type: ignore[no-untyped-def]
        if self.training:
            seen.append(k)
        return forward(self, queries, k)

    monkeypatch.setattr(BudgetGeometricSelector, "forward", spy)
    budgets = (PLANTED, 2 * PLANTED)
    config = RetrievalConfig(**{**CONFIG.__dict__, "budgets": budgets, "eval_budgets": budgets,
                                "steps": 12, "eval_every": 6})  # fmt: skip
    train_retrieval_selector(TRAIN, VALIDATION, config, prior=MISLEADING, per_budget_prior=True)
    assert seen == budget_schedule(12, budgets, config.seed)


def test_an_untrained_budget_selector_starts_as_the_variance_method() -> None:
    config = RetrievalConfig(**{**CONFIG.__dict__, "steps": 1, "eval_every": 1, "lr": 0.0})
    _, history = train_retrieval_selector(
        TRAIN, VALIDATION, config, prior=MISLEADING, per_budget_prior=True
    )
    assert history[0]["val_rerank_ndcg10"] == pytest.approx(
        rerank_ndcg10(GeometricSelector(MISLEADING), VALIDATION, (PLANTED,))
    )
    assert [key for key in history[0] if key.startswith("prior_weight")] == ["prior_weight_k8"]


def test_the_checkpoint_rule_is_rerank_or_validation() -> None:
    assert RetrievalConfig().checkpoint_rule == "rerank"
    with pytest.raises(ValueError, match="checkpoint_rule"):
        RetrievalConfig(checkpoint_rule="test")  # type: ignore[arg-type]


VALIDATION_RULE = RetrievalConfig(**{**CONFIG.__dict__, "steps": 30, "eval_every": 10,
                                     "checkpoint_rule": "validation"})  # fmt: skip


def test_the_validation_rule_returns_every_evaluated_state_as_a_candidate() -> None:
    _, history, candidates = train_retrieval_candidates(TRAIN, VALIDATION, VALIDATION_RULE)
    assert [step for step, _ in candidates] == [0, 10, 20, 30]
    assert [h["step"] for h in history] == [0, 10, 20, 30]
    assert all(v.device.type == "cpu" for _, state in candidates for v in state.values())


def test_the_rerank_rule_trains_identically_and_returns_no_candidates() -> None:
    model, history, _ = train_retrieval_candidates(TRAIN, VALIDATION, VALIDATION_RULE)
    rerank = RetrievalConfig(**{**VALIDATION_RULE.__dict__, "checkpoint_rule": "rerank"})
    same, same_history, none = train_retrieval_candidates(TRAIN, VALIDATION, rerank)
    assert none == [] and same_history == history
    assert all(torch.equal(model.state_dict()[k], same.state_dict()[k]) for k in model.state_dict())


def test_the_defaults_leave_histories_without_the_new_keys() -> None:
    config = RetrievalConfig(**{**CONFIG.__dict__, "steps": 10, "eval_every": 10})
    _, history = train_retrieval_selector(
        TRAIN, VALIDATION, config, prior=MISLEADING, learn_prior_weight=True
    )
    assert set(history[-1]) == {
        "step", "val_rerank_ndcg10", "soft_mass", "beta", "train_loss", "prior_weight"
    }  # fmt: skip
