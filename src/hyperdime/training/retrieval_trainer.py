"""Training the linear selector on masked retrieval (F5 research D2–D6).

The selector is the Learning-to-Select ``LinearSelector``; only the objective differs. Each step
masks a batch of training queries with a relaxed top-k of the selector's scores and minimizes
``masked_retrieval_loss`` against the queries' positives, hard negatives, and in-batch positives.
Every ``eval_every`` steps the hard-masked selector reranks each validation query's own documents,
and the checkpoint with the highest mean nDCG@10 over ``eval_budgets`` is kept. Under the aligned
``checkpoint_rule="validation"`` (F9 research D3) every evaluated state is also returned as a
candidate, for ``retrieval_selection`` to score on the full corpus.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from hyperdime.baselines.learning_to_select import (
    BudgetGeometricSelector,
    GeometricSelector,
    LinearSelector,
)
from hyperdime.selection.relaxed import soft_mass_on_topk, soft_top_k
from hyperdime.selection.topk import resolve_k, top_k_mask
from hyperdime.training.examples import RetrievalExamples
from hyperdime.training.losses import masked_retrieval_loss

# The 10%-90% budgets of the 1024 dimensions; 100% is the full query.
BUDGETS = tuple(resolve_k(step / 10, 1024) for step in range(1, 10))
HEADLINE_K = resolve_k(0.3, 1024)
CHECKPOINT_RULES = ("rerank", "validation")
Candidates = list[tuple[int, dict[str, Tensor]]]


@dataclass(frozen=True)
class RetrievalConfig:
    """``budgets`` are the budgets drawn during training, one per batch (one for a fixed-budget
    selector); ``eval_budgets`` those averaged when choosing the checkpoint.

    ``checkpoint_rule`` ``rerank`` keeps the state with the best validation rerank score (F5–F8);
    ``validation`` also returns every evaluated state as a candidate (F9 research D3).
    ``nonnegative_prior`` projects prior weights to ``>= 0`` after each step (research D2).
    """

    budgets: tuple[int, ...] = (HEADLINE_K,)
    eval_budgets: tuple[int, ...] = BUDGETS
    lr: float = 1e-3
    score_temperature: float = 0.05
    beta_end: float = 0.1
    steps: int = 20_000
    batch_size: int = 32
    weight_decay: float = 1e-4
    eval_every: int = 1_000
    seed: int = 0
    device: str = "cpu"
    checkpoint_rule: Literal["rerank", "validation"] = "rerank"
    nonnegative_prior: bool = False

    def __post_init__(self) -> None:
        if self.checkpoint_rule not in CHECKPOINT_RULES:
            raise ValueError(f"checkpoint_rule must be one of {CHECKPOINT_RULES}")


def beta_at(step: int, steps: int, beta_end: float) -> float:
    """Mask temperature at ``step``: geometric from 1 at the first step to ``beta_end``."""
    return float(beta_end ** (step / max(steps - 1, 1)))


def budget_schedule(steps: int, budgets: Sequence[int], seed: int) -> list[int]:
    """One budget per step, drawn uniformly from ``budgets`` by a generator seeded with ``seed``."""
    draws = torch.randint(len(budgets), (steps,), generator=torch.Generator().manual_seed(seed))
    return [budgets[i] for i in draws.tolist()]


def train_retrieval_selector(
    train: RetrievalExamples,
    validation: RetrievalExamples,
    config: RetrievalConfig,
    start: Mapping[str, Tensor] | None = None,
    prior: Tensor | None = None,
    learn_prior_weight: bool = False,
    per_budget_prior: bool = False,
) -> tuple[LinearSelector, list[dict[str, float]]]:
    """Train for ``config.steps`` steps from ``start`` (or a seeded initialization; with a corpus
    ``prior`` variance, a ``GeometricSelector`` whose linear map starts at zero, with
    ``learn_prior_weight`` a learned scale on that prior, and with ``per_budget_prior`` a
    ``BudgetGeometricSelector`` with one scale per training and evaluation budget); returns the
    best validation rerank checkpoint on the CPU and one history row per evaluation (step 0
    included)."""
    model, history, _ = train_retrieval_candidates(
        train, validation, config, start, prior, learn_prior_weight, per_budget_prior
    )
    return model, history


def train_retrieval_candidates(
    train: RetrievalExamples,
    validation: RetrievalExamples,
    config: RetrievalConfig,
    start: Mapping[str, Tensor] | None = None,
    prior: Tensor | None = None,
    learn_prior_weight: bool = False,
    per_budget_prior: bool = False,
) -> tuple[LinearSelector, list[dict[str, float]], Candidates]:
    """As ``train_retrieval_selector``, with the candidates of the ``validation`` checkpoint rule:
    ``(step, state)`` on the CPU at every evaluation, step 0 included (empty under ``rerank``)."""
    device = torch.device(config.device)
    torch.manual_seed(config.seed)
    model: LinearSelector
    if prior is None:
        model = LinearSelector(train.queries.shape[1])
    elif per_budget_prior:
        model = BudgetGeometricSelector(prior, sorted({*config.budgets, *config.eval_budgets}))
    else:
        model = GeometricSelector(prior, learn_weight=learn_prior_weight)
    if start is not None:
        model.load_state_dict(start)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.steps)
    schedule = budget_schedule(config.steps, config.budgets, config.seed)
    batches = _batches(len(train), config.batch_size, config.steps, config.seed)
    candidates = _rerank_candidates(validation)
    projected = 0
    history = [
        _evaluate(model, validation, candidates, config, step=0, beta=1.0, loss=None,
                  projected=projected)
    ]  # fmt: skip
    best_score, best_state = history[0]["val_rerank_ndcg10"], copy.deepcopy(model.state_dict())
    kept: Candidates = []
    if config.checkpoint_rule == "validation":
        kept.append((0, _cpu_state(model)))
    total, count = 0.0, 0
    for step, (rows, k) in enumerate(zip(batches, schedule, strict=True), start=1):
        beta = beta_at(step - 1, config.steps, config.beta_end)
        model.train()
        loss = _batch_loss(model, train, rows, k, beta, config.score_temperature, device)
        optimizer.zero_grad()
        torch.autograd.backward(loss)
        optimizer.step()
        scheduler.step()
        if config.nonnegative_prior and project_prior_weights(model):
            projected += 1
        total, count = total + loss.item(), count + 1
        if step % config.eval_every == 0 or step == config.steps:
            row = _evaluate(
                model, validation, candidates, config, step, beta, total / count, projected
            )
            history.append(row)
            total, count = 0.0, 0
            if config.checkpoint_rule == "validation":
                kept.append((step, _cpu_state(model)))
            if row["val_rerank_ndcg10"] > best_score:
                best_score, best_state = row["val_rerank_ndcg10"], copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model.cpu().eval(), history, kept


def project_prior_weights(model: LinearSelector) -> bool:
    """Clamp the prior weights (one, or one per budget) to ``>= 0`` (research D2); returns
    whether any weight changed."""
    changed = False
    with torch.no_grad():
        for name, weight in model.named_parameters():
            if name in ("prior_weight", "prior_weights") and bool((weight < 0).any()):
                weight.clamp_(min=0.0)
                changed = True
    return changed


def scores_at(model: LinearSelector, queries: Tensor, k: int) -> Tensor:
    """The selector's log-importance at budget ``k``; only a budget selector depends on it."""
    found: Tensor = (
        model(queries, k) if isinstance(model, BudgetGeometricSelector) else model(queries)
    )
    return found


def _cpu_state(model: LinearSelector) -> dict[str, Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def rerank_ndcg10(
    model: LinearSelector, examples: RetrievalExamples, budgets: Sequence[int]
) -> float:
    """Mean over ``budgets`` and queries of nDCG@10 when the hard-masked query reranks its own
    positives and distinct negatives (gains proportional to ``2**grade - 1``)."""
    return _rerank(model, examples, _rerank_candidates(examples), budgets, torch.device("cpu"))


def _batches(n: int, size: int, steps: int, seed: int) -> list[Tensor]:
    generator = torch.Generator().manual_seed(seed)
    batches: list[Tensor] = []
    while len(batches) < steps:
        order = torch.randperm(n, generator=generator)
        batches += [order[i : i + size] for i in range(0, n - size + 1, size)] or [order]
    return batches[:steps]


def _gather(examples: RetrievalExamples, rows: Tensor, device: torch.device) -> tuple[Tensor, ...]:
    """The batch's queries, its distinct documents, and its indices remapped into them; positives
    are trimmed to the batch's longest list (they are left-packed)."""
    width = int((examples.positives[rows] >= 0).sum(1).max())
    positives, negatives = examples.positives[rows, :width], examples.negatives[rows]
    wanted = torch.cat([positives.flatten(), negatives.flatten()]).clamp(min=0)
    unique, inverse = torch.unique(wanted, return_inverse=True)
    local_pos = inverse[: positives.numel()].view_as(positives).masked_fill(positives < 0, -1)
    local_neg = inverse[positives.numel() :].view_as(negatives)
    documents = examples.documents[unique].float()
    return (
        examples.queries[rows].float().to(device),
        documents.to(device),
        local_pos.to(device),
        examples.weights[rows, :width].to(device),
        local_neg.to(device),
    )


def _batch_loss(
    model: LinearSelector,
    examples: RetrievalExamples,
    rows: Tensor,
    k: int,
    beta: float,
    score_temperature: float,
    device: torch.device,
) -> Tensor:
    queries, documents, positives, weights, negatives = _gather(examples, rows, device)
    mask = soft_top_k(scores_at(model, queries, k), k, beta)
    return masked_retrieval_loss(
        queries, mask, documents, positives, weights, negatives, score_temperature
    )


def _rerank_candidates(examples: RetrievalExamples) -> Tensor:
    """Per query, its positives then its distinct negatives, padded with -1."""
    rows = []
    for positives, negatives in zip(examples.positives, examples.negatives, strict=True):
        own = positives[positives >= 0].tolist()
        rows.append(own + [d for d in dict.fromkeys(negatives.tolist()) if d not in own])
    width = max(len(r) for r in rows)
    return torch.tensor([r + [-1] * (width - len(r)) for r in rows])


@torch.no_grad()
def _rerank(
    model: LinearSelector,
    examples: RetrievalExamples,
    candidates: Tensor,
    budgets: Sequence[int],
    device: torch.device,
    batch: int = 128,
) -> float:
    model.eval()
    scores = []
    for start in range(0, len(examples), batch):
        rows = torch.arange(start, min(start + batch, len(examples)))
        queries = examples.queries[rows].float().to(device)
        per_budget = isinstance(model, BudgetGeometricSelector)
        importance = None if per_budget else model(queries)
        cand = candidates[rows]
        cand = cand[:, : int((cand >= 0).sum(1).max())]  # candidates are left-packed
        gains = _gains(examples, rows, cand).to(device)
        docs = examples.documents[cand.clamp(min=0)].float().to(device)
        cand = cand.to(device)
        for k in budgets:
            scores_k = model(queries, k) if importance is None else importance
            masked = queries * top_k_mask(scores_k, k).float()
            logits = torch.einsum("bd,bcd->bc", masked, docs).masked_fill(cand < 0, float("-inf"))
            scores.append(_ndcg10(logits, gains))
    return float(torch.cat(scores).mean())


def _gains(examples: RetrievalExamples, rows: Tensor, candidates: Tensor) -> Tensor:
    """Positives come first among the candidates, so their weights are the leading gains."""
    width = min(examples.positives.shape[1], candidates.shape[1])
    gains = torch.zeros(candidates.shape, dtype=torch.float32)
    gains[:, :width] = examples.weights[rows, :width]
    return gains


def _ndcg10(logits: Tensor, gains: Tensor) -> Tensor:
    discounts = 1.0 / torch.log2(torch.arange(2, 12, device=logits.device, dtype=torch.float32))
    top = logits.topk(min(10, logits.shape[1]), dim=-1).indices
    dcg = (gains.gather(1, top) * discounts[: top.shape[1]]).sum(-1)
    ideal = (
        gains.sort(-1, descending=True).values[:, : top.shape[1]] * discounts[: top.shape[1]]
    ).sum(-1)
    return dcg / ideal


def _evaluate(
    model: LinearSelector,
    validation: RetrievalExamples,
    candidates: Tensor,
    config: RetrievalConfig,
    step: int,
    beta: float,
    loss: float | None,
    projected: int = 0,
) -> dict[str, float]:
    """One history row; the soft-mask mass uses at most 1,024 validation queries."""
    device = torch.device(config.device)
    ndcg = _rerank(model, validation, candidates, config.eval_budgets, device)
    mass = soft_mass(model, validation.queries, config.eval_budgets, beta, device)
    row = {"step": float(step), "val_rerank_ndcg10": ndcg, "soft_mass": mass, "beta": beta}
    if loss is not None:
        row["train_loss"] = loss
    weight = getattr(model, "prior_weight", None)
    if weight is not None:
        row["prior_weight"] = float(weight.detach())
    if isinstance(model, BudgetGeometricSelector):
        for k, value in zip(model.budgets.tolist(), model.prior_weights.tolist(), strict=True):
            row[f"prior_weight_k{k}"] = value
    if config.nonnegative_prior:
        row["projected_steps"] = float(projected)
    return row


@torch.no_grad()
def soft_mass(
    model: LinearSelector,
    queries: Tensor,
    budgets: Sequence[int],
    beta: float,
    device: torch.device | str = "cpu",
) -> float:
    """Mean over ``budgets`` of the relaxed mask's mass on the hard top-k at temperature ``beta``,
    on at most 1,024 ``queries`` (the guard of F5 research D3)."""
    queries = queries[:1024].float().to(device)
    if isinstance(model, BudgetGeometricSelector):
        found = [scores_at(model, queries, k) for k in budgets]
    else:
        found = [model(queries)] * len(budgets)
    return float(
        torch.stack(
            [
                soft_mass_on_topk(soft_top_k(importance, k, beta), importance, k).mean()
                for importance, k in zip(found, budgets, strict=True)
            ]
        ).mean()
    )
