"""Training the linear selector on masked retrieval (F5 research D2–D6).

The selector is the Learning-to-Select ``LinearSelector``; only the objective differs. Each step
masks a batch of training queries with a relaxed top-k of the selector's scores and minimizes
``masked_retrieval_loss`` against the queries' positives, hard negatives, and in-batch positives.
Every ``eval_every`` steps the hard-masked selector reranks each validation query's own documents,
and the checkpoint with the highest mean nDCG@10 over ``eval_budgets`` is kept.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from hyperdime.baselines.learning_to_select import LinearSelector
from hyperdime.selection.relaxed import soft_mass_on_topk, soft_top_k
from hyperdime.selection.topk import resolve_k, top_k_mask
from hyperdime.training.examples import RetrievalExamples
from hyperdime.training.losses import masked_retrieval_loss

# The 10%-90% budgets of the 1024 dimensions; 100% is the full query.
BUDGETS = tuple(resolve_k(step / 10, 1024) for step in range(1, 10))
HEADLINE_K = resolve_k(0.3, 1024)


@dataclass(frozen=True)
class RetrievalConfig:
    """``budgets`` are the budgets drawn during training, one per batch (one for a fixed-budget
    selector); ``eval_budgets`` those averaged when choosing the checkpoint."""

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
) -> tuple[LinearSelector, list[dict[str, float]]]:
    """Train for ``config.steps`` steps from ``start`` (or a seeded initialization); returns the
    best validation checkpoint on the CPU and one history row per evaluation (step 0 included)."""
    device = torch.device(config.device)
    torch.manual_seed(config.seed)
    model = LinearSelector(train.queries.shape[1])
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
    history = [_evaluate(model, validation, candidates, config, step=0, beta=1.0, loss=None)]
    best_score, best_state = history[0]["val_rerank_ndcg10"], copy.deepcopy(model.state_dict())
    total, count = 0.0, 0
    for step, (rows, k) in enumerate(zip(batches, schedule, strict=True), start=1):
        beta = beta_at(step - 1, config.steps, config.beta_end)
        model.train()
        loss = _batch_loss(model, train, rows, k, beta, config.score_temperature, device)
        optimizer.zero_grad()
        torch.autograd.backward(loss)
        optimizer.step()
        scheduler.step()
        total, count = total + loss.item(), count + 1
        if step % config.eval_every == 0 or step == config.steps:
            row = _evaluate(model, validation, candidates, config, step, beta, total / count)
            history.append(row)
            total, count = 0.0, 0
            if row["val_rerank_ndcg10"] > best_score:
                best_score, best_state = row["val_rerank_ndcg10"], copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model.cpu().eval(), history


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
    mask = soft_top_k(model(queries), k, beta)
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
        importance = model(queries)
        cand = candidates[rows]
        cand = cand[:, : int((cand >= 0).sum(1).max())]  # candidates are left-packed
        gains = _gains(examples, rows, cand).to(device)
        docs = examples.documents[cand.clamp(min=0)].float().to(device)
        cand = cand.to(device)
        for k in budgets:
            masked = queries * top_k_mask(importance, k).float()
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
) -> dict[str, float]:
    """One history row; the soft-mask mass uses at most 1,024 validation queries."""
    device = torch.device(config.device)
    ndcg = _rerank(model, validation, candidates, config.eval_budgets, device)
    with torch.no_grad():
        importance = model(validation.queries[:1024].float().to(device))
        mass = torch.stack(
            [
                soft_mass_on_topk(soft_top_k(importance, k, beta), importance, k).mean()
                for k in config.eval_budgets
            ]
        ).mean()
    row = {"step": float(step), "val_rerank_ndcg10": ndcg, "soft_mass": float(mass), "beta": beta}
    if loss is not None:
        row["train_loss"] = loss
    return row
