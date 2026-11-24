"""P3 objective study (ADR-0020): target versus loss versus direct ranking (exploratory).

Every arm trains the same ``Selector`` (``z = W q + b``; KBk/KBa add per-budget offsets) with
``target_study.train`` and differs only in its loss and its target encoding:

- KL arms: ``KL(t ‖ softmax z)`` on the graded L2S target ``t_off`` (KG, KGI) or the rank-step
  average ``t̄`` of the nested contrastive masks (KS); with budget offsets on per-budget ``m_k/k``
  (KBk) or on ``t̄`` for every budget (KBa).
- Membership arms: mean over budgets of ``BCE(relaxed_k(z), label_k)``, where ``relaxed_k`` is the
  standardized-score sigmoid around the k-th/(k+1)-th midpoint (ADR-0019 E2). Labels are binary
  masks (MS) or the same relaxation of the teacher's own log-target at the step's β (MG). Labels
  are per-dimension memberships in [0, 1]; a normalized distribution is never given to BCE.
- Ranking arms (R0, RI): mean over budgets of ``masked_retrieval_loss`` with the relaxed mask.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from hyperdime.evaluation.benchmark_stats import bootstrap_means, holm, sign_flip_p
from hyperdime.evaluation.target_study import Selector, loss_budget_input, loss_kl
from hyperdime.training.examples import RetrievalExamples
from hyperdime.training.losses import masked_retrieval_loss
from hyperdime.training.retrieval_trainer import _gather

LEARNING_RATES = (3e-5, 1e-4, 3e-4, 1e-3)
POST_HOC_RATES = (3e-3, 1e-2)  # ADR-0020 amendment A1 (post hoc, every arm)
OFFICIAL_LR = 1e-4
SCORE_TEMPERATURE = 0.05
SEEDS = (42, 0, 1)
TRAINED = ("KG", "KS", "MS", "MG", "KBk", "KBa", "R0", "RI", "KGI")
INITIALIZED = ("RI", "KGI")
BUDGET_INPUT = ("KBk", "KBa")
CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("C1", "KS", "KG"),
    ("C2", "MS", "MG"),
    ("C3", "MS", "KS"),
    ("C4", "MG", "KG"),
    ("C6", "KBk", "KBa"),
    ("C7", "KBa", "KS"),
    ("C8", "R0", "KG"),
    ("C9", "RI", "KGI"),
    ("C10", "RI", "R0"),
    ("C11", "KG", "REF"),
)
MEANING = {
    "C1": "target given KL: rank-step vs graded",
    "C2": "target given membership: binary vs graded",
    "C3": "loss given the same information: membership vs KL",
    "C4": "loss given graded targets: membership vs KL",
    "C6": "per-budget vs aggregated, KL with offsets",
    "C7": "budget-offset capacity, t̄ under KL",
    "C8": "direct ranking vs tuned L2S",
    "C9": "objective given L2S init: ranking vs KL",
    "C10": "init given ranking: L2S vs seeded",
    "C11": "lr tuning of L2S",
}
Loss = Callable[[Selector, Tensor, Tensor, float], Tensor]


# ----------------------------------------------------------------------------- encodings


def relaxed_membership(scores: Tensor, sizes: Sequence[int], beta: float) -> Tensor:
    """``[n, B, D]``: ``σ((ẑ − θ_k)/β)`` with ``ẑ`` standardized per row and ``θ_k`` midway between
    the k-th and (k+1)-th largest ``ẑ`` (no gradient), as ADR-0019's E2 loss."""
    z = (scores - scores.mean(-1, keepdim=True)) / (
        scores.std(-1, unbiased=False, keepdim=True) + 1e-6
    )
    ordered = torch.sort(z.detach(), dim=-1, descending=True).values
    out = []
    for k in sizes:
        theta = (ordered[:, k - 1 : k] + ordered[:, k : k + 1]) / 2
        out.append(torch.sigmoid((z - theta) / beta))
    return torch.stack(out, 1)


def membership_loss(
    model: Selector, x: Tensor, labels: Tensor, beta: float, sizes: Sequence[int]
) -> Tensor:
    """Mean over budgets of ``BCE(relaxed_k(z), labels[:, k])``; labels in [0, 1], ``[n, B, D]``."""
    soft = relaxed_membership(model.logits(x), sizes, beta)
    labels = labels.to(soft.dtype)
    return torch.stack(
        [F.binary_cross_entropy(soft[:, b], labels[:, b]) for b in range(len(sizes))]
    ).mean()


def graded_labels(log_target: Tensor, sizes: Sequence[int], beta: float) -> Tensor:
    """MG's labels: the teacher's own log-target under the student's relaxation at ``beta``."""
    with torch.no_grad():
        return relaxed_membership(log_target, sizes, beta)


def ranking_loss(
    model: Selector,
    rows: Tensor,
    examples: RetrievalExamples,
    sizes: Sequence[int],
    beta: float,
    temperature: float = SCORE_TEMPERATURE,
) -> Tensor:
    """Mean over budgets of ``masked_retrieval_loss`` of the relaxed mask (raw queries, as S1)."""
    device = next(model.parameters()).device
    queries, documents, positives, weights, negatives = _gather(examples, rows.cpu(), device)
    soft = relaxed_membership(model.logits(queries), sizes, beta)
    return torch.stack(
        [
            masked_retrieval_loss(
                queries, soft[:, b], documents, positives, weights, negatives, temperature
            )
            for b in range(len(sizes))
        ]
    ).mean()


@dataclass(frozen=True)
class ArmData:
    """Fit-query tensors in one row order: raw queries ``x``, ``t_off`` ``[n, D]``, nested masks
    ``[n, B, D]``, and the ranking examples (row i is fit query i)."""

    x: Tensor
    official: Tensor
    masks: Tensor
    examples: RetrievalExamples
    sizes: tuple[int, ...]

    @property
    def average(self) -> Tensor:
        sizes = self.masks.sum(-1, keepdim=True).to(torch.float64)
        return (self.masks.to(torch.float64) / sizes).mean(-2).to(torch.float32)


def arm_payload(arm: str, data: ArmData) -> tuple[Tensor, Loss, int]:
    """(payload indexed like ``x``, loss, number of budget offsets) for one ADR-0020 arm."""
    n, b, d = data.masks.shape
    sizes = data.sizes
    if arm in ("KG", "KGI"):
        return data.official, lambda m, x, t, _b: loss_kl(m, x, t), 0
    if arm == "KS":
        return data.average, lambda m, x, t, _b: loss_kl(m, x, t), 0
    if arm == "MS":
        return data.masks, lambda m, x, t, beta: membership_loss(m, x, t, beta, sizes), 0
    if arm == "MG":
        log_target = torch.log(data.official.to(torch.float64)).to(torch.float32)
        return (
            log_target,
            lambda m, x, t, beta: membership_loss(m, x, graded_labels(t, sizes, beta), beta, sizes),
            0,
        )
    if arm == "KBk":
        per_budget = data.masks.to(torch.float32)
        per_budget = per_budget / per_budget.sum(-1, keepdim=True)
        return per_budget, lambda m, x, t, _b: loss_budget_input(m, x, t), b
    if arm == "KBa":
        repeated = data.average[:, None, :].expand(n, b, d).contiguous()
        return repeated, lambda m, x, t, _b: loss_budget_input(m, x, t), b
    if arm in ("R0", "RI"):
        index = torch.arange(n)
        examples = data.examples
        return (
            index,
            lambda m, _x, rows, beta: ranking_loss(m, rows, examples, sizes, beta),
            0,
        )
    raise ValueError(f"unknown arm {arm}")


# ----------------------------------------------------------------------------- selection


def select_lr(tune_a: Mapping[float, float]) -> float:
    """The learning rate with the best tune-A score; ties go to the smaller rate."""
    return min(tune_a, key=lambda lr: (-tune_a[lr], lr))


# ----------------------------------------------------------------------------- statistics


def two_sided_p(diffs: np.ndarray, rng: np.random.Generator, draws: int) -> float:
    """Two-sided sign-flip p-value: twice the smaller one-sided p, capped at 1."""
    upper = sign_flip_p(diffs, rng, draws)
    lower = sign_flip_p(-diffs, rng, draws)
    return float(min(1.0, 2 * min(upper, lower)))


def contrast(
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
    family: int,
    draws: int = 10_000,
) -> dict[str, Any]:
    """Paired contrast of per-seed, per-query values ``[S, Q]`` (same seeds, same queries).

    ``D`` is the mean over queries of the seed-mean difference. The intervals are paired bootstrap
    percentiles over queries: pointwise 95% and Bonferroni at ``0.05/family``. ``spread`` is the
    range over seeds of the same-seed mean difference."""
    if a.shape != b.shape:
        raise ValueError("contrast needs arrays of one shape [seeds, queries]")
    diffs = (a - b).mean(0)
    boot = bootstrap_means(diffs, rng, draws)
    alpha = 0.05 / family
    low, high = np.quantile(boot, [0.025, 0.975])
    blow, bhigh = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    per_seed = (a - b).mean(1)
    return {
        "D": float(diffs.mean()),
        "low": float(low),
        "high": float(high),
        "bonferroni_low": float(blow),
        "bonferroni_high": float(bhigh),
        "p": two_sided_p(diffs, rng, draws),
        "spread": float(per_seed.max() - per_seed.min()),
        "per_seed": [float(v) for v in per_seed],
    }


def family_holm(results: dict[str, dict[str, Any]]) -> None:
    """Add Holm-adjusted p-values over the listed contrasts (in place)."""
    adjusted = holm({name: r["p"] for name, r in results.items()})
    for name, r in results.items():
        r["holm_p"] = adjusted[name]


def reading(per_corpus: Mapping[str, Mapping[str, Any]]) -> str:
    """ADR-0020 rule: supported (same sign on every corpus, |D| above the seed spread on every
    corpus, Bonferroni interval excluding zero on at least one), mixed (signs differ), else not
    separated."""
    values = list(per_corpus.values())
    signs = {np.sign(v["D"]) for v in values}
    if len(signs) > 1:
        return "mixed"
    above = all(abs(v["D"]) > v["spread"] for v in values)
    excludes = any(v["bonferroni_low"] > 0 or v["bonferroni_high"] < 0 for v in values)
    return "supported" if above and excludes and 0 not in signs else "not separated"


def parameters(arm: str, dim: int = 1024, budgets: int = 9) -> int:
    """Trainable parameters of an arm's selector."""
    return dim * dim + dim + (budgets * dim if arm in BUDGET_INPUT else 0)
