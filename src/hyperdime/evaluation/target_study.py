"""P3 target-encoding study (ADR-0019): mask math, target encodings, losses, and one trainer.

Everything here is exploratory/development. Masks are boolean ``[..., D]`` arrays; a budget's mask
keeps the top-k of a score with ties to the lower index (``top_k_dims``). The trainer reproduces the
authors' ``train_predictor`` loop exactly (AdamW, cosine schedule stepped per epoch, ``DataLoader``
shuffling, ``KLDivLoss(batchmean)`` on ``log_softmax``) when it is given their loss, so arms differ
from Learning2Select only in the loss they are given. ``CE`` is linear in its target, so a
per-budget cross-entropy without a budget input equals the averaged-target one; the per-budget
arms therefore use a nonlinear membership loss (E2) or a budget input (E3, with E3c as its
capacity control).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

ARMS = ("E0", "E1", "E2", "E3", "E3c", "E4")
LAMBDAS = (0.1, 0.3, 1.0)
BETA_START, BETA_END = 1.0, 0.1


# ----------------------------------------------------------------------------- mask math


def top_k_mask(scores: Tensor, k: int) -> Tensor:
    """Boolean top-k per row, ties to the lower index (stable sort), as ``top_k_dims``."""
    order = torch.sort(-scores, dim=-1, stable=True).indices[..., :k]
    mask = torch.zeros_like(scores, dtype=torch.bool)
    return mask.scatter(-1, order, True)


def jaccard(a: Tensor, b: Tensor) -> Tensor:
    """|a ∧ b| / |a ∨ b| over the last axis (1 when both are empty)."""
    inter = (a & b).sum(-1).to(torch.float64)
    union = (a | b).sum(-1).to(torch.float64)
    return torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(inter))


def mask_average(masks: Tensor) -> Tensor:
    """``t = (1/B) Σ_k m_k / k`` for masks ``[..., B, D]``: F12's teacher target."""
    sizes = masks.sum(-1, keepdim=True).to(torch.float64)
    return (masks.to(torch.float64) / sizes).mean(-2)


def kl_rows(target: Tensor, log_probs: Tensor) -> Tensor:
    """KL(target ‖ exp(log_probs)) per row, with 0 · log 0 = 0."""
    t = target.to(torch.float64)
    lp = log_probs.to(torch.float64)
    terms = torch.where(t > 0, t * (torch.log(t.clamp(min=1e-300)) - lp), torch.zeros_like(t))
    return terms.sum(-1)


def ranks_from_scores(scores: Tensor, positive_rows: Tensor) -> Tensor:
    """Ranks ``[P, B]`` of ``positive_rows`` for score columns ``[N, B]`` (lower-row ties)."""
    own = scores[positive_rows]  # [P, B]
    rows = torch.arange(scores.shape[0], device=scores.device)
    higher = (scores[None, :, :] > own[:, None, :]).sum(1)
    tied = (
        (scores[None, :, :] == own[:, None, :])
        & (rows[None, :, None] < positive_rows[:, None, None])
    ).sum(1)
    return 1 + higher + tied


def margins(scores: Tensor, positive_rows: Tensor) -> Tensor:
    """Best positive score minus best non-positive score, per score column ``[N, B]``."""
    is_pos = torch.zeros(scores.shape[0], dtype=torch.bool, device=scores.device)
    is_pos[positive_rows] = True
    best_pos = scores[is_pos].max(0).values
    best_neg = scores[~is_pos].max(0).values
    return best_pos - best_neg


def nesting(masks: Tensor) -> Tensor:
    """Per consecutive budget pair, share of the smaller mask inside the larger: ``[..., B-1]``."""
    small, large = masks[..., :-1, :], masks[..., 1:, :]
    return (small & large).sum(-1).to(torch.float64) / small.sum(-1).clamp(min=1)


# ----------------------------------------------------------------------------- model and losses


class Selector(nn.Module):
    """``z = W q + b`` (the authors' ``DimensionPredictor`` layout), with an optional per-budget
    offset ``c_k`` (E3, E3c), zero-initialized. Created after seeding; the offset uses no RNG."""

    def __init__(self, dim: int, budgets: int = 0) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim))
        self.offsets = nn.Parameter(torch.zeros(budgets, dim)) if budgets else None

    def logits(self, x: Tensor, budget: int | None = None) -> Tensor:
        z: Tensor = self.net(x)
        if self.offsets is not None and budget is not None:
            z = z + self.offsets[budget]
        return z

    def forward(self, x: Tensor) -> Tensor:
        return F.log_softmax(self.logits(x), dim=-1)


_KL = nn.KLDivLoss(reduction="batchmean")


def loss_kl(model: Selector, x: Tensor, target: Tensor) -> Tensor:
    """The authors' loss: ``KLDivLoss(batchmean)(log_softmax(z), target)``."""
    return _KL(model(x), target)  # type: ignore[no-any-return]


def loss_membership(
    model: Selector, x: Tensor, masks: Tensor, beta: float, sizes: Sequence[int]
) -> Tensor:
    """E2: mean over budgets of BCE(σ((ẑ − θ_k)/β), m_k), ẑ standardized per row, θ_k midway
    between the k-th and (k+1)-th of ẑ without gradient."""
    z = model.logits(x)
    zs = (z - z.mean(-1, keepdim=True)) / (z.std(-1, unbiased=False, keepdim=True) + 1e-6)
    ordered = torch.sort(zs.detach(), dim=-1, descending=True).values
    losses = []
    for b, k in enumerate(sizes):
        theta = (ordered[:, k - 1 : k] + ordered[:, k : k + 1]) / 2
        soft = torch.sigmoid((zs - theta) / beta)
        losses.append(F.binary_cross_entropy(soft, masks[:, b].to(soft.dtype)))
    return torch.stack(losses).mean()


def loss_budget_input(model: Selector, x: Tensor, targets: Tensor) -> Tensor:
    """E3/E3c: mean over budgets of KL(target_k ‖ softmax(z + c_k)), ``targets`` ``[n, B, D]``."""
    losses = [
        _KL(F.log_softmax(model.logits(x, b), dim=-1), targets[:, b])
        for b in range(targets.shape[1])
    ]
    return torch.stack(losses).mean()


def loss_hybrid(
    model: Selector, x: Tensor, official: Tensor, average: Tensor, lam: float
) -> Tensor:
    """E4: KL(t_off ‖ p) + λ KL(t̄ ‖ p)."""
    log_probs = model(x)
    return _KL(log_probs, official) + lam * _KL(log_probs, average)  # type: ignore[no-any-return]


# ----------------------------------------------------------------------------- trainer


@dataclass
class TrainResult:
    model: Selector
    final_loss: float
    steps: int


def train(
    x: Tensor,
    payload: Tensor,
    loss: Callable[[Selector, Tensor, Tensor, float], Tensor],
    *,
    epochs: int,
    seed: int,
    device: str,
    budgets: int = 0,
    lr: float = 1e-4,
    batch_size: int = 32,
    set_seed: Callable[[int], None] | None = None,
    start: dict[str, Tensor] | None = None,
) -> TrainResult:
    """The authors' ``train_predictor`` loop with ``loss(model, x_batch, payload_batch, beta)``.

    ``set_seed`` is the authors' ``set_random_seed`` when available (identical stream); the model
    is created after seeding and before the loader is iterated, as in their code. β anneals
    geometrically from 1 to 0.1 over all optimizer steps (used by relaxed-mask losses only).
    ``start`` loads a state into the model after it is created; loading draws no random numbers,
    so the batch order equals a seeded run's.
    """
    (set_seed or _seed)(seed)
    dataset = torch.utils.data.TensorDataset(x, payload)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = Selector(x.shape[1], budgets)
    if start is not None:
        result = model.load_state_dict(start, strict=False)
        if result.unexpected_keys or set(result.missing_keys) - {"offsets"}:
            raise ValueError(f"start state does not fit the selector: {result}")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    total = epochs * max(len(loader), 1)
    step, last = 0, 0.0
    for _ in range(epochs):
        model.train()
        running = 0.0
        for xb, pb in loader:
            xb, pb = xb.to(device), pb.to(device)
            beta = BETA_START * (BETA_END / BETA_START) ** (step / max(total - 1, 1))
            optimizer.zero_grad()
            value = loss(model, xb, pb, beta)
            torch.autograd.backward(value)
            optimizer.step()
            running += value.item()
            step += 1
        scheduler.step()
        last = running / max(len(loader), 1)
    return TrainResult(model.cpu(), last, step)


def _seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
