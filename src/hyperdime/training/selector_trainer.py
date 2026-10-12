"""Learning-to-Select selector training, following the official implementation's recipe."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from hyperdime.baselines.learning_to_select import LinearSelector
from hyperdime.training.losses import selector_kl_loss


@dataclass(frozen=True)
class TrainConfig:
    """Defaults match the official Learning-to-Select code: AdamW, cosine decay, fixed epochs.

    The ADR-0003 candidates add an L2 penalty toward an anchor selector (``l2_to_anchor``), a
    constant learning rate (``schedule``), and early stopping on validation KL (``patience``).
    """

    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 200
    seed: int = 0
    device: str = "cpu"
    schedule: Literal["cosine", "constant"] = "cosine"
    patience: int | None = None
    l2_to_anchor: float = 0.0


def train_selector(
    train_queries: Tensor,
    train_targets: Tensor,
    val_queries: Tensor,
    val_targets: Tensor,
    config: TrainConfig,
    anchor: Mapping[str, Tensor] | None = None,
) -> tuple[LinearSelector, list[dict[str, float]]]:
    """Train on ``KL(target || prediction)`` for ``config.epochs`` epochs.

    With the default cosine schedule the learning rate decays over the run; the returned model is
    the state with the lowest validation KL (no early stopping unless ``patience`` is set), as in
    the official implementation. The returned model is on the CPU.
    """
    if train_queries.shape != train_targets.shape or val_queries.shape != val_targets.shape:
        raise ValueError("queries and targets must share shape [N, D].")
    if config.l2_to_anchor and anchor is None:
        raise ValueError("l2_to_anchor needs an anchor selector state.")
    device = torch.device(config.device)
    torch.manual_seed(config.seed)
    model = LinearSelector(train_queries.shape[1]).to(device)
    train_queries, train_targets = train_queries.to(device), train_targets.to(device)
    val_queries, val_targets = val_queries.to(device), val_targets.to(device)
    fixed = {k: v.to(device) for k, v in (anchor or {}).items()}
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        if config.schedule == "cosine"
        else None
    )
    generator = torch.Generator().manual_seed(config.seed)
    best_loss, best_state = float("inf"), copy.deepcopy(model.state_dict())
    since_best = 0
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        model.train()
        order = torch.randperm(train_queries.shape[0], generator=generator).to(device)
        total = 0.0
        for start in range(0, len(order), config.batch_size):
            rows = order[start : start + config.batch_size]
            loss = selector_kl_loss(train_targets[rows], model(train_queries[rows]))
            if config.l2_to_anchor:
                loss = loss + config.l2_to_anchor * sum(
                    ((param - fixed[name]) ** 2).sum() for name, param in model.named_parameters()
                )
            optimizer.zero_grad()
            torch.autograd.backward(loss)
            optimizer.step()
            total += loss.item() * len(rows)
        if scheduler is not None:
            scheduler.step()
        model.eval()
        with torch.no_grad():
            val_loss = selector_kl_loss(val_targets, model(val_queries)).item()
        history.append({"epoch": epoch, "train_kl": total / len(order), "val_kl": val_loss})
        if val_loss < best_loss:
            best_loss, best_state = val_loss, copy.deepcopy(model.state_dict())
            since_best = 0
        else:
            since_best += 1
            if config.patience is not None and since_best >= config.patience:
                break
    model.load_state_dict(best_state)
    return model.cpu().eval(), history
