"""Learning-to-Select selector training, following the official implementation's recipe."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import Tensor

from hyperdime.baselines.learning_to_select import LinearSelector
from hyperdime.training.losses import selector_kl_loss


@dataclass(frozen=True)
class TrainConfig:
    """Defaults match the official Learning-to-Select code: AdamW, cosine decay, fixed epochs."""

    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 200
    seed: int = 0


def train_selector(
    train_queries: Tensor,
    train_targets: Tensor,
    val_queries: Tensor,
    val_targets: Tensor,
    config: TrainConfig,
) -> tuple[LinearSelector, list[dict[str, float]]]:
    """Train on ``KL(target || prediction)`` for ``config.epochs`` epochs.

    The learning rate follows a cosine schedule over the run; the returned model is the state
    with the lowest validation KL (no early stopping), as in the official implementation.
    """
    if train_queries.shape != train_targets.shape or val_queries.shape != val_targets.shape:
        raise ValueError("queries and targets must share shape [N, D].")
    torch.manual_seed(config.seed)
    model = LinearSelector(train_queries.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    generator = torch.Generator().manual_seed(config.seed)
    best_loss, best_state = float("inf"), copy.deepcopy(model.state_dict())
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        model.train()
        order = torch.randperm(train_queries.shape[0], generator=generator)
        total = 0.0
        for start in range(0, len(order), config.batch_size):
            rows = order[start : start + config.batch_size]
            loss = selector_kl_loss(train_targets[rows], model(train_queries[rows]))
            optimizer.zero_grad()
            torch.autograd.backward(loss)
            optimizer.step()
            total += loss.item() * len(rows)
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_loss = selector_kl_loss(val_targets, model(val_queries)).item()
        history.append({"epoch": epoch, "train_kl": total / len(order), "val_kl": val_loss})
        if val_loss < best_loss:
            best_loss, best_state = val_loss, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model.eval(), history
