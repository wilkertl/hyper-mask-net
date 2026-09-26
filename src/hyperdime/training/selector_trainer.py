"""Learning-to-Select selector training with early stopping on validation KL."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import Tensor

from hyperdime.baselines.learning_to_select import LinearSelector
from hyperdime.training.losses import selector_kl_loss


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 0.01
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 5
    seed: int = 0


def train_selector(
    train_queries: Tensor,
    train_targets: Tensor,
    val_queries: Tensor,
    val_targets: Tensor,
    config: TrainConfig,
) -> tuple[LinearSelector, list[dict[str, float]]]:
    """Train on ``KL(target || prediction)``; return the best-validation model and the history."""
    if train_queries.shape != train_targets.shape or val_queries.shape != val_targets.shape:
        raise ValueError("queries and targets must share shape [N, D].")
    torch.manual_seed(config.seed)
    model = LinearSelector(train_queries.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    generator = torch.Generator().manual_seed(config.seed)
    best_loss, best_state, stale = float("inf"), copy.deepcopy(model.state_dict()), 0
    history: list[dict[str, float]] = []
    for epoch in range(config.max_epochs):
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
        model.eval()
        with torch.no_grad():
            val_loss = selector_kl_loss(val_targets, model(val_queries)).item()
        history.append({"epoch": epoch, "train_kl": total / len(order), "val_kl": val_loss})
        if val_loss < best_loss - 1e-6:
            best_loss, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    model.load_state_dict(best_state)
    return model.eval(), history
