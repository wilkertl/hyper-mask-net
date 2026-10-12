"""Importance methods for E0 (F3 research D3); top-k always goes through ``top_k_mask``.

Each method maps query embeddings ``(Q, D)`` to importance scores ``(Q, D)``; higher means keep.
"""

from __future__ import annotations

import torch
from torch import Tensor

from hyperdime.baselines.learning_to_select import LinearSelector


def prefix_importance(queries: Tensor) -> Tensor:
    """MRL prefix truncation: the first k dimensions (risk R5)."""
    count, dim = queries.shape
    return torch.arange(dim, 0, -1, dtype=torch.float32).expand(count, dim).clone()


def random_importance(queries: Tensor, seed: int = 0) -> Tensor:
    """Seeded uniform scores per query: a random k-subset that is reproducible for a seed."""
    return torch.rand(queries.shape, generator=torch.Generator().manual_seed(seed))


def global_importance(queries: Tensor, mean_target: Tensor) -> Tensor:
    """One ranking of dimensions for every query: the environment's mean training target."""
    return mean_target.float().expand(queries.shape[0], -1).clone()


def selector_importance(queries: Tensor, model: LinearSelector, batch: int = 8192) -> Tensor:
    """A trained Learning-to-Select selector's log-importance."""
    with torch.no_grad():
        return torch.cat([model(queries[i : i + batch]) for i in range(0, len(queries), batch)])


def oracle_scores(queries: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
    """Raw oracle ``e_q * (p - n)`` from the evaluated queries' own labels; its top-k does not
    depend on the temperature. Non-deployable: an upper bound, never a training input."""
    return queries.float() * (positive - negative)
