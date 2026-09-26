"""Oracle targets with aggregated positives and hard negatives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from hyperdime.oracle.importance import oracle_importance


def aggregate_targets(
    queries: Tensor,
    documents: Tensor,
    positives: Sequence[Mapping[int, int]],
    negatives: Sequence[Sequence[int]],
    temperature: float,
) -> Tensor:
    """``softmax(e_q * (p - n) / tau)`` per query.

    ``p`` is the relevance-weighted mean of the positive document rows and ``n`` the mean of the
    hard-negative rows; ``positives[i]`` maps document row to graded relevance for query ``i``.
    """
    if not len(positives) == len(negatives) == queries.shape[0]:
        raise ValueError("positives and negatives must align with query rows.")
    aggregated_p = torch.empty_like(queries, dtype=torch.float32)
    aggregated_n = torch.empty_like(queries, dtype=torch.float32)
    for row, (labels, pool) in enumerate(zip(positives, negatives, strict=True)):
        weights = {doc: gain for doc, gain in labels.items() if gain > 0}
        if not weights or not pool:
            raise ValueError(f"query row {row} needs at least one positive and one negative.")
        rows = torch.tensor(list(weights))
        gains = torch.tensor(list(weights.values()), dtype=torch.float32)
        aggregated_p[row] = gains @ documents[rows].float() / gains.sum()
        aggregated_n[row] = documents[torch.tensor(list(pool))].float().mean(dim=0)
    return oracle_importance(queries.float(), aggregated_p, aggregated_n, temperature)
