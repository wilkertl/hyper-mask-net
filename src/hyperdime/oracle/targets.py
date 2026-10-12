"""Oracle targets with aggregated positives and hard negatives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
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
    """``softmax(e_q * (p - n) / tau)`` per query, with ``p`` and ``n`` from ``aggregate_pairs``."""
    aggregated_p, aggregated_n = aggregate_pairs(queries, documents, positives, negatives)
    return oracle_importance(queries.float(), aggregated_p, aggregated_n, temperature)


def aggregate_pairs(
    queries: Tensor,
    documents: Tensor,
    positives: Sequence[Mapping[int, int]],
    negatives: Sequence[Sequence[int]],
) -> tuple[Tensor, Tensor]:
    """Aggregated positive ``p`` and negative ``n`` per query, each of shape ``[batch, D]``.

    ``p`` is the mean of the positive document rows weighted by the exponential gain
    ``2**rel - 1`` (as in the official Learning-to-Select code) and ``n`` the mean of the
    hard-negative rows, repeats included; ``positives[i]`` maps document row to graded relevance
    for query ``i``.
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
        gains = 2.0 ** torch.tensor(list(weights.values()), dtype=torch.float32) - 1.0
        aggregated_p[row] = gains @ documents[rows].float() / gains.sum()
        aggregated_n[row] = documents[torch.tensor(list(pool))].float().mean(dim=0)
    return aggregated_p, aggregated_n


def oracle_pairs(
    query_ids: Sequence[str],
    query_vectors: np.ndarray,
    query_rows: Mapping[str, int],
    documents: np.ndarray,
    doc_rows: Mapping[str, int],
    positives: Mapping[str, Mapping[str, int]],
    pools: Mapping[str, Sequence[str]],
) -> tuple[Tensor, Tensor, Tensor]:
    """``(queries, p, n)`` for ``query_ids``, gathering only the needed rows of the (possibly
    memory-mapped) document matrix. ``positives`` maps a query to its graded judgments (only
    grades > 0 are used) and ``pools`` to its sampled negatives (repeats kept)."""
    needed: set[int] = set()
    for qid in query_ids:
        needed.update(doc_rows[d] for d, g in positives[qid].items() if g > 0 and d in doc_rows)
        needed.update(doc_rows[d] for d in pools[qid])
    ordered = sorted(needed)
    local = {row: i for i, row in enumerate(ordered)}
    gathered = torch.from_numpy(np.asarray(documents[ordered], dtype=np.float32))
    queries = torch.from_numpy(
        np.asarray(query_vectors[[query_rows[qid] for qid in query_ids]], dtype=np.float32)
    )
    local_positives = [
        {local[doc_rows[d]]: g for d, g in positives[qid].items() if g > 0 and d in doc_rows}
        for qid in query_ids
    ]
    local_pools = [[local[doc_rows[d]] for d in pools[qid]] for qid in query_ids]
    p, n = aggregate_pairs(queries, gathered, local_positives, local_pools)
    return queries, p, n


def targets_at(queries: Tensor, p: Tensor, n: Tensor, temperature: float) -> Tensor:
    """``softmax(e_q * (p - n) / tau)``: one temperature over precomputed pairs (research R4)."""
    return oracle_importance(queries, p, n, temperature)


def uniform_kl(targets: Tensor) -> float:
    """Mean KL(target || uniform): the loss of a predictor that ignores the query (G1 (a))."""
    dim = targets.shape[1]
    safe = targets.clamp_min(1e-30)
    return float((targets * (safe * dim).log()).sum(dim=1).mean())
