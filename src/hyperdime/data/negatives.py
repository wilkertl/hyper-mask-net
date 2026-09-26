"""Hard-negative mining over frozen embeddings."""

from __future__ import annotations

from collections.abc import Collection, Sequence

import torch
from torch import Tensor

from hyperdime.retrieval.exact import exact_search


def mine_hard_negatives(
    query_embeddings: Tensor,
    document_embeddings: Tensor,
    document_ids: Sequence[str],
    positive_ids: Sequence[Collection[str]],
    pool_size: int,
    depth: int,
) -> list[list[str]]:
    """Return, per query, the ``pool_size`` highest-scoring documents not judged positive."""
    if len(document_ids) != document_embeddings.shape[0]:
        raise ValueError("document_ids must align with document_embeddings rows.")
    if len(positive_ids) != query_embeddings.shape[0]:
        raise ValueError("positive_ids must align with query_embeddings rows.")
    if not 1 <= pool_size < depth:
        raise ValueError("pool_size must be at least 1 and smaller than depth.")
    _, indices = exact_search(query_embeddings, document_embeddings, depth)
    pools: list[list[str]] = []
    for row, positives in zip(indices.tolist(), positive_ids, strict=True):
        pool = [document_ids[i] for i in row if document_ids[i] not in positives][:pool_size]
        if len(pool) < pool_size:
            raise ValueError(f"depth {depth} yields fewer than {pool_size} non-positive documents.")
        pools.append(pool)
    return pools


def sample_hard_negatives(
    query_embeddings: Tensor,
    document_embeddings: Tensor,
    document_ids: Sequence[str],
    positive_ids: Sequence[Collection[str]],
    count: int,
    depth: int,
    generator: torch.Generator,
) -> list[list[str]]:
    """Sample ``count`` documents per query, with replacement, from its top-``depth`` non-positives.

    Follows the official Learning-to-Select label generation: the pool is the full-embedding
    top-``depth`` minus judged positives; a pool smaller than ``count`` is kept whole and topped
    up with uniformly random documents, positives excluded.
    """
    if len(document_ids) != document_embeddings.shape[0]:
        raise ValueError("document_ids must align with document_embeddings rows.")
    if len(positive_ids) != query_embeddings.shape[0]:
        raise ValueError("positive_ids must align with query_embeddings rows.")
    if count < 1 or not 1 <= depth <= len(document_ids):
        raise ValueError("count must be positive and depth in [1, number of documents].")
    _, indices = exact_search(query_embeddings, document_embeddings, depth)
    pools: list[list[str]] = []
    for row, positives in zip(indices.tolist(), positive_ids, strict=True):
        pool = [document_ids[i] for i in row if document_ids[i] not in positives]
        if len(pool) >= count:
            picks = torch.randint(len(pool), (count,), generator=generator).tolist()
            pools.append([pool[i] for i in picks])
            continue
        others = [doc_id for doc_id in document_ids if doc_id not in positives]
        if not others:
            raise ValueError("every document is a judged positive; no negative can be sampled.")
        fill = torch.randint(len(others), (count - len(pool),), generator=generator).tolist()
        pools.append(pool + [others[i] for i in fill])
    return pools
