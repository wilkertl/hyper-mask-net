"""Hard-negative mining over frozen embeddings."""

from __future__ import annotations

from collections.abc import Collection, Sequence

import numpy as np
import torch
from torch import Tensor

from hyperdime.contracts.schemas import NegativePool
from hyperdime.data.prepare import DatasetBundle
from hyperdime.retrieval.exact import exact_search, exact_search_streamed


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
    return sample_from_rankings(indices, document_ids, positive_ids, count, generator)


def sample_from_rankings(
    rankings: Tensor,
    document_ids: Sequence[str],
    excluded: Sequence[Collection[str]],
    count: int,
    generator: torch.Generator,
) -> list[list[str]]:
    """Sample ``count`` ids per query, with replacement, from its ranked rows minus ``excluded``.

    ``rankings`` holds each query's top-``depth`` document rows (best first). A pool smaller than
    ``count`` is kept whole and topped up with uniformly random non-excluded documents.
    """
    if len(excluded) != rankings.shape[0]:
        raise ValueError("excluded must align with rankings rows.")
    pools: list[list[str]] = []
    for row, skip in zip(rankings.tolist(), excluded, strict=True):
        pool = [document_ids[i] for i in row if document_ids[i] not in skip]
        if len(pool) >= count:
            picks = torch.randint(len(pool), (count,), generator=generator).tolist()
            pools.append([pool[i] for i in picks])
            continue
        others = [doc_id for doc_id in document_ids if doc_id not in skip]
        if not others:
            raise ValueError("every document is a judged positive; no negative can be sampled.")
        fill = torch.randint(len(others), (count - len(pool),), generator=generator).tolist()
        pools.append(pool + [others[i] for i in fill])
    return pools


def build_negative_pools(
    bundle: DatasetBundle,
    query_vectors: np.ndarray,
    query_ids: Sequence[str],
    documents: np.ndarray,
    document_ids: Sequence[str],
    *,
    depth: int = 1000,
    count: int = 64,
    seed: int = 0,
    device: str = "cpu",
    chunk_size: int = 1_000_000,
) -> list[NegativePool]:
    """ADR-0002 pools for the training and validation queries of a bundle loaded without test
    judgments: ``count`` sampled with replacement from the exact top-``depth`` minus judged
    positives and the query's own id, one seeded generator over train then validation."""
    rows = {qid: i for i, qid in enumerate(query_ids)}
    targets: list[tuple[str, str]] = [
        (split, qid) for split in ("train", "validation") for qid in bundle.query_ids(split)
    ]
    positives = {**bundle.positives("train"), **bundle.positives("validation")}
    missing = [qid for _, qid in targets if qid not in rows]
    if missing:
        raise ValueError(f"no query vectors for {missing[:5]}")
    vectors = torch.from_numpy(
        np.asarray(query_vectors[[rows[qid] for _, qid in targets]], dtype=np.float32)
    )
    rankings = exact_search_streamed(
        vectors, documents, min(depth, len(document_ids)), chunk_size=chunk_size, device=device
    )
    excluded = [positives.get(qid, set()) | {qid} for _, qid in targets]
    generator = torch.Generator().manual_seed(seed)
    sampled = sample_from_rankings(rankings, document_ids, excluded, count, generator)
    return [
        NegativePool(
            query_id=qid,
            environment=bundle.environment,
            split=split,
            doc_ids=pool,
            depth=depth,
            pool_size=count,
            seed=seed,
        )
        for (split, qid), pool in zip(targets, sampled, strict=True)
    ]
