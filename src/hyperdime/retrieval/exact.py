"""Exact inner-product retrieval."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def exact_search(
    query_embeddings: Tensor, document_embeddings: Tensor, top_n: int, batch_size: int = 1024
) -> tuple[Tensor, Tensor]:
    """Return float32 ``(scores, indices)`` of shape ``[num_queries, top_n]``."""
    if query_embeddings.ndim != 2 or document_embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [N, D].")
    if query_embeddings.shape[1] != document_embeddings.shape[1]:
        raise ValueError("query and document dimensions differ.")
    if not 1 <= top_n <= document_embeddings.shape[0]:
        raise ValueError("top_n must be in [1, number of documents].")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    documents = document_embeddings.float()
    score_parts: list[Tensor] = []
    index_parts: list[Tensor] = []
    for start in range(0, query_embeddings.shape[0], batch_size):
        scores = query_embeddings[start : start + batch_size].float() @ documents.T
        values, indices = scores.topk(top_n, dim=-1)
        score_parts.append(values)
        index_parts.append(indices)
    return torch.cat(score_parts), torch.cat(index_parts)


def exact_search_streamed(
    queries: Tensor,
    documents: np.ndarray,
    depth: int,
    *,
    chunk_size: int = 1_000_000,
    query_batch: int = 1024,
    device: str = "cpu",
) -> Tensor:
    """Row indices of each query's top-``depth`` documents by exact inner product.

    ``documents`` may be a memory-mapped float16 matrix larger than RAM: each chunk of
    ``chunk_size`` rows is moved to ``device`` once and merged into a running top-``depth`` for
    every query, so the full score matrix never exists (F2 research D6).
    """
    if queries.ndim != 2 or documents.ndim != 2 or queries.shape[1] != documents.shape[1]:
        raise ValueError("queries and documents must have shapes [Q, D] and [N, D].")
    total = documents.shape[0]
    if not 1 <= depth <= total:
        raise ValueError("depth must be in [1, number of documents].")
    if chunk_size < 1 or query_batch < 1:
        raise ValueError("chunk_size and query_batch must be positive.")
    query_rows = queries.float().to(device)
    best_scores = torch.full((query_rows.shape[0], 0), float("-inf"), device=device)
    best_rows = torch.zeros((query_rows.shape[0], 0), dtype=torch.long, device=device)
    for start in range(0, total, chunk_size):
        block = np.array(documents[start : start + chunk_size], dtype=np.float32)
        chunk = torch.from_numpy(block).to(device)
        scores_parts, rows_parts = [], []
        for first in range(0, query_rows.shape[0], query_batch):
            batch = query_rows[first : first + query_batch]
            scores = batch @ chunk.T
            top = scores.topk(min(depth, chunk.shape[0]), dim=-1)
            merged_scores = torch.cat([best_scores[first : first + query_batch], top.values], dim=1)
            merged_rows = torch.cat(
                [best_rows[first : first + query_batch], top.indices + start], dim=1
            )
            keep = merged_scores.topk(min(depth, merged_scores.shape[1]), dim=-1)
            scores_parts.append(keep.values)
            rows_parts.append(merged_rows.gather(1, keep.indices))
        best_scores, best_rows = torch.cat(scores_parts), torch.cat(rows_parts)
    return best_rows.cpu()
