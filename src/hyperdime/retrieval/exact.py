"""Exact inner-product retrieval."""

from __future__ import annotations

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
