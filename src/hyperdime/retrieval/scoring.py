"""Rankings from exact scores, with BEIR's identical-ID exclusion."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor

from hyperdime.retrieval.exact import exact_search


def rank_documents(
    query_embeddings: Tensor,
    document_embeddings: Tensor,
    query_ids: Sequence[str],
    document_ids: Sequence[str],
    depth: int,
    exclude_identical_ids: bool = True,
) -> dict[str, list[str]]:
    """Top-``depth`` document IDs per query; a document with the query's own ID is skipped.

    The exclusion follows BEIR's evaluation for datasets such as ArguAna, whose queries also
    appear in the corpus.
    """
    if len(query_ids) != query_embeddings.shape[0]:
        raise ValueError("query_ids must align with query_embeddings rows.")
    extra = 1 if exclude_identical_ids else 0
    _, indices = exact_search(
        query_embeddings, document_embeddings, min(depth + extra, len(document_ids))
    )
    rankings: dict[str, list[str]] = {}
    for qid, row in zip(query_ids, indices.tolist(), strict=True):
        ranked = [document_ids[i] for i in row]
        if exclude_identical_ids:
            ranked = [doc_id for doc_id in ranked if doc_id != qid]
        rankings[qid] = ranked[:depth]
    return rankings
