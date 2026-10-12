"""Exact masked evaluation of many (method, k) pairs in one corpus pass (F3 research D1).

Each pair masks the query matrix (unselected coordinates zeroed, never re-normalized); all masked
copies are stacked and ranked together by ``exact_search_streamed`` over the unmasked documents,
so a memory-mapped corpus is read once per query set.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from hyperdime.contracts.schemas import EvaluationResult
from hyperdime.evaluation.metrics import query_metrics
from hyperdime.retrieval.exact import exact_search_streamed
from hyperdime.selection.topk import apply_mask, top_k_mask

DIM = 1024
METRICS = ("ndcg@10", "mrr@10", "mrr@100", "recall@100", "ap")


@dataclass(frozen=True)
class QuerySet:
    """Queries of one evaluation set with their judgments and metric conventions."""

    name: str
    query_ids: list[str]
    vectors: Tensor
    judgments: Mapping[str, Mapping[str, int]]
    relevance_threshold: int = 1
    depth: int = 100


@dataclass
class Evaluation:
    results: list[EvaluationResult]
    rankings: dict[tuple[str, int], list[list[str]]] = field(default_factory=dict)
    excluded_queries: int = 0


def masked_queries(queries: Tensor, importance: Tensor | None, k: int) -> Tensor:
    """Zero all but each query's top-``k`` dimensions; ``None`` (or k = D) is the full query."""
    if importance is None or k == queries.shape[1]:
        return queries.float()
    return apply_mask(queries.float(), top_k_mask(importance, k))


def evaluate_methods(
    query_set: QuerySet,
    methods: Mapping[str, Tensor | None],
    ks: Sequence[int],
    documents: np.ndarray,
    doc_ids: Sequence[str],
    *,
    environment: str,
    protocol: str = "in-domain",
    chunk_size: int = 1_000_000,
    device: str = "cpu",
    keep_k: Sequence[int] = (),
    pairs: Sequence[tuple[str, int]] | None = None,
) -> Evaluation:
    """Mean and per-query metrics for every method at every k, from one streamed corpus pass.

    ``methods`` maps a method name to per-query importance scores ``(Q, D)`` aligned with
    ``query_set.query_ids`` (``None`` for the unmasked query). ``pairs`` overrides the full
    product of methods and ``ks``; rankings are kept for the budgets in ``keep_k``. Queries with no
    document at or above the relevance threshold are excluded from the metrics and counted.
    """
    judged = [
        i
        for i, qid in enumerate(query_set.query_ids)
        if any(
            g >= query_set.relevance_threshold for g in query_set.judgments.get(qid, {}).values()
        )
    ]
    query_ids = [query_set.query_ids[i] for i in judged]
    queries = query_set.vectors[judged].float()
    if pairs is None:
        pairs = [(name, k) for name in methods for k in ks]
    variants = []
    for name, k in pairs:
        importance = methods[name]
        variants.append(
            masked_queries(queries, None if importance is None else importance[judged], k)
        )
    depth = min(query_set.depth + 1, len(doc_ids))  # one extra in case a document is the query
    rows = exact_search_streamed(
        torch.cat(variants), documents, depth, chunk_size=chunk_size, device=device
    )
    results, rankings = [], {}
    for index, (name, k) in enumerate(pairs):
        block = rows[index * len(query_ids) : (index + 1) * len(query_ids)].tolist()
        ranked = [
            [doc_ids[r] for r in row if doc_ids[r] != qid][: query_set.depth]
            for row, qid in zip(block, query_ids, strict=True)
        ]
        per_query = {
            qid: {
                metric: value
                for metric, value in query_metrics(
                    ranking,
                    query_set.judgments[qid],
                    (10, 100),
                    relevance_threshold=query_set.relevance_threshold,
                ).items()
                if metric in METRICS
            }
            for qid, ranking in zip(query_ids, ranked, strict=True)
        }
        means = {m: sum(v[m] for v in per_query.values()) / len(per_query) for m in METRICS}
        results.append(
            EvaluationResult(
                environment=environment,
                protocol=protocol,
                split="test",
                selector=name,
                k=k,
                metrics=means,
                per_query=per_query,
                num_queries=len(per_query),
            )
        )
        if k in keep_k:
            rankings[(name, k)] = ranked
    return Evaluation(results, rankings, len(query_set.query_ids) - len(query_ids))


E0_RATIOS = tuple(step / 10 for step in range(1, 11))


def ratio_budgets(ratios: Sequence[float] = E0_RATIOS) -> list[int]:
    """Retained-dimension budgets ``floor(ratio * 1024)`` for the E0 ratios 0.1 ... 1.0."""
    from hyperdime.selection.topk import resolve_k

    return sorted({resolve_k(r, DIM) for r in ratios})
