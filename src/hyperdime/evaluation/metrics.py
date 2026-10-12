"""Ranking metrics with per-query values for paired significance tests."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

Qrels = Mapping[str, Mapping[str, int]]


def query_metrics(
    ranking: Sequence[str],
    judgments: Mapping[str, int],
    cutoffs: Sequence[int],
    *,
    relevance_threshold: int = 1,
) -> dict[str, float]:
    """nDCG (graded, linear gain, as trec_eval) at each cutoff; reciprocal rank and recall at each
    cutoff and average precision over the whole ranking, counting a document as relevant when its
    grade is at least ``relevance_threshold`` (2 for TREC DL passage judgments)."""
    if relevance_threshold < 1:
        raise ValueError("relevance_threshold must be at least 1.")
    graded = {doc_id: gain for doc_id, gain in judgments.items() if gain > 0}
    relevant = {doc_id for doc_id, gain in judgments.items() if gain >= relevance_threshold}
    ideal_gains = sorted(graded.values(), reverse=True)
    values: dict[str, float] = {}
    for cutoff in cutoffs:
        if cutoff < 1:
            raise ValueError("cutoffs must be positive.")
        top = ranking[:cutoff]
        dcg = sum(graded.get(doc_id, 0) / math.log2(rank + 2) for rank, doc_id in enumerate(top))
        idcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(ideal_gains[:cutoff]))
        hits = sum(doc_id in relevant for doc_id in top)
        first = next((rank for rank, doc_id in enumerate(top, 1) if doc_id in relevant), None)
        values[f"ndcg@{cutoff}"] = dcg / idcg if idcg else 0.0
        values[f"recall@{cutoff}"] = hits / len(relevant) if relevant else 0.0
        values[f"mrr@{cutoff}"] = 1 / first if first else 0.0
    found, precision_sum = 0, 0.0
    for rank, doc_id in enumerate(ranking, 1):
        if doc_id in relevant:
            found += 1
            precision_sum += found / rank
    values["ap"] = precision_sum / len(relevant) if relevant else 0.0
    return values


def evaluate_run(
    run: Mapping[str, Sequence[str]], qrels: Qrels, cutoffs: Sequence[int] = (10, 100)
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Return mean metrics and per-query metrics over queries with a relevant judgment."""
    judged = sorted(qid for qid, labels in qrels.items() if any(g > 0 for g in labels.values()))
    if not judged:
        raise ValueError("qrels contain no query with a relevant document.")
    missing = [qid for qid in judged if qid not in run]
    if missing:
        raise KeyError(f"run has no ranking for {len(missing)} judged queries, e.g. {missing[0]}.")
    per_query = {qid: query_metrics(run[qid], qrels[qid], cutoffs) for qid in judged}
    names = next(iter(per_query.values())).keys()
    means = {
        name: sum(values[name] for values in per_query.values()) / len(judged) for name in names
    }
    return means, per_query
