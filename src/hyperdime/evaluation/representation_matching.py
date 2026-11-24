"""Coverage of a score-matched reference control for P3: can each positive find a reference?

Descriptive, in-sample, and unjudged documents stay unjudged. Scores are computed only against a
bounded seeded corpus sample, reused by all queries of a repeat, so the corpus is never ranked. This
measures matching only; it computes no mask, no ordering, and no curve (ADR-0014).

Matching is exact. Positives and sample documents lie on a line and a positive accepts a document
within a fixed caliper, so the intervals have equal width. ``largest_matching_size`` is the classic
exact two-pointer scan. ``exact_matching`` then finds, among matchings of that size, one with the
least total absolute residual: for equal-width intervals an optimal matching never crosses, so a
dynamic program over the sorted lists is exact. Each checks the other on every call.
"""

from __future__ import annotations

import hashlib
import json
import resource
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hyperdime.evaluation.representation import POSITIVE_STRATA, distribution, stratum_label
from hyperdime.evaluation.representation_controls import pair_indices, query_stream
from hyperdime.evaluation.representation_engine import (
    Clock,
    device_memory,
    gather,
    reset_device_memory,
)

MULTI_STRATA = POSITIVE_STRATA[1:]
STRATUM_FLOOR = 50
COMPLETE, TOO_FEW_DOCUMENTS, NO_DOCUMENT_IN_CALIPER, CONTENTION = 0, 1, 2, 3
REASONS = {
    COMPLETE: "complete",
    TOO_FEW_DOCUMENTS: "fewer eligible sample documents than positives",
    NO_DOCUMENT_IN_CALIPER: "a positive has no eligible sample document within the caliper",
    CONTENTION: "every positive has a document, but positives compete for too few",
}
DECILES = 10


def allocate_strata(sizes: Sequence[int], total: int, floor: int = STRATUM_FLOOR) -> list[int]:
    """Queries to draw per stratum: up to ``floor`` from each, the rest in proportion to what is
    left, by largest remainder. Deterministic, and never more than a stratum holds."""
    if sum(sizes) <= total:
        return list(sizes)
    floor = min(floor, total // max(sum(1 for size in sizes if size), 1))
    taken = [min(size, floor) for size in sizes]
    remaining = total - sum(taken)
    left = [size - got for size, got in zip(sizes, taken, strict=True)]
    pool = sum(left)
    shares = [remaining * size / pool if pool else 0.0 for size in left]
    extra = [int(share) for share in shares]
    order = sorted(range(len(sizes)), key=lambda i: (extra[i] - shares[i], i))
    for index in order[: remaining - sum(extra)]:
        extra[index] += 1
    return [got + more for got, more in zip(taken, extra, strict=True)]


def select_stratified(
    counts: Mapping[str, int], seed: int, total: int
) -> tuple[list[str], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Queries chosen by positive-count stratum, by a seeded hash within each.

    Returns sorted ids, each query's stratum, its sampling weight ``N_h / n_h``, and a table of
    strata. Only queries with at least two positives enter.
    """
    members: list[list[str]] = [[] for _ in MULTI_STRATA]
    for qid, count in counts.items():
        for index, (low, high) in enumerate(MULTI_STRATA):
            if count >= low and (high is None or count <= high):
                members[index].append(qid)
    drawn = allocate_strata([len(m) for m in members], total)
    stratum_of: dict[str, int] = {}
    weight_of: dict[str, float] = {}
    table = []
    for index, (low, high) in enumerate(MULTI_STRATA):
        ranked = sorted(
            members[index], key=lambda q: hashlib.sha256(f"{seed}:{q}".encode()).hexdigest()
        )
        chosen = ranked[: drawn[index]]
        weight = len(members[index]) / len(chosen) if chosen else 0.0
        for qid in chosen:
            stratum_of[qid], weight_of[qid] = index, weight
        table.append({
            "positives": stratum_label(low, high), "eligible_queries": len(members[index]),
            "selected_queries": len(chosen), "sampling_weight": weight,
        })  # fmt: skip
    ids = sorted(stratum_of)
    return (
        ids,
        np.array([stratum_of[q] for q in ids], dtype=np.int64),
        np.array([weight_of[q] for q in ids]),
        table,
    )


def largest_matching_size(positives: np.ndarray, candidates: np.ndarray, caliper: float) -> int:
    """Exact largest number of positives matched to distinct candidates; both lists ascending."""
    matched, next_candidate = 0, 0
    for score in positives.tolist():
        while next_candidate < candidates.size and score - candidates[next_candidate] > caliper:
            next_candidate += 1
        if next_candidate < candidates.size and candidates[next_candidate] - score <= caliper:
            matched += 1
            next_candidate += 1
    return matched


def _nearby(positives: np.ndarray, candidates: np.ndarray, caliper: float) -> np.ndarray:
    """Candidates worth considering: each positive's nearest ``P`` on each side, in its caliper.
    A matching using any other could swap to a closer free one and lose nothing."""
    count = positives.size
    middle = np.searchsorted(candidates, positives)
    low = np.searchsorted(candidates, positives - 2 * caliper)
    high = np.searchsorted(candidates, positives + 2 * caliper, side="right")
    spans = [
        np.arange(max(int(lo), int(mid) - count), min(int(hi), int(mid) + count))
        for lo, mid, hi in zip(low, middle, high, strict=True)
    ]
    return np.unique(np.concatenate(spans)) if spans else np.empty(0, dtype=np.int64)


def exact_matching(
    positives: np.ndarray, candidates: np.ndarray, caliper: float
) -> tuple[np.ndarray, np.ndarray]:
    """Positions ``(in positives, in candidates)`` of a largest matching with the least total
    absolute residual; both lists ascending. Ties resolve the same way on every run."""
    keep = _nearby(positives, candidates, caliper)
    empty = np.empty(0, dtype=np.int64)
    if not keep.size:
        return empty, empty
    near = candidates[keep]
    rows, columns = positives.size, near.size
    # A match counts far more than any residual, so size wins and residual breaks ties.
    bonus = 2.0 * rows * caliper + 1.0
    table = np.zeros((rows + 1, columns + 1))
    for row in range(rows):
        residual = np.abs(near - positives[row])
        cost = np.where(residual <= caliper, residual - bonus, np.inf)
        best = np.minimum(table[row, 1:], table[row, :-1] + cost)
        table[row + 1, 0] = table[row, 0]
        table[row + 1, 1:] = np.minimum.accumulate(np.minimum(best, table[row, 0]))
    chosen_rows, chosen_columns = [], []
    row, column = rows, columns
    while row > 0 and column > 0:
        if table[row, column] == table[row, column - 1]:
            column -= 1
        elif table[row, column] == table[row - 1, column]:
            row -= 1
        else:
            chosen_rows.append(row - 1)
            chosen_columns.append(column - 1)
            row, column = row - 1, column - 1
    found_rows = np.array(chosen_rows[::-1], dtype=np.int64)
    found = keep[np.array(chosen_columns[::-1], dtype=np.int64)] if chosen_columns else empty
    return found_rows, found


def nearest_unused_size(positives: np.ndarray, candidates: np.ndarray, caliper: float) -> int:
    """What greedy nearest-unused matching gets, positives taken in the order given; kept to show
    how many complete groups it would have missed, not used for any result."""
    used: set[int] = set()
    matched = 0
    for score in positives.tolist():
        at = int(np.searchsorted(candidates, score))
        left, right = at - 1, at
        while left >= 0 and left in used:
            left -= 1
        while right < candidates.size and right in used:
            right += 1
        options = []
        if left >= 0 and score - candidates[left] <= caliper:
            options.append((score - candidates[left], left))
        if right < candidates.size and candidates[right] - score <= caliper:
            options.append((candidates[right] - score, right))
        if options:
            used.add(min(options)[1])
            matched += 1
    return matched


def failure_reason(positives: np.ndarray, candidates: np.ndarray, caliper: float, size: int) -> int:
    """Why a group is incomplete. ``size`` is exact, so none of these is an algorithm's failure."""
    if size == positives.size:
        return COMPLETE
    if candidates.size < positives.size:
        return TOO_FEW_DOCUMENTS
    at = np.searchsorted(candidates, positives)
    above = np.where(at < candidates.size, candidates[np.minimum(at, candidates.size - 1)], np.inf)
    below = np.where(at > 0, candidates[np.maximum(at - 1, 0)], -np.inf)
    if (np.minimum(above - positives, positives - below) > caliper).any():
        return NO_DOCUMENT_IN_CALIPER
    return CONTENTION


@dataclass
class _Env:
    query_ids: Sequence[str]
    query_vector: Callable[[str], np.ndarray]
    documents: np.ndarray
    positives: Mapping[str, np.ndarray]
    judged: Mapping[str, np.ndarray]


def mean_cosine(vectors: torch.Tensor, first: np.ndarray, second: np.ndarray) -> float:
    """Mean cosine over the given pairs of rows."""
    units = vectors / vectors.norm(dim=1, keepdim=True)
    index_a = torch.from_numpy(first).to(vectors.device)
    index_b = torch.from_numpy(second).to(vectors.device)
    return float((units[index_a] * units[index_b]).sum(dim=1).mean().item())


def analyze_matching(
    env: _Env,
    *,
    seed: int,
    repeats: int,
    sample_size: int,
    caliper: float,
    max_query_pairs: int,
    weights: np.ndarray,
    strata: np.ndarray,
    strata_table: list[dict[str, Any]],
    device: str,
    batch_queries: int,
    path: Callable[[str], Path],
) -> dict[str, Any]:
    """Match every selected query's positives in every repeat, and summarize coverage."""
    started = time.perf_counter()
    reset_device_memory(device)
    clock = Clock(device)
    queries = len(env.query_ids)
    corpus = env.documents.shape[0]
    counts = np.array([env.positives[q].size for q in env.query_ids], dtype=np.int64)
    shape = (queries, repeats)
    matched = np.zeros(shape, dtype=np.int64)
    reasons = np.zeros(shape, dtype=np.int64)
    available = np.zeros(shape, dtype=np.int64)
    greedy = np.zeros(shape, dtype=np.int64)
    disagreements = 0
    positive_cosine = np.full(queries, np.nan)
    matched_cosine = np.full(shape, np.nan)
    random_cosine = np.full(shape, np.nan)
    sample_mean = np.full(shape, np.nan)
    store: dict[str, list[np.ndarray]] = {k: [] for k in ("query", "repeat", "positive_row",
                                                           "document_row")}  # fmt: skip
    scores: dict[str, list[np.ndarray]] = {"positive": [], "reference": [], "unmatched": []}
    with clock.span("read"):
        query_rows = np.stack([np.asarray(env.query_vector(q), np.float64) for q in env.query_ids])
    for repeat in range(repeats):
        rng = np.random.default_rng([seed, 40 + repeat])
        sample_rows = np.sort(rng.permutation(corpus)[: min(sample_size, corpus // 2)])
        with clock.span("read"):
            sample_np = gather(env.documents, sample_rows)
        with clock.span("transfer"):
            sample = torch.from_numpy(sample_np).to(device)
        del sample_np
        for start in range(0, queries, batch_queries):
            stop = start + batch_queries
            with clock.span("transfer"):
                batch = torch.from_numpy(query_rows[start:stop]).to(device)
            with clock.span("compute"):
                ordered, order = torch.sort(batch @ sample.T, dim=1, stable=True)
            with clock.span("transfer"):
                ordered_np, order_np = ordered.cpu().numpy(), order.cpu().numpy()
            for offset, qid in enumerate(env.query_ids[start:stop]):
                index = start + offset
                with clock.span("match"):
                    keep = ~np.isin(sample_rows, env.judged[qid])
                    usable = keep[order_np[offset]]
                    candidates = ordered_np[offset][usable]
                    positions = order_np[offset][usable]
                    rows = env.positives[qid]
                    positive_scores = np.asarray(gather(env.documents, rows) @ query_rows[index])
                    by_score = np.argsort(positive_scores, kind="stable")
                    sorted_scores = positive_scores[by_score]
                    size = largest_matching_size(sorted_scores, candidates, caliper)
                    found_rows, found_columns = exact_matching(sorted_scores, candidates, caliper)
                    disagreements += int(found_rows.size != size)
                    available[index, repeat] = candidates.size
                    matched[index, repeat] = size
                    reasons[index, repeat] = failure_reason(
                        sorted_scores, candidates, caliper, size
                    )
                    greedy[index, repeat] = nearest_unused_size(
                        positive_scores, candidates, caliper
                    )
                    sample_mean[index, repeat] = float(candidates.mean()) if candidates.size else 0
                store["query"].append(np.full(found_rows.size, index, dtype=np.int64))
                store["repeat"].append(np.full(found_rows.size, repeat, dtype=np.int64))
                store["positive_row"].append(rows[by_score[found_rows]])
                store["document_row"].append(sample_rows[positions[found_columns]])
                scores["positive"].append(sorted_scores[found_rows])
                scores["reference"].append(candidates[found_columns])
                unmatched = np.ones(rows.size, dtype=bool)
                unmatched[found_rows] = False
                scores["unmatched"].append(sorted_scores[unmatched])
                if size == rows.size:
                    with clock.span("compute"):
                        pair_rng = query_stream(seed, 200 + repeat, qid)
                        first, second = pair_indices(rows.size, max_query_pairs, pair_rng)
                        matched_cosine[index, repeat] = mean_cosine(
                            sample[torch.from_numpy(positions[found_columns]).to(device)],
                            first,
                            second,
                        )
                        drawn = pair_rng.permutation(positions.size)[: rows.size]
                        random_cosine[index, repeat] = mean_cosine(
                            sample[torch.from_numpy(positions[drawn]).to(device)], first, second
                        )
            del ordered, order, ordered_np, order_np
        del sample
    with clock.span("compute"):
        for index, qid in enumerate(env.query_ids):
            pair_rng = query_stream(seed, 199, qid)
            first, second = pair_indices(counts[index], max_query_pairs, pair_rng)
            vectors = torch.from_numpy(gather(env.documents, env.positives[qid])).to(device)
            positive_cosine[index] = mean_cosine(vectors, first, second)
    with clock.span("write"):
        path("selected-queries.json").write_text(json.dumps(list(env.query_ids)), encoding="utf-8")
        rows_found: dict[str, Any] = {key: np.concatenate(v) for key, v in store.items()}
        np.savez_compressed(path("matched-rows.npz"), **rows_found)
        np.savez_compressed(
            path("query-results.npz"),
            positives=counts, stratum=strata, weight=weights, matched=matched, reason=reasons,
            available=available, nearest_unused=greedy, positive_cosine=positive_cosine,
            matched_cosine=matched_cosine, random_cosine=random_cosine, sample_mean=sample_mean,
            positive_score=np.concatenate(scores["positive"]),
            reference_score=np.concatenate(scores["reference"]),
            unmatched_score=np.concatenate(scores["unmatched"]),
        )  # fmt: skip
    summary = summarize(
        counts, strata, weights, strata_table, matched, reasons, greedy, positive_cosine,
        matched_cosine, random_cosine, sample_mean,
        {name: np.concatenate(items) for name, items in scores.items()},
    )  # fmt: skip
    summary["exact_check_disagreements"] = disagreements
    summary["settings"] = {
        "repeats": repeats, "sample_documents": int(min(sample_size, corpus // 2)),
        "caliper": caliper, "max_query_pairs": max_query_pairs, "batch_queries": batch_queries,
        "corpus_documents": int(corpus),
    }  # fmt: skip
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    total = time.perf_counter() - started
    summary["cost"] = {
        "device": device, "device_memory": device_memory(device), "total_seconds": total,
        "phase_seconds": clock.seconds, "peak_rss_mib": peak,
        "seconds_per_query_repeat": total / max(queries * repeats, 1),
    }  # fmt: skip
    return summary


def stratified_mean(
    values: np.ndarray, strata: np.ndarray, table: list[dict[str, Any]]
) -> dict[str, float | None]:
    """Stratified estimate over all eligible queries and its standard error, with a finite
    population correction; values are per query, and undefined ones leave their stratum's mean."""
    total = sum(row["eligible_queries"] for row in table)
    estimate, variance = 0.0, 0.0
    for index, row in enumerate(table):
        chosen = values[strata == index]
        chosen = chosen[np.isfinite(chosen)]
        if not chosen.size:
            if row["eligible_queries"]:
                return {"estimate": None, "standard_error": None}
            continue
        share = row["eligible_queries"] / total
        estimate += share * float(chosen.mean())
        if chosen.size > 1:
            sampled = row["selected_queries"] / row["eligible_queries"]
            variance += share**2 * (1 - sampled) * float(chosen.var(ddof=1)) / chosen.size
    return {"estimate": estimate, "standard_error": float(np.sqrt(variance))}


def summarize(
    counts: np.ndarray,
    strata: np.ndarray,
    weights: np.ndarray,
    table: list[dict[str, Any]],
    matched: np.ndarray,
    reasons: np.ndarray,
    greedy: np.ndarray,
    positive_cosine: np.ndarray,
    matched_cosine: np.ndarray,
    random_cosine: np.ndarray,
    sample_mean: np.ndarray,
    scores: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Coverage in the selected sample, and weighted to all eligible queries."""
    complete = matched == counts[:, None]
    query_complete = complete.mean(axis=1)
    found: dict[str, Any] = {"strata": []}
    for index, row in enumerate(table):
        chosen = strata == index
        found["strata"].append({
            **row,
            "complete_rate": float(query_complete[chosen].mean()) if chosen.any() else None,
            "complete_rate_by_repeat": complete[chosen].mean(axis=0).tolist() if chosen.any()
            else [],
            "pair_rate": float(matched[chosen].sum() / (counts[chosen].sum() * matched.shape[1]))
            if chosen.any() else None,
            "positives_in_selection": int(counts[chosen].sum()),
        })  # fmt: skip
    found["sample"] = {
        "queries": int(counts.size),
        "pairs": int(counts.sum()),
        "complete_rate_unweighted": float(query_complete.mean()),
        "pair_rate_unweighted": float(matched.sum() / (counts.sum() * matched.shape[1])),
        "complete_rate_by_repeat_unweighted": complete.mean(axis=0).tolist(),
        "complete_in_all_repeats": int(complete.all(axis=1).sum()),
        "complete_in_no_repeat": int((~complete.any(axis=1)).sum()),
    }
    by_repeat = [
        stratified_mean(complete[:, repeat].astype(np.float64), strata, table)
        for repeat in range(matched.shape[1])
    ]
    found["weighted"] = {
        "complete_rate": stratified_mean(query_complete, strata, table),
        "complete_rate_by_repeat": [row["estimate"] for row in by_repeat],
        "pair_rate": _weighted_pair_rate(matched, counts, weights),
    }
    found["reasons"] = {REASONS[code]: int((reasons == code).sum()) for code in REASONS}
    found["nearest_unused"] = {
        "exact_complete_query_repeats": int(complete.sum()),
        "greedy_complete_query_repeats": int((greedy == counts[:, None]).sum()),
        "query_repeats_greedy_falsely_failed": int((complete & (greedy < counts[:, None])).sum()),
    }
    positive, reference = scores["positive"], scores["reference"]
    found["scores"] = {
        "matched_positive": distribution(positive),
        "matched_reference": distribution(reference),
        "unmatched_positive": distribution(scores["unmatched"]),
        "residual_reference_minus_positive": distribution(reference - positive),
        "absolute_residual": distribution(np.abs(reference - positive)),
        "largest_absolute_residual": float(np.abs(reference - positive).max())
        if positive.size
        else None,
        "sample_mean_score_per_query": distribution(sample_mean),
        "pair_rate_by_positive_score_decile": _deciles(scores),
    }
    found["selection"] = _selection(counts, complete, positive_cosine, weights)
    mean_matched, mean_random = _row_means(matched_cosine), _row_means(random_cosine)
    found["cosine"] = {
        "queries_complete_at_least_once": int(complete.any(axis=1).sum()),
        "positive": distribution(positive_cosine[complete.any(axis=1)]),
        "matched_reference": distribution(mean_matched),
        "random_reference": distribution(mean_random),
        "positive_minus_matched": distribution(positive_cosine - mean_matched),
        "positive_minus_random": distribution(positive_cosine - mean_random),
        "matched_by_repeat": _nanmean_columns(matched_cosine),
    }
    return found


def _row_means(values: np.ndarray) -> np.ndarray:
    """Each row's mean of finite values; NaN for a row with none, without a warning."""
    seen = np.isfinite(values)
    counts = seen.sum(axis=1)
    totals = np.where(seen, values, 0.0).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        result: np.ndarray = totals / counts
    return result


def _nanmean_columns(values: np.ndarray) -> list[float | None]:
    return [
        float(values[:, column][np.isfinite(values[:, column])].mean())
        if np.isfinite(values[:, column]).any() else None
        for column in range(values.shape[1])
    ]  # fmt: skip


def _weighted_pair_rate(matched: np.ndarray, counts: np.ndarray, weights: np.ndarray) -> float:
    """Share of the eligible queries' positive pairs matched, as a ratio of weighted totals."""
    found = float((weights[:, None] * matched).sum())
    return found / float((weights * counts).sum() * matched.shape[1])


def _deciles(scores: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    """Pair match rate over positive-score deciles of the selection; pairs, not queries."""
    matched, unmatched = scores["positive"], scores["unmatched"]
    every = np.concatenate([matched, unmatched])
    if not every.size:
        return []
    edges = np.quantile(every, np.linspace(0, 1, DECILES + 1))
    rows = []
    for index in range(DECILES):
        low, high = edges[index], edges[index + 1]
        top = index == DECILES - 1
        hits = ((matched >= low) & (matched <= high if top else matched < high)).sum()
        misses = ((unmatched >= low) & (unmatched <= high if top else unmatched < high)).sum()
        total = hits + misses
        rows.append({
            "low": float(low), "high": float(high), "pairs": int(total),
            "pair_rate": float(hits / total) if total else None,
        })  # fmt: skip
    return rows


def _selection(
    counts: np.ndarray, complete: np.ndarray, positive_cosine: np.ndarray, weights: np.ndarray
) -> list[dict[str, Any]]:
    """Which queries can be matched at all: what a matched comparison would leave out."""
    rows = []
    for label, chosen in (
        ("complete in every repeat", complete.all(axis=1)),
        ("complete in some repeats", complete.any(axis=1) & ~complete.all(axis=1)),
        ("complete in no repeat", ~complete.any(axis=1)),
    ):
        total = float(weights.sum())
        rows.append({
            "group": label, "queries": int(chosen.sum()),
            "weighted_share": float(weights[chosen].sum() / total) if total else None,
            "positives": distribution(counts[chosen]),
            "positive_cosine": distribution(positive_cosine[chosen]),
        })  # fmt: skip
    return rows
