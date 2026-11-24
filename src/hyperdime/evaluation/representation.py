"""Experiment P3: what per-dimension contributions ``c_j = q_j * d_j`` reveal about known pairs.

Descriptive analysis in the stored retrieval space (ADR-0014). Every relevant document stays an
individual row until aggregation is itself the thing analyzed. Two notions are measured, and a
third is deliberately not: (1) maximizing similarity to one known relevant document, (2) preserving
the original similarity, and not (3) improving a ranking. This module scores only judged documents;
``representation_controls`` adds the all-cardinality curves and an unjudged reference.

Queries are masked only as an analytic device: nothing here writes to an embedding matrix, a masked
query is never re-normalized, and no summary treats an unjudged document as nonrelevant.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hyperdime.evaluation.representation_engine import Clock, write_pair_orders
from hyperdime.evaluation.selector_shift import spearman

DIM = 1024
FRACTIONS = (0.5, 0.8, 0.9, 0.95, 0.99)
TOP_SHARES = (1, 10, 100)
SOURCE_TOP = 50
POSITIVE_STRATA = ((1, 1), (2, 5), (6, 20), (21, None))
EXPLAINED_COMPONENTS = 10
FLUSH_QUERIES = 1024
CHUNK_PAIRS = 4096
Judgments = Mapping[str, Mapping[str, int]]


def contributions(query: np.ndarray, documents: np.ndarray) -> np.ndarray:
    """``c_ij = q_j * d_ij`` in float64, ``[documents, D]``; each row sums to ``dot(q, d)``."""
    return np.asarray(documents, dtype=np.float64) * np.asarray(query, dtype=np.float64)


def contribution_order(values: np.ndarray) -> np.ndarray:
    """Dimensions by descending contribution, per row; ties go to the lower dimension index."""
    return np.argsort(-values, axis=1, kind="stable")


def cumulative_scores(values: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Curve of each pair's score over its ordering: entry ``k - 1`` is the largest unnormalized
    inner product any mask of exactly ``k`` dimensions can give that pair."""
    return np.cumsum(np.take_along_axis(values, order, axis=1), axis=1)


def budget_masks(order: np.ndarray, k: int) -> np.ndarray:
    """Boolean ``[rows, D]`` masks with exactly ``k`` dimensions, from an existing ordering."""
    masks = np.zeros(order.shape, dtype=bool)
    np.put_along_axis(masks, order[:, :k], True, axis=1)
    return masks


def relevance_gains(grades: Sequence[int] | np.ndarray) -> np.ndarray:
    """Exponential gains ``2**grade - 1``, whole numbers and so exact in float64."""
    gains: np.ndarray = 2.0 ** np.asarray(grades, dtype=np.float64) - 1.0
    if gains.size == 0 or (gains <= 0).any():
        raise ValueError("gain weights need at least one positive grade and no grade 0.")
    return gains


def gain_weights(grades: Sequence[int] | np.ndarray) -> np.ndarray:
    """Normalized exponential gains ``2**grade - 1``, the repository's oracle weighting."""
    gains = relevance_gains(grades)
    weights: np.ndarray = gains / gains.sum()
    return weights


def chance_jaccard(k: int, dim: int = DIM) -> float:
    """Jaccard of two independent uniform k-subsets at their expected overlap ``k²/dim``, the
    reference printed next to every mask overlap."""
    return k / (2 * dim - k)


def query_balanced(values: np.ndarray, query_index: np.ndarray, queries: int) -> np.ndarray:
    """Mean of ``values`` (first axis: pairs) within each query, so each query counts once."""
    counts = np.bincount(query_index, minlength=queries).astype(np.float64)
    if (counts == 0).any():
        raise ValueError("every query must have at least one row.")
    totals = np.zeros((queries, *values.shape[1:]), dtype=np.float64)
    np.add.at(totals, query_index, values)
    shape = (queries,) + (1,) * (values.ndim - 1)
    result: np.ndarray = totals / counts.reshape(shape)
    return result


def distribution(values: np.ndarray) -> dict[str, float | int | None]:
    """Mean and 10th/50th/90th percentiles of finite values; ``None`` when there are none."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"n": 0, "mean": None, "p10": None, "p50": None, "p90": None}
    p10, p50, p90 = np.quantile(finite, [0.1, 0.5, 0.9]).tolist()
    return {"n": int(finite.size), "mean": float(finite.mean()), "p10": p10, "p50": p50, "p90": p90}


def average_ranks(rows: np.ndarray) -> np.ndarray:
    """Ranks within each row; tied values share their average rank."""
    ranks = np.empty(rows.shape, dtype=np.float64)
    for index, row in enumerate(rows):
        _, inverse, counts = np.unique(row, return_inverse=True, return_counts=True)
        ends = np.cumsum(counts)
        ranks[index] = (ends - (counts - 1) / 2.0 - 1.0)[inverse]
    return ranks


def pair_statistics(
    query: np.ndarray,
    documents: np.ndarray,
    budgets: Sequence[int],
    curve_k: Sequence[int],
    order: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """The exact decomposition of one query against each document row, one entry per pair.

    Ratios use a mass as denominator, never a signed score: positive mass ``P = Σ max(c, 0)``,
    negative mass ``N = Σ max(-c, 0)``, and total mass ``P + N``; a zero denominator gives NaN, and
    summaries drop NaN. ``log|c| = log|q| + log|d|`` decomposes exactly on dimensions where both
    are nonzero, so ``var(log|c|)`` splits into query, document, and covariance shares that sum to
    one and need no threshold. ``order`` is each row's ordering, when it is already known.
    """
    q = np.asarray(query, dtype=np.float64)
    docs = np.asarray(documents, dtype=np.float64)
    c = contributions(q, docs)
    order = contribution_order(c) if order is None else order
    cumulative = cumulative_scores(c, order)
    full = docs @ q
    positive_part, negative_part = np.clip(c, 0, None), np.clip(-c, 0, None)
    pos, neg = positive_part.sum(axis=1), negative_part.sum(axis=1)
    n_pos, n_neg = (c > 0).sum(axis=1), (c < 0).sum(axis=1)
    rows = np.arange(c.shape[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        stats: dict[str, np.ndarray] = {
            "score": full,
            "decomposition_error": np.abs(c.sum(axis=1) - full),
            "query_norm": np.full(c.shape[0], np.linalg.norm(q)),
            "document_norm": np.linalg.norm(docs, axis=1),
            "positive_mass": pos,
            "negative_mass": neg,
            "negative_share": neg / (pos + neg),
            "n_positive": n_pos.astype(np.float64),
            "n_negative": n_neg.astype(np.float64),
            "n_zero": (c == 0).sum(axis=1).astype(np.float64),
            "sign_agreement": n_pos / (n_pos + n_neg),
            "effective_positive_dims": pos**2 / (positive_part**2).sum(axis=1),
            "effective_negative_dims": neg**2 / (negative_part**2).sum(axis=1),
            "k_return": (cumulative >= full[:, None] - 1e-12).argmax(axis=1) + 1.0,
        }
        for top in TOP_SHARES:
            kept = cumulative[rows, np.minimum(top, np.maximum(n_pos, 1)) - 1]
            stats[f"top_{top}_share"] = np.where(n_pos > 0, kept, 0.0) / pos
        for fraction in FRACTIONS:
            reached = cumulative >= fraction * pos[:, None]
            stats[f"dims_for_{round(fraction * 100)}"] = np.where(
                pos > 0, reached.argmax(axis=1) + 1.0, np.nan
            )
        stats["curve"] = cumulative[:, np.asarray(curve_k) - 1] / pos[:, None]
        scores = cumulative[:, np.asarray(budgets) - 1]
        # All positive contributions sort first, so positive mass kept at k is the curve at
        # min(k, n_pos); whatever a k-subset beyond that adds is negative mass.
        kept_rows = np.minimum.outer(np.asarray(budgets), np.maximum(n_pos, 1)).T - 1
        positive_kept = np.where(n_pos[:, None] > 0, cumulative[rows[:, None], kept_rows], 0.0)
        stats["max_score"] = scores
        stats["removed_positive_mass"] = pos[:, None] - positive_kept
        stats["removed_negative_mass"] = neg[:, None] - (positive_kept - scores)
        stats["preservation_error"] = np.abs(scores - full[:, None]) / (pos + neg)[:, None]
        stats["gain_over_full"] = (scores - full[:, None]) / (pos + neg)[:, None]
    stats.update(_magnitude_sources(q, docs, order))
    return stats


def _magnitude_sources(q: np.ndarray, docs: np.ndarray, order: np.ndarray) -> dict[str, np.ndarray]:
    """Whether large contributions come from the query's magnitude, the document's, or both."""
    both = (q != 0) & (docs != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_q = np.where(both, np.log(np.abs(q)), 0.0)
        log_d = np.where(both, np.log(np.abs(docs)), 0.0)
        count = both.sum(axis=1)
        mean_q, mean_d = log_q.sum(axis=1) / count, log_d.sum(axis=1) / count
        centered_q = np.where(both, log_q - mean_q[:, None], 0.0)
        centered_d = np.where(both, log_d - mean_d[:, None], 0.0)
        var_q = (centered_q**2).sum(axis=1) / count
        var_d = (centered_d**2).sum(axis=1) / count
        cov = (centered_q * centered_d).sum(axis=1) / count
        total = var_q + var_d + 2 * cov
        found = {
            "log_variance_query_share": var_q / total,
            "log_variance_document_share": var_d / total,
            "log_variance_covariance_share": 2 * cov / total,
        }
    top = order[:, :SOURCE_TOP]
    query_percentile = _percentiles(np.abs(q)[None, :])[0]
    document_percentile = _percentiles(np.abs(docs))
    found["top_query_percentile"] = query_percentile[top].mean(axis=1)
    found["top_document_percentile"] = np.take_along_axis(document_percentile, top, axis=1).mean(
        axis=1
    )
    return found


def _percentiles(values: np.ndarray) -> np.ndarray:
    percentiles: np.ndarray = average_ranks(values) / (values.shape[1] - 1)
    return percentiles


def query_agreement(
    values: np.ndarray, order: np.ndarray, grades: np.ndarray, budgets: Sequence[int]
) -> dict[str, np.ndarray]:
    """How one query's positives agree, keeping conflicts visible; NaN with a single positive."""
    count, dim = values.shape
    budget_shape = (len(budgets),)
    found: dict[str, np.ndarray] = {
        name: np.full(budget_shape, np.nan)
        for name in (
            "mean_jaccard", "min_jaccard", "core_dims", "majority_dims", "minority_dims",
            "union_dims", "same_grade_jaccard", "cross_grade_jaccard",
        )
    }  # fmt: skip
    for name in (
        "always_helpful_dims", "always_harmful_dims", "sign_conflict_dims", "sign_conflict_share",
        "mean_spearman", "min_spearman", "same_grade_spearman", "cross_grade_spearman",
    ):  # fmt: skip
        found[name] = np.array(np.nan)
    if count < 2:
        return found
    above = np.triu_indices(count, 1)
    same_grade = (grades[:, None] == grades[None, :])[above]
    for b, k in enumerate(budgets):
        masks = budget_masks(order, k)
        intersection = (masks.astype(np.float32) @ masks.T.astype(np.float32))[above]
        jaccard = intersection / (2 * k - intersection)
        support = masks.sum(axis=0)
        found["mean_jaccard"][b] = jaccard.mean()
        found["min_jaccard"][b] = jaccard.min()
        found["core_dims"][b] = (support == count).sum()
        found["majority_dims"][b] = ((support * 2 > count) & (support < count)).sum()
        found["minority_dims"][b] = ((support > 0) & (support * 2 <= count)).sum()
        found["union_dims"][b] = (support > 0).sum()
        if same_grade.any():
            found["same_grade_jaccard"][b] = jaccard[same_grade].mean()
        if not same_grade.all():
            found["cross_grade_jaccard"][b] = jaccard[~same_grade].mean()
    helpful, harmful = values > 0, values < 0
    conflict = helpful.any(axis=0) & harmful.any(axis=0)
    found["always_helpful_dims"] = np.array(float(helpful.all(axis=0).sum()))
    found["always_harmful_dims"] = np.array(float(harmful.all(axis=0).sum()))
    found["sign_conflict_dims"] = np.array(float(conflict.sum()))
    found["sign_conflict_share"] = np.array(float(conflict.sum()) / dim)
    correlation = np.atleast_2d(np.corrcoef(average_ranks(values)))[above]
    found["mean_spearman"] = np.array(float(correlation.mean()))
    found["min_spearman"] = np.array(float(correlation.min()))
    if same_grade.any():
        found["same_grade_spearman"] = np.array(float(correlation[same_grade].mean()))
    if not same_grade.all():
        found["cross_grade_spearman"] = np.array(float(correlation[~same_grade].mean()))
    return found


def aggregation_diagnostics(
    query: np.ndarray,
    documents: np.ndarray,
    values: np.ndarray,
    order: np.ndarray,
    grades: np.ndarray,
    budgets: Sequence[int],
) -> dict[str, np.ndarray]:
    """What the query-level weighted mean of ``c`` keeps and loses. Not a ranking comparison: it
    measures only how far the mean's mask is from each positive's own mask and own maximum."""
    count, dim = values.shape
    weights = gain_weights(grades)
    mean = weights @ values
    identity = np.asarray(query, dtype=np.float64) * (
        weights @ np.asarray(documents, dtype=np.float64)
    )
    absolute = weights @ np.abs(values).sum(axis=1)
    within = (weights[:, None] * (values - mean) ** 2).sum()
    between = (mean**2).sum()
    flipped = (np.sign(values) * np.sign(mean) < 0).any(axis=0)
    aggregate_order = contribution_order(mean[None, :])
    cumulative = cumulative_scores(values, order)
    masses = np.abs(values).sum(axis=1)
    found: dict[str, np.ndarray] = {
        "identity_error": np.array(float(np.abs(mean - identity).max())),
        "cancelled_share": np.array(1.0 - float(np.abs(mean).sum() / absolute))
        if absolute > 0
        else np.array(np.nan),
        "within_positive_variance_share": np.array(float(within / (within + between)))
        if within + between > 0
        else np.array(np.nan),
        "flipped_dims": np.array(float(flipped.sum())),
    }
    series: dict[str, list[float]] = {
        name: []
        for name in (
            "mean_overlap", "min_overlap", "mean_regret", "worst_regret", "minority_dropped",
            "conflicted_in_mask",
        )
    }  # fmt: skip
    conflict = (values > 0).any(axis=0) & (values < 0).any(axis=0)
    for k in budgets:
        kept = aggregate_order[0, :k]
        masks = budget_masks(order, k)
        overlap = masks[:, kept].sum(axis=1) / k
        with np.errstate(divide="ignore", invalid="ignore"):
            regret = (cumulative[:, k - 1] - values[:, kept].sum(axis=1)) / masses
        support = masks.sum(axis=0)
        minority = (support > 0) & (support * 2 <= count)
        dropped = np.ones(dim, dtype=bool)
        dropped[kept] = False
        series["mean_overlap"].append(float(weights @ overlap))
        series["min_overlap"].append(float(overlap.min()))
        series["mean_regret"].append(float(weights @ regret))
        series["worst_regret"].append(float(regret.max()))
        series["minority_dropped"].append(
            float((minority & dropped).sum() / minority.sum())
            if count > 1 and minority.any()
            else float("nan")
        )
        series["conflicted_in_mask"].append(float(conflict[kept].sum() / k))
    found.update({name: np.array(items) for name, items in series.items()})
    return found


def split_assignments(queries: int, seed: int, draws: int) -> np.ndarray:
    """``[draws, queries]`` half labels (0/1), balanced per draw, deterministic for a seed; a
    query keeps all of its positives because every accumulator is per query."""
    rng = np.random.default_rng(seed)
    labels = np.zeros((draws, queries), dtype=np.int8)
    for draw in range(draws):
        labels[draw, rng.permutation(queries)[: queries // 2]] = 1
    return labels


def select_queries(query_ids: Iterable[str], seed: int, limit: int | None) -> list[str]:
    """The first ``limit`` ids by a seeded SHA-256 order, returned sorted; every id if no limit.
    A hash makes the selection stable when the id list grows."""
    ordered = sorted(
        query_ids, key=lambda qid: hashlib.sha256(f"{seed}:{qid}".encode()).hexdigest()
    )
    return sorted(ordered if limit is None else ordered[:limit])


def top_dims(profile: np.ndarray, k: int) -> np.ndarray:
    """A profile's top ``k`` dimensions; ties go to the lower index."""
    selected: np.ndarray = contribution_order(profile[None, :])[0, :k]
    return selected


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    """Spearman with the repository's definition; ``None`` for a constant profile."""
    value = spearman(torch.from_numpy(a.astype(np.float64)), torch.from_numpy(b.astype(np.float64)))
    return value if np.isfinite(value) else None


def stratum_label(low: int, high: int | None) -> str:
    return f"{low}+" if high is None else (str(low) if low == high else f"{low}-{high}")


def _stratum(count: int) -> int:
    for index, (low, high) in enumerate(POSITIVE_STRATA):
        if count >= low and (high is None or count <= high):
            return index
    raise ValueError(f"a query needs at least one positive, got {count}")


def _curve_points(budgets: Sequence[int], dim: int) -> list[int]:
    grid = np.unique(np.round(np.geomspace(1, dim, 48)).astype(int)).tolist()
    return sorted({*grid, *budgets, dim})


def _sorted_profile(profile: np.ndarray, points: int = 128) -> list[float]:
    """A descending profile at evenly spaced ranks, enough to draw its shape."""
    ordered = np.sort(profile)[::-1]
    picks = np.linspace(0, ordered.size - 1, points).round().astype(int)
    return [float(value) for value in ordered[picks]]


def analyze_environment(
    query_ids: Sequence[str],
    query_vector: Callable[[str], np.ndarray],
    documents: np.ndarray,
    doc_rows: Mapping[str, int],
    judgments: Judgments,
    budgets: Sequence[int],
    *,
    seed: int,
    stability_draws: int,
    path: Callable[[str], Path],
    device: str = "cpu",
    batch_pairs: int = 8192,
    clock: Clock | None = None,
) -> dict[str, Any]:
    """Run the whole analysis for selected training queries; ``path`` names staged output files.

    Contributions and orderings run in batches on ``device``, once per pair, and go to disk. The
    per-pair statistics and query summaries then read those orderings, on the CPU.

    ``judgments`` must hold only training labels (the CLI guarantees it). Documents are opened read
    only; masks are computed from contributions and applied to no stored matrix.
    """
    started = time.perf_counter()
    dim = documents.shape[1]
    curve_k = _curve_points(budgets, dim)
    labels = {
        qid: sorted((doc_rows[d], d, g) for d, g in judgments[qid].items() if d in doc_rows)
        for qid in query_ids
    }
    positives = {qid: [row for row in found if row[2] > 0] for qid, found in labels.items()}
    empty = [qid for qid, found in positives.items() if not found]
    if empty:
        raise ValueError(
            f"{len(empty)} selected queries have no relevant document, e.g. {empty[0]}"
        )
    n_pairs = sum(len(found) for found in positives.values())
    queries = len(query_ids)
    contribution_file = np.lib.format.open_memmap(
        path("pair-contributions.npy"), mode="w+", dtype=np.float32, shape=(n_pairs, dim)
    )
    order_file = np.lib.format.open_memmap(
        path("pair-order.npy"), mode="w+", dtype=np.uint16, shape=(n_pairs, dim)
    )
    mean_file = np.lib.format.open_memmap(
        path("query-mean-contributions.npy"), mode="w+", dtype=np.float32, shape=(queries, dim)
    )
    pair_columns: dict[str, list[np.ndarray]] = {}
    query_columns: dict[str, list[np.ndarray]] = {}
    nonrelevant_columns: dict[str, list[np.ndarray]] = {}
    nonrelevant_queries = 0
    query_index = np.empty(n_pairs, dtype=np.int64)
    positive_counts = np.empty(queries, dtype=np.int64)
    grades_all = np.empty(n_pairs, dtype=np.int64)
    corpus = _Corpus(budgets, dim, queries, seed, stability_draws)
    clock = clock or Clock(device)
    with clock.span("read"):
        query_rows = np.stack(
            [np.asarray(query_vector(qid), dtype=np.float64) for qid in query_ids]
        )
    write_pair_orders(
        query_rows,
        documents,
        np.array([item[0] for qid in query_ids for item in positives[qid]], dtype=np.int64),
        np.repeat(np.arange(queries), [len(positives[qid]) for qid in query_ids]),
        contribution_file,
        order_file,
        device=device,
        batch_pairs=batch_pairs,
        clock=clock,
    )
    order_seconds = time.perf_counter() - started
    row = 0
    with path("pairs.jsonl").open("w", encoding="utf-8") as pairs_out:
        for index, qid in enumerate(query_ids):
            found = positives[qid]
            rows = [item[0] for item in found]
            grades = np.array([item[2] for item in found], dtype=np.int64)
            q = query_rows[index]
            docs = np.asarray(documents[rows], dtype=np.float64)
            values = contributions(q, docs)
            end = row + len(rows)
            order = np.asarray(order_file[row:end], dtype=np.int64)
            mean_file[index] = values.mean(axis=0)
            query_index[row:end] = index
            grades_all[row:end] = grades
            positive_counts[index] = len(rows)
            for key, value in pair_statistics(q, docs, budgets, curve_k, order).items():
                pair_columns.setdefault(key, []).append(value)
            for key, value in query_agreement(values, order, grades, budgets).items():
                query_columns.setdefault(key, []).append(value)
            for key, value in aggregation_diagnostics(
                q, docs, values, order, grades, budgets
            ).items():
                query_columns.setdefault(f"aggregate_{key}", []).append(value)
            corpus.add(index, order, q, docs)
            for item, doc_row in zip(found, rows, strict=True):
                print(
                    f'{{"query_id": "{qid}", "doc_id": "{item[1]}", "grade": {item[2]}, '
                    f'"document_row": {doc_row}}}',
                    file=pairs_out,
                )
            judged_zero = [item[0] for item in labels[qid] if item[2] == 0]
            if judged_zero:
                nonrelevant_queries += 1
                zero_docs = np.asarray(documents[judged_zero], dtype=np.float64)
                for key, value in pair_statistics(q, zero_docs, budgets, curve_k).items():
                    nonrelevant_columns.setdefault(key, []).append(value)
            row = end
    contribution_file.flush()
    order_file.flush()
    mean_file.flush()
    scoring_seconds = time.perf_counter() - started

    pair_stats = {key: np.concatenate(items) for key, items in pair_columns.items()}
    query_stats = {key: np.stack(items) for key, items in query_columns.items()}
    _save_arrays(
        path("pair-stats.npz"), {"query_index": query_index, "grade": grades_all, **pair_stats}
    )
    _save_arrays(path("query-stats.npz"), {"positives": positive_counts, **query_stats})
    corpus_started = time.perf_counter()
    corpus_summary = corpus.summary(order_file, query_index, mean_file)
    np.save(path("frequency.npy"), corpus.frequency())
    nonrelevant = _nonrelevant(nonrelevant_columns, nonrelevant_queries, queries, path)
    corpus_seconds = time.perf_counter() - corpus_started

    summary: dict[str, Any] = {
        "budgets": list(budgets),
        "chance_jaccard": [chance_jaccard(k, dim) for k in budgets],
        "curve_k": curve_k,
        "coverage": _coverage(labels, query_ids, judgments, doc_rows, positive_counts, grades_all),
        "identities": {
            "max_decomposition_error": float(pair_stats["decomposition_error"].max()),
            "max_aggregation_identity_error": float(query_stats["aggregate_identity_error"].max()),
            "pairs": n_pairs,
        },
        "pairs": _pair_tables(pair_stats, query_index, positive_counts, budgets),
        "curve": _curve(pair_stats["curve"], query_index, positive_counts),
        "agreement": _agreement_tables(query_stats, positive_counts, len(budgets)),
        "grades": _grade_tables(pair_stats, query_index, grades_all, queries),
        "corpus": corpus_summary,
        "finding_stability": _finding_stability(
            pair_stats, query_stats, query_index, corpus.assignments, corpus.budgets, queries
        ),
        "nonrelevant": nonrelevant,
        "cost": {
            "order_seconds": order_seconds,
            "pair_seconds": scoring_seconds,
            "corpus_seconds": corpus_seconds,
            "pairs_per_second": n_pairs / scoring_seconds if scoring_seconds > 0 else None,
        },
    }
    del contribution_file, order_file, mean_file
    return summary


class _Corpus:
    """Query-balanced recurrence: every accumulator adds each query with total weight one."""

    def __init__(
        self, budgets: Sequence[int], dim: int, queries: int, seed: int, draws: int
    ) -> None:
        self.budgets = list(budgets)
        self.dim = dim
        self.queries = queries
        self.seed = seed
        self.draws = draws
        count = len(self.budgets)
        self.balanced = np.zeros((count, dim))
        self.pair_weighted = np.zeros((count, dim))
        self.pairs_seen = 0
        self.assignments = split_assignments(queries, seed, draws)
        self.halves = np.zeros((draws, 2, count, dim))
        self.null_halves = np.zeros((2, count, dim))
        self.strata = np.zeros((len(POSITIVE_STRATA), count, dim))
        self.stratum_queries = np.zeros(len(POSITIVE_STRATA), dtype=np.int64)
        self.cross = np.zeros((count, dim, dim))
        self.pending: list[np.ndarray] = []
        self.mean_abs_query = np.zeros(dim)
        self.mean_abs_document = np.zeros(dim)
        self.null_rng = np.random.default_rng([seed, 1])

    def add(self, index: int, order: np.ndarray, query: np.ndarray, docs: np.ndarray) -> None:
        count = order.shape[0]
        fractions = np.stack([budget_masks(order, k).sum(axis=0) / count for k in self.budgets])
        self.balanced += fractions
        self.pair_weighted += fractions * count
        self.pairs_seen += count
        self.halves[np.arange(self.draws), self.assignments[:, index]] += fractions
        half = self.assignments[0, index] if self.draws else index % 2
        for b in range(len(self.budgets)):
            self.null_halves[half, b] += fractions[b][self.null_rng.permutation(self.dim)]
        stratum = _stratum(count)
        self.strata[stratum] += fractions
        self.stratum_queries[stratum] += 1
        self.mean_abs_query += np.abs(query)
        self.mean_abs_document += np.abs(docs).mean(axis=0)
        self.pending.append(fractions)
        if len(self.pending) >= FLUSH_QUERIES:
            self._flush()

    def _flush(self) -> None:
        if self.pending:
            matrix = np.stack(self.pending)
            for b in range(len(self.budgets)):
                self.cross[b] += matrix[:, b].T @ matrix[:, b]
            self.pending = []

    def frequency(self) -> np.ndarray:
        """``[budgets, D]`` query-balanced selection frequency; each row sums to its budget."""
        result: np.ndarray = self.balanced / self.queries
        return result

    def summary(
        self,
        order_file: np.ndarray,
        query_index: np.ndarray,
        means: np.ndarray,
    ) -> dict[str, Any]:
        self._flush()
        frequency = self.frequency()
        pair_weighted = self.pair_weighted / self.pairs_seen
        marginal = (self.mean_abs_query / self.queries) * (self.mean_abs_document / self.queries)
        per_budget: list[dict[str, Any]] = []
        core_overlap = self._core_overlap(frequency, order_file, query_index)
        for b, k in enumerate(self.budgets):
            profile = frequency[b]
            shares = profile / profile.sum()
            nonzero = shares[shares > 0]
            top = top_dims(profile, k)
            per_budget.append({
                "k": k,
                "chance_frequency": k / self.dim,
                "max_frequency": float(profile.max()),
                "effective_dims": float(np.exp(-(nonzero * np.log(nonzero)).sum())),
                "dims_never_selected": int((profile == 0).sum()),
                "top_k_share": float(profile[top].sum() / k),
                "sorted_profile": _sorted_profile(profile),
                "pair_weighted_spearman": rank_correlation(profile, pair_weighted[b]),
                "marginal_magnitude_spearman": rank_correlation(profile, marginal),
                "core_overlap": distribution(core_overlap[b]),
            })  # fmt: skip
        return {
            "per_budget": per_budget,
            "budget_neighbors": [
                rank_correlation(frequency[i], frequency[i + 1])
                for i in range(len(self.budgets) - 1)
            ],
            "stability": self._stability(means),
            "dimension_contributions": _dimension_table(means),
            "groups": self._groups(frequency),
        }

    def _core_overlap(
        self, frequency: np.ndarray, order_file: np.ndarray, query_index: np.ndarray
    ) -> list[np.ndarray]:
        found = []
        for b, k in enumerate(self.budgets):
            core = np.zeros(self.dim, dtype=bool)
            core[top_dims(frequency[b], k)] = True
            overlap = np.empty(order_file.shape[0])
            for start in range(0, order_file.shape[0], CHUNK_PAIRS):
                block = np.asarray(order_file[start : start + CHUNK_PAIRS, :k], dtype=np.int64)
                overlap[start : start + block.shape[0]] = core[block].mean(axis=1)
            found.append(query_balanced(overlap, query_index, self.queries))
        return found

    def _stability(self, means: np.ndarray) -> dict[str, Any]:
        rows = []
        for b, k in enumerate(self.budgets):
            spearmans, overlaps = [], []
            for draw in range(self.draws):
                a, c = self.halves[draw, 0, b], self.halves[draw, 1, b]
                value = rank_correlation(a, c)
                if value is not None:
                    spearmans.append(value)
                overlaps.append(np.isin(top_dims(a, k), top_dims(c, k)).mean())
            rows.append({
                "k": k,
                "spearman": _range(spearmans),
                "top_k_overlap": _range(overlaps),
                "shuffled_dimension_spearman": rank_correlation(
                    self.null_halves[0, b], self.null_halves[1, b]
                ),
            })  # fmt: skip
        contribution = []
        array = np.asarray(means, dtype=np.float64)
        for draw in range(self.draws):
            half = self.assignments[draw] == 1
            value = rank_correlation(array[half].mean(axis=0), array[~half].mean(axis=0))
            if value is not None:
                contribution.append(value)
        return {
            "draws": self.draws,
            "per_budget": rows,
            "mean_contribution_spearman": _range(contribution),
        }

    def _groups(self, frequency: np.ndarray) -> list[dict[str, Any]]:
        """How far per-query selection departs from one shared profile, at each checkpoint: the
        spectrum of the query-by-dimension selection matrix, and each positive-count stratum
        against the corpus profile."""
        found = []
        for b, k in enumerate(self.budgets):
            mean = frequency[b]
            covariance = self.cross[b] / self.queries - np.outer(mean, mean)
            eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)[::-1]
            total = float(eigenvalues.sum())
            against = []
            for index, (low, high) in enumerate(POSITIVE_STRATA):
                size = int(self.stratum_queries[index])
                value = rank_correlation(self.strata[index, b] / size, mean) if size else None
                against.append({"positives": stratum_label(low, high), "queries": size,
                                "spearman_with_corpus": value})  # fmt: skip
            found.append({
                "k": k,
                "explained_by_components": (
                    eigenvalues[:EXPLAINED_COMPONENTS].cumsum() / total
                ).tolist()
                if total > 0
                else [],
                "strata": against,
            })  # fmt: skip
        return found


def _finding_stability(
    pair_stats: Mapping[str, np.ndarray],
    query_stats: Mapping[str, np.ndarray],
    query_index: np.ndarray,
    assignments: np.ndarray,
    budgets: Sequence[int],
    queries: int,
) -> dict[str, Any]:
    """How far each summary number moves between disjoint query halves, in its own units."""
    values = {
        name: _balanced_finite(pair_stats[name], query_index, queries)
        for name in ("negative_share", "sign_agreement", "effective_positive_dims", "dims_for_90",
                     "k_return")
    }  # fmt: skip
    for b, k in enumerate(budgets):
        values[f"mean_jaccard_k{k}"] = query_stats["mean_jaccard"][:, b]
    values["sign_conflict_share"] = query_stats["sign_conflict_share"]
    values["aggregate_cancelled_share_multi"] = np.where(
        np.isfinite(query_stats["mean_jaccard"][:, 0]),
        query_stats["aggregate_cancelled_share"],
        np.nan,
    )
    found: dict[str, Any] = {}
    for name, column in values.items():
        differences = [
            abs(float(np.nanmean(column[half == 1])) - float(np.nanmean(column[half == 0])))
            for half in assignments
            if np.isfinite(column[half == 1]).any() and np.isfinite(column[half == 0]).any()
        ]
        found[name] = {"overall": _finite_mean(column), **_range(differences)}
    return found


def _save_arrays(target: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with target.open("wb") as handle:
        np.savez(handle, **arrays)  # type: ignore[arg-type]


def _range(values: Sequence[float]) -> dict[str, float | None]:
    if not len(values):
        return {"mean": None, "min": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "min": float(array.min()), "max": float(array.max())}


def _dimension_table(means: np.ndarray) -> dict[str, Any]:
    """Per dimension over queries: how each dimension's query-mean contribution is distributed.
    The per-dimension rows stay in ``query-mean-contributions.npy``; this keeps the shape."""
    array = np.asarray(means, dtype=np.float64)
    medians = np.median(array, axis=0)
    p10, p90 = np.quantile(array, [0.1, 0.9], axis=0)
    ranks = np.argsort(-medians, kind="stable")
    points = np.linspace(0, array.shape[1] - 1, 128).round().astype(int)
    return {
        "rank": points.tolist(),
        "p10": p10[ranks][points].tolist(),
        "p50": medians[ranks][points].tolist(),
        "p90": p90[ranks][points].tolist(),
        "dims_median_positive": int((medians > 0).sum()),
        "dims_spanning_zero": int(((p10 < 0) & (p90 > 0)).sum()),
        "mean_positive_query_rate": float((array > 0).mean()),
    }


def _pair_tables(
    stats: Mapping[str, np.ndarray],
    query_index: np.ndarray,
    positive_counts: np.ndarray,
    budgets: Sequence[int],
) -> dict[str, Any]:
    """Every per-pair statistic, query-balanced for all, single-positive, and multi-positive
    queries; the pair-weighted mean is kept apart and labeled."""
    queries = positive_counts.size
    subsets = {"all": np.ones(queries, dtype=bool), "single": positive_counts == 1,
               "multi": positive_counts > 1}  # fmt: skip
    scalar: dict[str, Any] = {}
    per_budget: dict[str, Any] = {}
    for name, values in stats.items():
        if name == "curve":
            continue
        balanced = _balanced_finite(values, query_index, queries)
        if values.ndim == 1:
            scalar[name] = {
                subset: distribution(balanced[mask]) for subset, mask in subsets.items()
            } | {"pair_weighted_mean": _finite_mean(values)}
        else:
            per_budget[name] = {
                subset: [distribution(balanced[mask, b]) for b in range(len(budgets))]
                for subset, mask in subsets.items()
            } | {"pair_weighted_mean": [_finite_mean(values[:, b]) for b in range(len(budgets))]}
    return {"scalar": scalar, "per_budget": per_budget}


def _balanced_finite(values: np.ndarray, query_index: np.ndarray, queries: int) -> np.ndarray:
    """Per-query means of the finite entries; a query with none gives NaN."""
    finite = np.isfinite(values)
    totals = np.zeros((queries, *values.shape[1:]))
    counts = np.zeros((queries, *values.shape[1:]))
    np.add.at(totals, query_index, np.where(finite, values, 0.0))
    np.add.at(counts, query_index, finite.astype(np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        result: np.ndarray = totals / counts
    return result


def _finite_mean(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else None


def _curve(curves: np.ndarray, query_index: np.ndarray, counts: np.ndarray) -> dict[str, Any]:
    balanced = _balanced_finite(curves, query_index, counts.size)
    p10, p50, p90 = np.nanquantile(balanced, [0.1, 0.5, 0.9], axis=0)
    return {"p10": p10.tolist(), "p50": p50.tolist(), "p90": p90.tolist()}


def _agreement_tables(
    stats: Mapping[str, np.ndarray], positive_counts: np.ndarray, budgets: int
) -> dict[str, Any]:
    """Multi-positive queries only, overall and by positive-count stratum, for agreement measures;
    the aggregation diagnostics that also exist for single positives keep that subset apart."""
    multi = positive_counts > 1
    subsets = {"multi": multi, "single": positive_counts == 1}
    for low, high in POSITIVE_STRATA[1:]:
        mask = positive_counts >= low
        if high is not None:
            mask = mask & (positive_counts <= high)
        subsets[f"positives {stratum_label(low, high)}"] = mask
    found: dict[str, Any] = {}
    for name, values in stats.items():
        if values.ndim == 1:
            found[name] = {s: distribution(values[mask]) for s, mask in subsets.items()}
        else:
            found[name] = {
                s: [distribution(values[mask, b]) for b in range(budgets)]
                for s, mask in subsets.items()
            }
    return found


def _grade_tables(
    stats: Mapping[str, np.ndarray], query_index: np.ndarray, grades: np.ndarray, queries: int
) -> list[dict[str, Any]]:
    """Per relevance grade: each query counts once within the grade."""
    found = []
    for grade in sorted(set(grades.tolist())):
        chosen = grades == grade
        row: dict[str, Any] = {"grade": grade, "pairs": int(chosen.sum()),
                               "queries": int(np.unique(query_index[chosen]).size)}  # fmt: skip
        for name in ("score", "positive_mass", "negative_share", "effective_positive_dims",
                     "dims_for_90"):  # fmt: skip
            balanced = _balanced_finite(stats[name][chosen], query_index[chosen], queries)
            row[name] = distribution(balanced)
        found.append(row)
    return found


def _coverage(
    labels: Mapping[str, Sequence[tuple[int, str, int]]],
    query_ids: Sequence[str],
    judgments: Judgments,
    doc_rows: Mapping[str, int],
    positive_counts: np.ndarray,
    grades: np.ndarray,
) -> dict[str, Any]:
    judged = sum(len(judgments[qid]) for qid in query_ids)
    present = sum(len(labels[qid]) for qid in query_ids)
    zero = sum(1 for qid in query_ids for item in labels[qid] if item[2] == 0)
    strata = []
    for low, high in POSITIVE_STRATA:
        mask = positive_counts >= low
        if high is not None:
            mask = mask & (positive_counts <= high)
        strata.append({"positives": stratum_label(low, high), "queries": int(mask.sum()),
                       "pairs": int(positive_counts[mask].sum())})  # fmt: skip
    values, counts = np.unique(grades, return_counts=True)
    return {
        "selected_queries": len(query_ids),
        "judgments": judged,
        "judgments_without_document": judged - present,
        "relevant_pairs": int(positive_counts.sum()),
        "judged_nonrelevant_pairs": zero,
        "corpus_documents": len(doc_rows),
        "mean_unjudged_per_query": len(doc_rows) - present / max(len(query_ids), 1),
        "single_positive_queries": int((positive_counts == 1).sum()),
        "multi_positive_queries": int((positive_counts > 1).sum()),
        "positives_per_query": distribution(positive_counts.astype(np.float64)),
        "max_positives_per_query": int(positive_counts.max()),
        "strata": strata,
        "grades": {str(int(v)): int(c) for v, c in zip(values, counts, strict=True)},
    }


def _nonrelevant(
    columns: Mapping[str, list[np.ndarray]],
    with_judged: int,
    queries: int,
    path: Callable[[str], Path],
) -> dict[str, Any]:
    """Judged nonrelevant pairs only, never unjudged documents; these rows are not balanced against
    the positives or compared by a ranking measure, and are absent when grade 0 is absent."""
    found: dict[str, Any] = {"queries_with_judged_nonrelevant": with_judged,
                             "coverage": with_judged / queries if queries else 0.0}  # fmt: skip
    if not columns:
        return {**found, "pairs": 0, "statistics": {}}
    stats = {key: np.concatenate(items) for key, items in columns.items()}
    _save_arrays(path("nonrelevant-stats.npz"), stats)
    names = ("score", "positive_mass", "negative_mass", "negative_share", "sign_agreement",
             "effective_positive_dims", "top_10_share", "dims_for_90")  # fmt: skip
    return {**found, "pairs": int(stats["score"].size),
            "statistics": {name: distribution(stats[name]) for name in names}}  # fmt: skip
