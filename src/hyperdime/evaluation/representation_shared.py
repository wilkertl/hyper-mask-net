"""What one shared query mask costs its relevant documents, and how it transfers among them.

For a query and a relevant document, ``c_j = q_j * d_j``. The document's own top ``k`` gives the
largest score any ``k`` dimensions can give it. The top ``k`` of the weighted mean of a group's
contributions gives the largest weighted mean score, not each document's largest. Regret is a
document's own optimum minus its score under the shared mask, at the same ``k`` (ADR-0015).

Regret describes a compromise in similarity. It does not measure a ranking, and it does not say a
better ranking mask is impossible. Training judgments only; held-out documents here are documents of
a training query, not validation or test queries, and they say nothing about unseen queries.

One ordering per document or mean, one cumulative sum per score: every ``k`` comes from these, and
no mask is built.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from hyperdime.evaluation.representation import (
    POSITIVE_STRATA,
    gain_weights,
    relevance_gains,
    stratum_label,
)
from hyperdime.evaluation.representation_controls import (
    _curve_distribution,
    _repeat_band,
    query_stream,
    reference_rows,
    row_spearman,
)
from hyperdime.evaluation.representation_engine import (
    Clock,
    gather,
    jaccard_curves,
    ranks_of,
    stable_order,
)

FIXED_COUNTS = (2, 5, 20)
FIXED_DRAWS = 8
TRANSFER_CELLS: tuple[tuple[str, int | None, int | None], ...] = (
    ("1-1", 1, 1),
    ("2-2", 2, 2),
    ("5-5", 5, 5),
    ("10-10", 10, 10),
    ("20-20", 20, 20),
    ("half", None, None),
)
PARTITIONS = 8
TINY_REGRET = 1e-12
OVERLAP_BINS = (0.0, 0.25, 0.5, 0.75, 1.0)
WITHIN_QUERY_MIN_POSITIVES = 5
SHARED_MEASURES = (
    "mean_raw", "mean_norm", "p90_norm", "max_raw", "max_norm", "std_norm", "overlap_mean",
    "overlap_min", "worst_share", "top20_share", "random_mean_norm",
)  # fmt: skip
FIXED_MEASURES = ("mean_raw", "mean_norm", "max_raw", "max_norm", "worst_share")
CELL_MEASURES = (
    "held_mean_raw", "held_mean_norm", "held_max_norm", "construct_mean_norm",
    "construct_max_norm", "privileged_mean_norm", "reference_mean_norm", "reference_max_norm",
    "held_overlap",
)  # fmt: skip


def shared_order(values: Tensor, gains: Tensor) -> Tensor:
    """``[masks, D]`` orderings of each group's weighted mean; values ``[masks, docs, D]``, gains
    ``[masks, docs]``. Its top ``k`` maximizes the weighted mean score at that ``k``.

    The weighted sum orders dimensions exactly as the mean does, and with whole-number gains on
    products of float16 values it has no rounding. Normalized weights such as 1/5 round, and a
    device's order of summation then breaks exact ties differently; a tie in the mean can hide
    different contributions per document, so that would change scores. Ties go to the lower index.
    """
    return stable_order((gains[:, :, None] * values).sum(dim=1))


def scores_under(values: Tensor, order: Tensor) -> Tensor:
    """``[masks, docs, D]`` scores along each mask's ordering, every ``k`` at once."""
    index = order[:, None, :].expand(-1, values.shape[1], -1)
    return values.gather(2, index).cumsum(dim=2)


def optimum_curves(values: Tensor) -> tuple[Tensor, Tensor]:
    """``([docs, D] each document's best score at each k, [docs, D] its dimension ranks)``."""
    order = stable_order(values)
    return values.gather(1, order).cumsum(dim=1), ranks_of(order)


def masses(values: Tensor) -> Tensor:
    """``sum |c_j|`` per document; a document with no mass gets NaN, so its ratios stay undefined
    and never reach a mean, instead of dividing by a score that may be zero or negative."""
    found = values.abs().sum(dim=-1)
    return torch.where(found > 0, found, torch.full_like(found, torch.nan))


def overlap_curves(ranks: Tensor, order: Tensor) -> Tensor:
    """``[masks, docs, D]`` Jaccard of each document's own mask with the shared one at every k."""
    masks, docs, dim = ranks.shape
    mask_ranks = ranks_of(order)[:, None, :].expand(-1, docs, -1)
    found = jaccard_curves(ranks.reshape(-1, dim), mask_ranks.reshape(-1, dim))
    return found.view(masks, docs, dim)


def nan_max(values: Tensor, dim: int) -> Tensor:
    """Largest finite value along ``dim``; NaN only where all are undefined."""
    filled = torch.nan_to_num(values, nan=-math.inf)
    found = filled.max(dim=dim).values
    return torch.where(found == -math.inf, torch.full_like(found, torch.nan), found)


def nan_mean(values: Tensor, dim: int) -> Tensor:
    return values.nanmean(dim=dim)


def regret_shares(raw: Tensor) -> tuple[Tensor, Tensor]:
    """``[D]`` share of a group's regret on its worst document and its worst fifth; rows ``docs``.
    Undefined where the group has essentially no regret, as at ``k = D``."""
    total = raw.clamp(min=0).sum(dim=0)
    ordered = raw.clamp(min=0).sort(dim=0, descending=True).values
    top = math.ceil(0.2 * raw.shape[0])
    defined = total > TINY_REGRET
    undefined = torch.full_like(total, torch.nan)
    worst = torch.where(defined, ordered[0] / total, undefined)
    fifth = torch.where(defined, ordered[:top].sum(dim=0) / total, undefined)
    return worst, fifth


def group_summary(raw: Tensor, norm: Tensor, overlap: Tensor) -> dict[str, Tensor]:
    """What a query's positives see under one mask, each ``[D]``: one number per ``k``."""
    deviation = norm - nan_mean(norm, 0)
    seen = norm.isfinite().sum(dim=0).to(norm.dtype)
    variance = torch.nansum(deviation**2, dim=0) / seen
    worst, fifth = regret_shares(raw)
    return {
        "mean_raw": raw.mean(dim=0),
        "mean_norm": nan_mean(norm, 0),
        "p90_norm": torch.nanquantile(norm, 0.9, dim=0),
        "max_raw": raw.max(dim=0).values,
        "max_norm": nan_max(norm, 0),
        "std_norm": variance.sqrt(),
        "overlap_mean": overlap.mean(dim=0),
        "overlap_min": overlap.min(dim=0).values,
        "worst_share": worst,
        "top20_share": fifth,
    }


def within_query_correlation(norm: Tensor, overlap: Tensor) -> Tensor:
    """``[D]`` Pearson of regret and overlap across one query's positives; NaN if either is flat."""
    a = norm - norm.mean(dim=0)
    b = overlap - overlap.mean(dim=0)
    denominator = (a.pow(2).sum(dim=0) * b.pow(2).sum(dim=0)).sqrt()
    found = (a * b).sum(dim=0) / denominator
    return torch.where(denominator > 0, found, torch.full_like(found, torch.nan))


def partitions(
    count: int, construct: int, evaluate: int, rng: np.random.Generator, repeats: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """``(construction, evaluation)`` positions, disjoint and deterministic: every ordered split
    when there are at most ``repeats`` of them, else ``repeats`` seeded draws."""
    if construct + evaluate > count:
        raise ValueError(f"{construct} + {evaluate} documents asked of a group of {count}")
    total = math.comb(count, construct) * math.comb(count - construct, evaluate)
    if total <= repeats:
        found = []
        for chosen in combinations(range(count), construct):
            rest = [i for i in range(count) if i not in chosen]
            for held in combinations(rest, evaluate):
                found.append((np.array(chosen), np.array(held)))
        return found
    shuffles = [rng.permutation(count) for _ in range(repeats)]
    return [(p[:construct], p[construct : construct + evaluate]) for p in shuffles]


def cell_sizes(count: int, construct: int | None, evaluate: int | None) -> tuple[int, int] | None:
    """Sizes of one transfer cell for a query with ``count`` positives; ``None`` if ineligible.
    ``None`` sizes are the half split: as many as fit, with the odd document held out."""
    if construct is None or evaluate is None:
        return (count // 2, count - count // 2) if count >= 2 else None
    return (construct, evaluate) if construct + evaluate <= count else None


def uniform_mask_regret(optimum: Tensor, values: Tensor) -> Tensor:
    """Regret under a uniformly random ``k``-subset, in expectation: such a mask keeps ``k / D`` of
    the full score. A reference with no structure, computed exactly and not sampled."""
    dim = values.shape[-1]
    k = torch.arange(1, dim + 1, device=values.device, dtype=values.dtype) / dim
    return optimum - values.sum(dim=-1, keepdim=True) * k


@dataclass
class TransferResult:
    """One query in one cell, averaged over its partitions, each measure ``[D]``."""

    measures: dict[str, Tensor]
    per_partition: Tensor  # [partitions, D] held-out mean normalized regret
    partitions: int
    between_partition_sd: Tensor


def transfer_cell(
    values: Tensor,
    optimum: Tensor,
    mass: Tensor,
    ranks: Tensor,
    grades: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    privileged: Tensor,
    reference_values: Tensor,
) -> TransferResult:
    """Build one mask per split from its construction documents only, and score both sides.

    ``privileged`` is ``[docs, D]`` normalized regret under the mask of all positives, kept only as
    a labeled reference. ``reference_values`` is ``[splits, construct, D]``, contributions of
    unjudged documents with this query: a yardstick for what the shared query alone transfers.
    """
    device, dtype = values.device, values.dtype
    construct = torch.from_numpy(np.stack([c for c, _ in splits])).to(device)
    held = torch.from_numpy(np.stack([h for _, h in splits])).to(device)
    # Only construction grades reach the gains, and only construction documents reach the mean.
    gains = torch.from_numpy(np.stack([relevance_gains(grades[c]) for c, _ in splits]))
    order = shared_order(values[construct], gains.to(device, dtype))
    held_values = values[held]
    held_raw = optimum[held] - scores_under(held_values, order)
    held_norm = held_raw / mass[held][:, :, None]
    construct_raw = optimum[construct] - scores_under(values[construct], order)
    construct_norm = construct_raw / mass[construct][:, :, None]
    equal = torch.ones(reference_values.shape[:2], device=device, dtype=dtype)
    reference_order = shared_order(reference_values, equal)
    reference_raw = optimum[held] - scores_under(held_values, reference_order)
    reference_norm = reference_raw / mass[held][:, :, None]
    per_partition = nan_mean(held_norm, 1)
    return TransferResult(
        {
            "held_mean_raw": held_raw.mean(dim=(0, 1)),
            "held_mean_norm": nan_mean(per_partition, 0),
            "held_max_norm": nan_mean(nan_max(held_norm, 1), 0),
            "construct_mean_norm": nan_mean(nan_mean(construct_norm, 1), 0),
            "construct_max_norm": nan_mean(nan_max(construct_norm, 1), 0),
            "privileged_mean_norm": nan_mean(nan_mean(privileged[held], 1), 0),
            "reference_mean_norm": nan_mean(nan_mean(reference_norm, 1), 0),
            "reference_max_norm": nan_mean(nan_max(reference_norm, 1), 0),
            "held_overlap": overlap_curves(ranks[held], order).mean(dim=(0, 1)),
        },
        per_partition,
        len(splits),
        per_partition.std(dim=0, unbiased=False),
    )


def fixed_count(
    values: Tensor, optimum: Tensor, mass: Tensor, grades: np.ndarray, draws: np.ndarray
) -> dict[str, Tensor]:
    """A mask of each drawn group, ``draws`` ``[draws, m]``, scored on that group: a query's worst
    positive then always means the worst of ``m``, and queries can be compared."""
    device, dtype = values.device, values.dtype
    index = torch.from_numpy(draws).to(device)
    gains = torch.from_numpy(np.stack([relevance_gains(grades[d]) for d in draws]))
    group = values[index]
    raw = optimum[index] - scores_under(group, shared_order(group, gains.to(device, dtype)))
    norm = raw / mass[index][:, :, None]
    positive = raw.clamp(min=0)
    total = positive.sum(dim=1)
    worst = positive.max(dim=1).values / total
    worst = torch.where(total > TINY_REGRET, worst, torch.full_like(worst, torch.nan))
    return {
        "mean_raw": raw.mean(dim=(0, 1)),
        "mean_norm": nan_mean(nan_mean(norm, 1), 0),
        "max_raw": raw.max(dim=1).values.mean(dim=0),
        "max_norm": nan_mean(nan_max(norm, 1), 0),
        "worst_share": nan_mean(worst, 0),
    }


@dataclass
class _Checks:
    """Identities a correct run holds to rounding; recorded, and reported with the results."""

    lowest_raw_regret: float = 0.0
    largest_full_k_regret: float = 0.0
    largest_mean_identity_error: float = 0.0
    zero_mass_documents: int = 0
    orderings_differing_from_canonical: int | None = None
    documents_compared_with_canonical: int = 0

    def update(self, raw: Tensor, mean_identity: float) -> None:
        self.lowest_raw_regret = min(self.lowest_raw_regret, raw.min().item())
        self.largest_full_k_regret = max(self.largest_full_k_regret, raw[:, -1].abs().max().item())
        self.largest_mean_identity_error = max(self.largest_mean_identity_error, mean_identity)


@dataclass
class _Accumulators:
    """Pair-weighted totals, kept apart from the query-balanced files."""

    dim: int
    totals: dict[str, np.ndarray] = field(default_factory=dict)
    counts: dict[str, np.ndarray] = field(default_factory=dict)

    def add(self, name: str, values: Tensor) -> None:
        seen = values.isfinite()
        total = torch.where(seen, values, torch.zeros_like(values)).sum(dim=0).cpu().numpy()
        self.totals[name] = self.totals.get(name, np.zeros(self.dim)) + total
        self.counts[name] = (
            self.counts.get(name, np.zeros(self.dim)) + seen.sum(dim=0).cpu().numpy()
        )

    def means(self) -> dict[str, list[float]]:
        with np.errstate(divide="ignore", invalid="ignore"):
            return {
                name: (total / self.counts[name]).tolist() for name, total in self.totals.items()
            }


def _store(path: Callable[[str], Path], name: str, rows: int, dim: int) -> Any:
    return np.lib.format.open_memmap(path(name), mode="w+", dtype=np.float32, shape=(rows, dim))


def analyze_shared_masks(
    query_ids: Sequence[str],
    query_vector: Callable[[str], np.ndarray],
    documents: np.ndarray,
    positives: Mapping[str, np.ndarray],
    grades: Mapping[str, np.ndarray],
    judged: Mapping[str, set[int]],
    *,
    seed: int,
    device: str,
    clock: Clock,
    path: Callable[[str], Path],
    canonical_order: Callable[[str], np.ndarray | None] | None = None,
) -> dict[str, Any]:
    """Every selected multi-positive query, one at a time; memory is bounded by its largest group.

    ``positives[q]`` holds a query's relevant document rows, ascending, with ``grades[q]`` in step.
    ``judged`` excludes all of a query's judged documents from reference draws.
    """
    started = time.perf_counter()
    corpus, dim = documents.shape
    queries = len(query_ids)
    counts = np.array([positives[q].size for q in query_ids], dtype=np.int64)
    if (counts < 2).any():
        raise ValueError("this analysis takes only queries with at least two positives")
    shared_files = {m: _store(path, f"shared-{m}.npy", queries, dim) for m in SHARED_MEASURES}
    fixed_rows = {m: np.cumsum(counts >= m) - 1 for m in FIXED_COUNTS}
    fixed_files = {
        (m, name): _store(path, f"fixed{m}-{name}.npy", int((counts >= m).sum()), dim)
        for m in FIXED_COUNTS
        for name in FIXED_MEASURES
    }
    cell_eligible = {
        name: np.array([cell_sizes(int(c), a, h) is not None for c in counts])
        for name, a, h in TRANSFER_CELLS
    }
    cell_rows = {name: np.cumsum(mask) - 1 for name, mask in cell_eligible.items()}
    cell_files = {
        (name, measure): _store(path, f"transfer{name}-{measure}.npy", int(mask.sum()), dim)
        for name, mask in cell_eligible.items()
        for measure in CELL_MEASURES
    }
    repeat_totals = {name: np.zeros((PARTITIONS, dim)) for name, _, _ in TRANSFER_CELLS}
    repeat_counts = {name: np.zeros((PARTITIONS, dim)) for name, _, _ in TRANSFER_CELLS}
    sd_totals = {name: np.zeros(dim) for name, _, _ in TRANSFER_CELLS}
    sd_counts = {name: np.zeros(dim) for name, _, _ in TRANSFER_CELLS}
    partition_counts = {name: 0 for name, _, _ in TRANSFER_CELLS}
    pairs = _Accumulators(dim)
    bin_totals = np.zeros((dim, len(OVERLAP_BINS) - 1))
    bin_counts = np.zeros((dim, len(OVERLAP_BINS) - 1))
    within_total, within_count = np.zeros(dim), np.zeros(dim)
    checks = _Checks()
    if canonical_order is not None:
        checks.orderings_differing_from_canonical = 0
    bin_edges = torch.tensor(OVERLAP_BINS, device=device, dtype=torch.float64)

    for index, qid in enumerate(query_ids):
        rows = positives[qid]
        count = rows.size
        with clock.span("read"):
            docs_np = gather(documents, rows)
            q_np = np.asarray(query_vector(qid), dtype=np.float64)
            grade_np = np.asarray(grades[qid])
            plans = []
            for position, (name, a, h) in enumerate(TRANSFER_CELLS):
                sizes = cell_sizes(count, a, h)
                if sizes is None:
                    continue
                rng = query_stream(seed, 400 + position, qid)
                splits = partitions(count, sizes[0], sizes[1], rng, PARTITIONS)
                drawn = np.stack([
                    reference_rows(
                        corpus, judged[qid], sizes[0],
                        query_stream(seed, 500 + position * 16 + slot, qid),
                    )
                    for slot in range(len(splits))
                ])  # fmt: skip
                plans.append((name, splits, gather(documents, drawn.ravel())))
            draw_plans = []
            for m in FIXED_COUNTS:
                if count >= m:
                    rng = query_stream(seed, 450 + m, qid)
                    draws = (
                        np.arange(m)[None]
                        if count == m
                        else np.stack([rng.permutation(count)[:m] for _ in range(FIXED_DRAWS)])
                    )
                    draw_plans.append((m, draws))
        with clock.span("transfer"):
            docs = torch.from_numpy(docs_np).to(device)
            q = torch.from_numpy(q_np).to(device)
            reference_on = [
                (name, splits, torch.from_numpy(ref).to(device)) for name, splits, ref in plans
            ]
        with clock.span("compute"):
            values = q * docs
            optimum, ranks = optimum_curves(values)
            mass = masses(values)
            checks.zero_mass_documents += int(mass.isnan().sum().item())
            weights = torch.from_numpy(gain_weights(grade_np)).to(device)
            gains = torch.from_numpy(relevance_gains(grade_np)).to(device)
            order = shared_order(values[None], gains[None])
            scores = scores_under(values[None], order)[0]
            raw = optimum - scores
            norm = raw / mass[:, None]
            overlap = overlap_curves(ranks[None], order)[0]
            # Identity: the mean's top k gives exactly the weighted mean score under that mask.
            mean_curve = ((weights[:, None] * values).sum(dim=0))[order[0]].cumsum(dim=0)
            identity = (mean_curve - (weights[:, None] * scores).sum(dim=0)).abs().max().item()
            checks.update(raw, identity)
            summary = group_summary(raw, norm, overlap)
            summary["random_mean_norm"] = nan_mean(
                uniform_mask_regret(optimum, values) / mass[:, None], 0
            )
            fixed = [
                (m, fixed_count(values, optimum, mass, grade_np, draws)) for m, draws in draw_plans
            ]
            transfers = []
            for name, splits, unjudged in reference_on:
                construct_size = splits[0][0].size
                reference_values = unjudged.view(len(splits), construct_size, dim) * q
                transfers.append((name, transfer_cell(
                    values, optimum, mass, ranks, grade_np, splits, norm, reference_values
                )))  # fmt: skip
            pairs.add("mean_raw", raw)
            pairs.add("mean_norm", norm)
            bins = (torch.bucketize(overlap, bin_edges, right=True) - 1).clamp(0, 3)
            bin_sum = torch.zeros((dim, 4), device=device, dtype=torch.float64)
            bin_seen = torch.zeros((dim, 4), device=device, dtype=torch.float64)
            finite = norm.isfinite()
            bin_sum.scatter_add_(1, bins.T, torch.where(finite, norm, 0.0 * raw).T)
            bin_seen.scatter_add_(1, bins.T, finite.T.to(torch.float64))
            within = (
                within_query_correlation(norm, overlap)
                if count >= WITHIN_QUERY_MIN_POSITIVES
                else None
            )
        with clock.span("transfer"):
            shared_out = {k: v.to(torch.float32).cpu().numpy() for k, v in summary.items()}
            fixed_out = [
                (m, {k: v.to(torch.float32).cpu().numpy() for k, v in found.items()})
                for m, found in fixed
            ]
            transfer_out = [
                (
                    name,
                    {k: v.to(torch.float32).cpu().numpy() for k, v in result.measures.items()},
                    result.per_partition.cpu().numpy(),
                    result.partitions,
                    result.between_partition_sd.cpu().numpy(),
                )
                for name, result in transfers
            ]
            bin_totals += bin_sum.cpu().numpy()
            bin_counts += bin_seen.cpu().numpy()
            if within is not None:
                found = within.cpu().numpy()
                seen = np.isfinite(found)
                within_total += np.where(seen, found, 0.0)
                within_count += seen
            order_np = ranks.cpu().numpy() if canonical_order is not None else None
        with clock.span("write"):
            for name, values_out in shared_out.items():
                shared_files[name][index] = values_out
            for m, found_out in fixed_out:
                for name, values_out in found_out.items():
                    fixed_files[m, name][fixed_rows[m][index]] = values_out
            for name, measures, per_partition, splits_used, sd in transfer_out:
                row = cell_rows[name][index]
                partition_counts[name] += splits_used
                for measure, values_out in measures.items():
                    cell_files[name, measure][row] = values_out
                for slot in range(PARTITIONS):
                    current = per_partition[slot % splits_used]
                    seen = np.isfinite(current)
                    repeat_totals[name][slot] += np.where(seen, current, 0.0)
                    repeat_counts[name][slot] += seen
                seen = np.isfinite(sd)
                sd_totals[name] += np.where(seen, sd, 0.0)
                sd_counts[name] += seen
        if canonical_order is not None and order_np is not None:
            found_order = canonical_order(qid)
            if found_order is not None:
                checks.documents_compared_with_canonical += count
                assert checks.orderings_differing_from_canonical is not None
                recomputed = np.argsort(order_np, axis=1)
                checks.orderings_differing_from_canonical += int(
                    (recomputed != np.asarray(found_order, dtype=np.int64)).any(axis=1).sum()
                )
    with clock.span("write"):
        for mapped in (*shared_files.values(), *fixed_files.values(), *cell_files.values()):
            mapped.flush()
    pass_seconds = time.perf_counter() - started
    summary_started = time.perf_counter()
    found_summary = _summaries(
        counts, shared_files, fixed_files, cell_files, cell_eligible,
        repeat_totals, repeat_counts, sd_totals, sd_counts, partition_counts,
        pairs, bin_totals, bin_counts, within_total, within_count, dim,
    )  # fmt: skip
    found_summary["k"] = list(range(1, dim + 1))
    found_summary["checks"] = vars(checks)
    found_summary["settings"] = {
        "fixed_counts": list(FIXED_COUNTS), "fixed_draws": FIXED_DRAWS, "partitions": PARTITIONS,
        "cells": [name for name, _, _ in TRANSFER_CELLS], "overlap_bins": list(OVERLAP_BINS),
    }  # fmt: skip
    found_summary["cost"] = {
        "pass_seconds": pass_seconds,
        "summary_seconds": time.perf_counter() - summary_started,
    }
    return found_summary


def strata_of(counts: np.ndarray) -> dict[str, np.ndarray]:
    """All selected queries, and each positive-count stratum above one positive."""
    found = {"multi": np.ones(counts.size, dtype=bool)}
    for low, high in POSITIVE_STRATA[1:]:
        mask = counts >= low
        if high is not None:
            mask = mask & (counts <= high)
        found[f"positives {stratum_label(low, high)}"] = mask
    return found


def _distribution(matrix: np.ndarray) -> dict[str, Any]:
    return _curve_distribution(np.asarray(matrix, dtype=np.float64))


def _summaries(
    counts: np.ndarray,
    shared_files: Mapping[str, np.ndarray],
    fixed_files: Mapping[tuple[int, str], np.ndarray],
    cell_files: Mapping[tuple[str, str], np.ndarray],
    cell_eligible: Mapping[str, np.ndarray],
    repeat_totals: Mapping[str, np.ndarray],
    repeat_counts: Mapping[str, np.ndarray],
    sd_totals: Mapping[str, np.ndarray],
    sd_counts: Mapping[str, np.ndarray],
    partition_counts: Mapping[str, int],
    pairs: _Accumulators,
    bin_totals: np.ndarray,
    bin_counts: np.ndarray,
    within_total: np.ndarray,
    within_count: np.ndarray,
    dim: int,
) -> dict[str, Any]:
    strata = strata_of(counts)
    shared = {
        name: {subset: _distribution(np.asarray(matrix)[mask]) for subset, mask in strata.items()}
        for name, matrix in shared_files.items()
    }
    fixed: dict[str, Any] = {}
    for (m, name), matrix in fixed_files.items():
        fixed.setdefault(str(m), {"queries": int(matrix.shape[0])})[name] = _distribution(matrix)
    cells: dict[str, Any] = {}
    for name, mask in cell_eligible.items():
        own_counts = counts[mask]
        held = np.asarray(cell_files[name, "held_mean_norm"], dtype=np.float64)
        cell: dict[str, Any] = {
            "queries": int(mask.sum()),
            "positives": {
                "min": int(own_counts.min()) if own_counts.size else None,
                "max": int(own_counts.max()) if own_counts.size else None,
            },
            "partitions_evaluated": partition_counts[name],
            "repeat_band": _repeat_band(repeat_totals[name], repeat_counts[name]),
        }
        with np.errstate(divide="ignore", invalid="ignore"):
            cell["between_partition_sd_mean"] = (sd_totals[name] / sd_counts[name]).tolist()
        for measure in CELL_MEASURES:
            cell[measure] = _distribution(cell_files[name, measure])
        for label, other in (
            ("held_minus_construct", "construct_mean_norm"),
            ("held_minus_privileged", "privileged_mean_norm"),
            ("held_minus_reference", "reference_mean_norm"),
        ):
            cell[label] = _distribution(
                held - np.asarray(cell_files[name, other], dtype=np.float64)
            )
        if name == "half":
            cell["strata"] = {
                subset: {
                    "held_mean_norm": _distribution(
                        np.asarray(cell_files[name, "held_mean_norm"])[own[mask]]
                    ),
                    "construct_mean_norm": _distribution(
                        np.asarray(cell_files[name, "construct_mean_norm"])[own[mask]]
                    ),
                    "reference_mean_norm": _distribution(
                        np.asarray(cell_files[name, "reference_mean_norm"])[own[mask]]
                    ),
                }
                for subset, own in strata.items()
            }
        cells[name] = cell
    with np.errstate(divide="ignore", invalid="ignore"):
        bins = (bin_totals / bin_counts).T
    between = row_spearman(
        np.asarray(shared_files["overlap_mean"], dtype=np.float64).T,
        np.asarray(shared_files["mean_norm"], dtype=np.float64).T,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        within = within_total / within_count
    return {
        "coverage": {
            "queries": int(counts.size),
            "pairs": int(counts.sum()),
            "strata": {name: int(mask.sum()) for name, mask in strata.items()},
            "largest_group": int(counts.max()),
        },
        "shared": shared,
        "pair_weighted": pairs.means(),
        "fixed_count": fixed,
        "transfer": cells,
        "overlap": {
            "between_query_spearman": between.tolist(),
            "within_query_pearson_mean": within.tolist(),
            "within_query_pearson_queries": within_count.tolist(),
            "regret_by_overlap_bin": bins.tolist(),
            "pairs_by_overlap_bin": bin_counts.T.tolist(),
        },
    }
