"""Full-corpus rankings under the masks of ADR-0015: a privileged diagnostic (ADR-0016).

A mask that raises a relevant document's similarity need not raise its rank: other documents may
rise further. This module scores every document of the corpus under each mask, exactly, and keeps
only retrieved lists and rank counters.

Float16 values are multiples of 2^-24, so products are multiples of 2^-48; every partial sum of an
inner product of near-unit vectors stays below 2 in magnitude, and float64 holds it exactly in any
order of summation. Scores therefore do not depend on the device or the batch, ties are true ties,
and a tracked document's score computed alone equals the one seen during the scan.

Training judgments only. Competing documents are other documents, not confirmed irrelevant ones.
"""

from __future__ import annotations

import hashlib
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from hyperdime.evaluation.metrics import query_metrics
from hyperdime.evaluation.representation import relevance_gains
from hyperdime.evaluation.representation_controls import query_stream
from hyperdime.evaluation.representation_engine import Clock, gather, stable_order
from hyperdime.evaluation.representation_shared import (
    PARTITIONS,
    TRANSFER_CELLS,
    cell_sizes,
    masses,
    nan_mean,
    optimum_curves,
    partitions,
    scores_under,
    shared_order,
)

DIM = 1024
DEPTH = 100
CUTOFFS = (10, 100)
CHUNK_DOCS = 40_000
BATCH_MASKS = 512
# Masks times tracked documents per batch: 2^15 * 8 KiB bounds the threshold step at 256 MiB.
BATCH_TRACKED = 32_768
SAMPLE_QUERIES = 10
COMPETITORS = 10
CELLS = tuple(cell[0] for cell in TRANSFER_CELLS if cell[0] != "half")
ROWS = ("full", "prefix", "shared", "own", *CELLS)
LIST_METRICS = ("ndcg@10", "mrr@10", "mrr@100", "recall@100")
SCORE_RANKS = (1, 10, 100)
EVALUATED = (
    "rr", "recall10", "recall100", "logrank", "worst", "improved", "worsened", "score",
    "margin10", "z",
)  # fmt: skip
HELD_ONLY = ("held_ndcg10", "secondary_rr", "secondary_recall10", "secondary_recall100")
CONSTRUCTION = ("construction_rr", "construction_recall10")
MEASURES = (
    *LIST_METRICS,
    *EVALUATED,
    *(f"base_{name}" for name in EVALUATED if name not in ("improved", "worsened")),
    *HELD_ONLY,
    *(f"base_{name}" for name in HELD_ONLY),
    *CONSTRUCTION,
    *(f"base_{name}" for name in CONSTRUCTION),
    "corpus_mean_shift",
    "rank10_gain",
    "rank100_gain",
    "evaluated",
)
FULL_BUDGET = -1


def _row(name: str) -> int:
    return ROWS.index(name)


class ScanBatch:
    """Masked queries scored together. Counters stay on the CPU; only the chunk at hand is on the
    device.

    ``tracked`` is ``[masks, P]`` corpus rows, -1 past each mask's documents. Their scores must come
    from the same float16 inputs as the corpus, so that a tracked document meets its own score.
    """

    def __init__(
        self,
        masked: np.ndarray,
        tracked: np.ndarray,
        scores: np.ndarray,
        excluded: np.ndarray,
        depth: int,
    ) -> None:
        self.masked = torch.from_numpy(masked)
        self.depth = depth
        self.excluded = torch.from_numpy(excluded)
        count = masked.shape[0]
        self.count = count
        self.tracked_count = tracked.shape[1]
        rows = torch.from_numpy(tracked)
        pad = torch.full(rows.shape, torch.inf, dtype=torch.float64)
        bounds = torch.where(rows >= 0, torch.from_numpy(scores), pad)
        self.bounds, self.perm = bounds.sort(dim=1, stable=True)
        self.bound_rows = rows.gather(1, self.perm)
        self.scores = scores
        self.tracked = tracked
        self.histogram = torch.zeros((count, self.tracked_count + 1), dtype=torch.int64)
        self.lower = torch.zeros((count, self.tracked_count), dtype=torch.int64)
        self.equal = torch.zeros((count, self.tracked_count), dtype=torch.int64)
        self.top_scores = torch.full((count, depth), -torch.inf, dtype=torch.float64)
        self.top_rows = torch.full((count, depth), -1, dtype=torch.int64)
        self.total = torch.zeros(count, dtype=torch.float64)
        self.squares = torch.zeros(count, dtype=torch.float64)
        self.seen = 0

    def update(self, chunk: Tensor, start: int) -> None:
        """Merge one chunk of float64 document vectors, in increasing corpus order."""
        device = chunk.device
        size = chunk.shape[0]
        queries = self.masked.to(device).to(torch.float64)
        scores = queries @ chunk.T
        self.total += scores.sum(dim=1).cpu()
        self.squares += scores.square().sum(dim=1).cpu()
        self.seen += size
        offset = self.excluded.to(device) - start
        hit = (offset >= 0) & (offset < size)
        if hit.any():
            scores[hit.nonzero()[:, 0], offset[hit]] = -torch.inf
        if self.tracked_count:
            self._count(scores, start)
        if self.depth:
            self._merge(scores, start)

    def _count(self, scores: Tensor, start: int) -> None:
        device = scores.device
        bounds = self.bounds.to(device)
        left = torch.searchsorted(bounds, scores, right=False)
        right = torch.searchsorted(bounds, scores, right=True)
        histogram = torch.zeros(self.histogram.shape, dtype=torch.int64, device=device)
        histogram.scatter_add_(1, left, torch.ones_like(left))
        self.histogram += histogram.cpu()
        tied = (right > left).nonzero()
        if not tied.numel():
            return
        mask, doc = tied[:, 0], tied[:, 1]
        low = left[mask, doc]
        width = right[mask, doc] - low
        mask_each = mask.repeat_interleave(width)
        doc_each = doc.repeat_interleave(width)
        first = torch.cat([torch.zeros(1, device=device, dtype=width.dtype), width.cumsum(0)[:-1]])
        inside = torch.arange(int(width.sum().item()), device=device) - first.repeat_interleave(
            width
        )
        where = low.repeat_interleave(width) + inside
        before = start + doc_each < self.bound_rows.to(device)[mask_each, where]
        key = (mask_each.cpu(), where.cpu())
        self.equal.index_put_(key, torch.ones_like(key[0]), accumulate=True)
        self.lower.index_put_(key, before.long().cpu(), accumulate=True)

    def _merge(self, scores: Tensor, start: int) -> None:
        device = scores.device
        take = min(self.depth, scores.shape[1])
        cut = scores.topk(take, dim=1).values[:, -1]
        # A tie at the cut can fall on either side of topk; keep them all and decide by row.
        found = (scores >= cut[:, None]).nonzero()
        mask, doc = found[:, 0], found[:, 1]
        per_mask = torch.bincount(mask, minlength=self.count)
        first = torch.cat([torch.zeros(1, device=device, dtype=per_mask.dtype), per_mask.cumsum(0)])
        slot = torch.arange(mask.numel(), device=device) - first[mask]
        width = int(per_mask.max().item())
        new_scores = torch.full((self.count, width), -torch.inf, dtype=torch.float64, device=device)
        new_rows = torch.full((self.count, width), -1, dtype=torch.int64, device=device)
        new_scores[mask, slot] = scores[mask, doc]
        new_rows[mask, slot] = start + doc
        # Earlier chunks hold lower rows, so a stable sort keeps ties in row order.
        merged_scores = torch.cat([self.top_scores.to(device), new_scores], dim=1)
        merged_rows = torch.cat([self.top_rows.to(device), new_rows], dim=1)
        keep = stable_order(merged_scores)[:, : self.depth]
        self.top_scores = merged_scores.gather(1, keep).cpu()
        self.top_rows = merged_rows.gather(1, keep).cpu()

    def finish(self) -> dict[str, np.ndarray]:
        """Ranks in the order of ``tracked``; 0 for a pad or a document out of the competition."""
        histogram = self.histogram.numpy()
        above = np.cumsum(histogram[:, :0:-1], axis=1)[:, ::-1]
        rank = 1 + above + self.lower.numpy()
        inverse = np.argsort(self.perm.numpy(), axis=1)
        rank = np.take_along_axis(rank, inverse, axis=1)
        equal = np.take_along_axis(self.equal.numpy(), inverse, axis=1)
        competing = (self.tracked >= 0) & (self.tracked != self.excluded.numpy()[:, None])
        if (equal[competing] < 1).any():
            raise RuntimeError("a tracked document did not meet its own score during the scan")
        scores = self.top_scores.numpy().copy()
        rows = self.top_rows.numpy().copy()
        rows[np.isinf(scores)] = -1
        return {
            "rank": np.where(competing, rank, 0),
            "ties": np.where(competing, equal - 1, 0),
            "top_rows": rows,
            "top_scores": scores,
            "corpus_mean": self.total.numpy() / self.seen,
            "corpus_var": self.squares.numpy() / self.seen - (self.total.numpy() / self.seen) ** 2,
        }


def tracked_scores(
    masked: np.ndarray, tracked: np.ndarray, documents: np.ndarray, device: str
) -> np.ndarray:
    """``[masks, P]`` exact scores of the tracked rows; 0 past each mask's documents."""
    found = np.zeros(tracked.shape, dtype=np.float64)
    where = np.nonzero(tracked >= 0)
    if where[0].size:
        docs = torch.from_numpy(gather(documents, tracked[where])).to(device)
        queries = torch.from_numpy(masked[where[0]]).to(device).to(torch.float64)
        found[where] = (queries * docs).sum(dim=1).cpu().numpy()
    return found


def scan_corpus(
    documents: np.ndarray,
    batches: Sequence[ScanBatch],
    device: str,
    chunk_docs: int = CHUNK_DOCS,
    clock: Clock | None = None,
    log: Callable[[str], None] | None = None,
) -> None:
    """One pass over the corpus: each chunk moves to the device once and serves every batch."""
    total = documents.shape[0]
    for number, start in enumerate(range(0, total, chunk_docs), 1):
        if clock is None:
            chunk = _chunk(documents, start, chunk_docs, device)
        else:
            with clock.span("read"):
                chunk = _chunk(documents, start, chunk_docs, device)
        for batch in batches:
            if clock is None:
                batch.update(chunk, start)
            else:
                with clock.span("scan"):
                    batch.update(chunk, start)
        if log is not None and number % 25 == 0:
            log(f"chunk {number} of {(total + chunk_docs - 1) // chunk_docs}")


def _chunk(documents: np.ndarray, start: int, size: int, device: str) -> Tensor:
    block = np.array(documents[start : start + size], dtype=np.float16)
    return torch.from_numpy(block).to(device).to(torch.float64)


@dataclass(frozen=True)
class Source:
    """One mask ordering of a query; a budget turns it into a mask."""

    query: int
    row: int
    partition: int = -1
    positive: int = -1
    order: np.ndarray | None = None
    construction: np.ndarray | None = None
    held: np.ndarray | None = None


def plan_query(
    index: int,
    qid: str,
    query: np.ndarray,
    docs: np.ndarray,
    grades: np.ndarray,
    *,
    seed: int,
    device: str,
    own: bool = True,
    check: Callable[[str, np.ndarray], None] | None = None,
) -> list[Source]:
    """Every mask ordering of one query: the full and prefix rows, the all-positive mask, each
    positive's own mask, and construction masks built from the construction documents alone."""
    q = torch.from_numpy(np.asarray(query, dtype=np.float64)).to(device)
    vectors = torch.from_numpy(docs).to(device)
    values = q * vectors
    count = values.shape[0]
    gains = torch.from_numpy(relevance_gains(grades)).to(device, torch.float64)
    optimum, _ = optimum_curves(values)
    mass = masses(values)
    sources = [
        Source(index, _row("full")),
        Source(index, _row("prefix"), order=np.arange(DIM, dtype=np.int16)),
    ]
    shared = shared_order(values[None], gains[None])
    sources.append(Source(index, _row("shared"), order=shared[0].cpu().numpy().astype(np.int16)))
    if check is not None:
        raw = optimum - scores_under(values[None], shared)[0]
        check("shared", nan_mean(raw / mass[:, None], 0).cpu().numpy())
    if own:
        orders = stable_order(values).cpu().numpy().astype(np.int16)
        sources.extend(
            Source(index, _row("own"), positive=position, order=orders[position])
            for position in range(count)
        )
    for position, (name, construct, held) in enumerate(TRANSFER_CELLS):
        if name not in CELLS:
            continue
        sizes = cell_sizes(count, construct, held)
        if sizes is None:
            continue
        # The stream is the one ADR-0015 used for this cell; only its documents reach the mean.
        splits = partitions(
            count, sizes[0], sizes[1], query_stream(seed, 400 + position, qid), PARTITIONS
        )
        built = torch.from_numpy(np.stack([c for c, _ in splits])).to(device)
        held_rows = torch.from_numpy(np.stack([h for _, h in splits])).to(device)
        split_gains = torch.from_numpy(
            np.stack([relevance_gains(grades[c]) for c, _ in splits])
        ).to(device, torch.float64)
        order = shared_order(values[built], split_gains)
        if check is not None:
            raw = optimum[held_rows] - scores_under(values[held_rows], order)
            regret = nan_mean(nan_mean(raw / mass[held_rows][:, :, None], 1), 0)
            check(name, regret.cpu().numpy())
        for slot, (kept, left_out) in enumerate(splits):
            sources.append(
                Source(
                    index,
                    _row(name),
                    partition=slot,
                    order=order[slot].cpu().numpy().astype(np.int16),
                    construction=kept,
                    held=left_out,
                )
            )
    return sources


@dataclass
class MaskPlan:
    """One row per mask, in the order results are stored."""

    sources: list[Source]
    source: np.ndarray
    budget: np.ndarray  # index into the budget list; FULL_BUDGET for the full query
    tracked: np.ndarray  # number of documents each mask tracks


def expand(sources: Sequence[Source], budgets: Sequence[int], counts: np.ndarray) -> MaskPlan:
    """Each ordering at every budget; the full query once. ``counts`` is positives per query."""
    source: list[int] = []
    budget: list[int] = []
    tracked: list[int] = []
    for at, item in enumerate(sources):
        if item.row == _row("full"):
            source.append(at)
            budget.append(FULL_BUDGET)
            tracked.append(int(counts[item.query]))
            continue
        for slot in range(len(budgets)):
            source.append(at)
            budget.append(slot)
            tracked.append(1 if item.row == _row("own") else int(counts[item.query]))
    return MaskPlan(
        list(sources),
        np.array(source, dtype=np.int64),
        np.array(budget, dtype=np.int32),
        np.array(tracked, dtype=np.int32),
    )


def masked_vector(query: np.ndarray, order: np.ndarray | None, k: int) -> np.ndarray:
    """The query with all but ``order``'s first ``k`` dimensions zeroed; never re-normalized."""
    if order is None or k >= DIM:
        return np.array(query, dtype=np.float16)
    kept = np.zeros(DIM, dtype=np.float16)
    chosen = order[:k]
    kept[chosen] = query[chosen]
    return kept


def _batches(plan: MaskPlan, batch_masks: int, batch_tracked: int) -> list[np.ndarray]:
    """Mask indices per batch, retrieved lists apart from rank-only masks, by tracked count, so
    padding stays small and masks times tracked documents stays bounded."""
    rank_only = np.fromiter(
        (plan.sources[s].row == _row("own") for s in plan.source),
        dtype=bool,
        count=len(plan.source),
    )
    order = np.lexsort((plan.tracked, rank_only))
    found: list[np.ndarray] = []
    at = 0
    while at < order.size:
        take = min(batch_masks, order.size - at)
        slice_ = order[at : at + take]
        while take > 1 and (
            take * int(plan.tracked[slice_].max()) > batch_tracked
            or rank_only[slice_].any() != rank_only[slice_].all()
        ):
            take = max(1, take // 2)
            slice_ = order[at : at + take]
        found.append(slice_)
        at += take
    return found


def _measures_of(
    rank: np.ndarray,
    full_rank: np.ndarray,
    score: np.ndarray,
    full_score: np.ndarray,
    mask_stats: np.ndarray,
    full_stats: np.ndarray,
) -> dict[str, float]:
    """Per-positive measures over one evaluated set, against the same documents under the full
    query. ``*_stats`` hold corpus mean, corpus sd, and the scores at ranks 10 and 100."""

    def side(ranks: np.ndarray, scores: np.ndarray, stats: np.ndarray) -> dict[str, float]:
        mean, sd, at_ten = stats[:3]
        return {
            "rr": float((1 / ranks).mean()),
            "recall10": float((ranks <= 10).mean()),
            "recall100": float((ranks <= 100).mean()),
            "logrank": float(np.log(ranks).mean()),
            "worst": float(ranks.max()),
            "score": float(scores.mean()),
            "margin10": float((scores - at_ten).mean()),
            "z": float(((scores - mean) / sd).mean()) if sd > 0 else float("nan"),
        }

    here = side(rank, score, mask_stats)
    base = side(full_rank, full_score, full_stats)
    return {
        **here,
        **{f"base_{name}": value for name, value in base.items()},
        "improved": float((rank < full_rank).mean()),
        "worsened": float((rank > full_rank).mean()),
    }


def held_measures(
    rank: np.ndarray, construction_rank: np.ndarray, grades: np.ndarray
) -> dict[str, float]:
    """Diagnostics of held-out positives: nDCG@10 where construction positives keep their places
    with gain zero, and ranks that skip construction positives, a secondary reading of crowding."""
    gains = np.asarray(grades, dtype=np.float64)
    top = rank <= 10
    dcg = float((gains[top] / np.log2(rank[top] + 1)).sum())
    ideal = np.sort(gains)[::-1][:10]
    idcg = float((ideal / np.log2(np.arange(2, ideal.size + 2))).sum())
    competing = construction_rank[construction_rank > 0]
    before = np.searchsorted(np.sort(competing), rank, side="left")
    skipping = rank - before
    return {
        "held_ndcg10": dcg / idcg if idcg else 0.0,
        "secondary_rr": float((1 / skipping).mean()),
        "secondary_recall10": float((skipping <= 10).mean()),
        "secondary_recall100": float((skipping <= 100).mean()),
    }


def per_query_summary(
    plan: MaskPlan,
    results: Mapping[str, np.ndarray],
    offsets: np.ndarray,
    query_count: int,
    budgets: Sequence[int],
    grades: Sequence[np.ndarray],
) -> dict[str, np.ndarray]:
    """``[rows, budgets, queries]`` per-query means of every measure, partitions averaged within a
    query; NaN where a row has no value. The full row sits in the last budget slot."""
    slots = len(budgets) + 1
    totals = {name: np.zeros((len(ROWS), slots, query_count)) for name in MEASURES}
    seen = {name: np.zeros((len(ROWS), slots, query_count)) for name in MEASURES}

    def add(row: int, slot: int, query: int, values: Mapping[str, float]) -> None:
        for name, value in values.items():
            if not np.isnan(value):
                totals[name][row, slot, query] += value
                seen[name][row, slot, query] += 1

    def tracked_of(mask: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        low, high = offsets[mask], offsets[mask + 1]
        return results["position"][low:high], results["rank"][low:high], results["score"][low:high]

    def stats_of(mask: int) -> np.ndarray:
        return np.array(
            [
                results["corpus_mean"][mask],
                np.sqrt(max(results["corpus_var"][mask], 0.0)),
                results["score_at"][mask, 1],
                results["score_at"][mask, 2],
            ]
        )

    full_of: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for mask, source_at in enumerate(plan.source):
        source = plan.sources[source_at]
        if source.row != _row("full"):
            continue
        position, rank, score = tracked_of(mask)
        full_rank = np.zeros(grades[source.query].size, dtype=np.int64)
        full_score = np.zeros(grades[source.query].size)
        full_rank[position], full_score[position] = rank, score
        full_of[source.query] = (full_rank, full_score, stats_of(mask))

    for mask, source_at in enumerate(plan.source):
        source = plan.sources[source_at]
        slot = int(plan.budget[mask])
        if slot == FULL_BUDGET:
            slot = slots - 1
        full_rank, full_score, full_stats = full_of[source.query]
        position, rank, score = tracked_of(mask)
        ranks_here = np.zeros(full_rank.size, dtype=np.int64)
        scores_here = np.zeros(full_rank.size)
        ranks_here[position], scores_here[position] = rank, score
        if source.row == _row("own"):
            evaluate = np.array([source.positive])
        elif source.held is not None:
            evaluate = source.held
        else:
            evaluate = np.arange(full_rank.size)
        evaluate = evaluate[(ranks_here[evaluate] > 0) & (full_rank[evaluate] > 0)]
        if evaluate.size == 0:
            continue
        stats = stats_of(mask)
        values = _measures_of(
            ranks_here[evaluate], full_rank[evaluate], scores_here[evaluate], full_score[evaluate],
            stats, full_stats,
        )  # fmt: skip
        values["evaluated"] = float(evaluate.size)
        values["corpus_mean_shift"] = float(stats[0] - full_stats[0])
        values["rank10_gain"] = float(stats[2] - full_stats[2])
        values["rank100_gain"] = float(stats[3] - full_stats[3])
        if source.row != _row("own"):
            values.update({name: float(results[name][mask]) for name in LIST_METRICS})
        if source.held is not None and source.construction is not None:
            held_grades = grades[source.query][evaluate]
            here = held_measures(ranks_here[evaluate], ranks_here[source.construction], held_grades)
            base = held_measures(full_rank[evaluate], full_rank[source.construction], held_grades)
            values.update(here)
            values.update({f"base_{name}": value for name, value in base.items()})
            built = source.construction
            built = built[(ranks_here[built] > 0) & (full_rank[built] > 0)]
            if built.size:
                values["construction_rr"] = float((1 / ranks_here[built]).mean())
                values["base_construction_rr"] = float((1 / full_rank[built]).mean())
                values["construction_recall10"] = float((ranks_here[built] <= 10).mean())
                values["base_construction_recall10"] = float((full_rank[built] <= 10).mean())
        add(source.row, slot, source.query, values)

    found: dict[str, np.ndarray] = {}
    for name in MEASURES:
        with np.errstate(invalid="ignore", divide="ignore"):
            found[name] = totals[name] / seen[name]
    return found


def sample_queries(query_ids: Sequence[str], seed: int, count: int = SAMPLE_QUERIES) -> list[int]:
    """A seeded sample of query positions for score diagnostics, fixed before seeing outcomes."""
    ranked = sorted(
        range(len(query_ids)),
        key=lambda at: hashlib.sha256(f"{seed}:margin:{query_ids[at]}".encode()).hexdigest(),
    )
    return sorted(ranked[:count])


def _margin_rows(
    plan: MaskPlan,
    results: Mapping[str, np.ndarray],
    offsets: np.ndarray,
    picked: set[int],
    query_vector: Callable[[int], np.ndarray],
    positive_rows: Sequence[np.ndarray],
    documents: np.ndarray,
) -> dict[str, np.ndarray]:
    """For sampled masks of the shared and 1-1 rows: the worst-ranked evaluated positive against the
    mask's top non-positive documents, whose scores are taken under the full query as well."""
    full_rank: dict[int, np.ndarray] = {}
    for mask, source_at in enumerate(plan.source):
        source = plan.sources[source_at]
        if source.row == _row("full") and source.query in picked:
            low, high = offsets[mask], offsets[mask + 1]
            ranks = np.zeros(positive_rows[source.query].size, dtype=np.int64)
            ranks[results["position"][low:high]] = results["rank"][low:high]
            full_rank[source.query] = ranks
    rows: list[list[float]] = []
    for mask, source_at in enumerate(plan.source):
        source = plan.sources[source_at]
        if source.query not in picked or plan.budget[mask] == FULL_BUDGET:
            continue
        if source.row not in (_row("shared"), _row("1-1")) or source.partition > 0:
            continue
        low, high = offsets[mask], offsets[mask + 1]
        evaluate = source.held if source.held is not None else np.arange(high - low)
        ranks = np.zeros(positive_rows[source.query].size, dtype=np.int64)
        scores = np.zeros(positive_rows[source.query].size)
        ranks[results["position"][low:high]] = results["rank"][low:high]
        scores[results["position"][low:high]] = results["score"][low:high]
        evaluate = evaluate[ranks[evaluate] > 0]
        if evaluate.size == 0:
            continue
        worst = evaluate[np.argmax(ranks[evaluate])]
        retrieved = results["top_rows"][results["list_index"][mask]]
        others = [r for r in retrieved if r >= 0 and r not in set(positive_rows[source.query])]
        others = others[:COMPETITORS]
        if not others:
            continue
        query = np.asarray(query_vector(source.query), dtype=np.float64)
        docs = gather(documents, np.array(others, dtype=np.int64))
        full_scores = docs @ query
        positive = gather(documents, positive_rows[source.query][[worst]])[0]
        full_positive = float(positive @ query)
        k = int(results["budgets"][plan.budget[mask]])
        order = source.order if source.order is not None else np.arange(DIM)
        masked = np.zeros(DIM)
        masked[order[:k]] = query[order[:k]]
        masked_positive = float(positive @ masked)
        masked_others = docs @ masked
        rows.append(
            [
                source.query,
                source.row,
                k,
                int(ranks[worst]),
                int(full_rank[source.query][worst]),
                full_positive,
                masked_positive,
                float(full_scores.mean()),
                float(masked_others.mean()),
                len(others),
            ]
        )
    return {"rows": np.array(rows, dtype=np.float64).reshape(-1, 10)}


def analyze_rankings(
    query_ids: Sequence[str],
    query_vector: Callable[[str], np.ndarray],
    documents: np.ndarray,
    doc_ids: Sequence[str],
    positives: Mapping[str, np.ndarray],
    grades: Mapping[str, np.ndarray],
    judgments: Mapping[str, Mapping[str, int]],
    *,
    seed: int,
    budgets: Sequence[int],
    device: str,
    clock: Clock,
    path: Callable[[str], Path],
    own: bool = True,
    chunk_docs: int = CHUNK_DOCS,
    batch_masks: int = BATCH_MASKS,
    batch_tracked: int = BATCH_TRACKED,
    stored_regret: Callable[[int, str], np.ndarray | None] | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Rank the whole corpus under every mask of every query; write arrays and return a summary."""
    say = log or (lambda text: print(text, file=sys.stderr, flush=True))
    started = time.perf_counter()
    if documents.dtype != np.float16:
        raise ValueError("exact float64 scoring needs float16 documents")
    doc_rows = {doc: row for row, doc in enumerate(doc_ids)}
    excluded = np.array([doc_rows.get(qid, -1) for qid in query_ids], dtype=np.int64)
    counts = np.array([positives[qid].size for qid in query_ids], dtype=np.int64)
    largest_gap = 0.0
    regrets_compared = 0

    def checked(index: int) -> Callable[[str, np.ndarray], None] | None:
        if stored_regret is None:
            return None

        def check(name: str, curve: np.ndarray) -> None:
            nonlocal largest_gap, regrets_compared
            stored = stored_regret(index, name)
            if stored is None:
                return
            at = np.array([k - 1 for k in budgets])
            gap = np.nanmax(np.abs(curve[at] - stored[at]))
            largest_gap = max(largest_gap, float(gap))
            regrets_compared += at.size

        return check

    sources: list[Source] = []
    for index, qid in enumerate(query_ids):
        with clock.span("plan"):
            sources.extend(
                plan_query(
                    index,
                    qid,
                    np.asarray(query_vector(qid), dtype=np.float16),
                    gather(documents, positives[qid]),
                    grades[qid],
                    seed=seed,
                    device=device,
                    own=own,
                    check=checked(index),
                )
            )
    plan = expand(sources, budgets, counts)
    query_ids = list(query_ids)
    say(f"{len(sources)} orderings, {plan.source.size} masks, {documents.shape[0]} documents")
    batch_indices = _batches(plan, batch_masks, batch_tracked)
    batches: list[ScanBatch] = []
    with clock.span("thresholds"):
        for members in batch_indices:
            batches.append(
                _build_batch(
                    plan,
                    members,
                    query_ids,
                    query_vector,
                    positives,
                    excluded,
                    budgets,
                    documents,
                    device,
                )  # fmt: skip
            )
    say(f"{len(batches)} batches; scanning")
    scan_corpus(documents, batches, device, chunk_docs, clock, say)
    with clock.span("metrics"):
        arrays, ties_total, excluded_positives = _collect(
            plan, batch_indices, batches, query_ids, doc_ids, judgments, budgets
        )
        results = {**arrays, "budgets": np.asarray(budgets, dtype=np.int64)}
        offsets = arrays["offsets"]
        picks = sample_queries(query_ids, seed)
        per_query = per_query_summary(
            plan, results, offsets, len(query_ids), budgets, [grades[qid] for qid in query_ids]
        )
        margins = _margin_rows(
            plan,
            results,
            offsets,
            set(picks),
            lambda at: np.asarray(query_vector(query_ids[at])),
            [positives[qid] for qid in query_ids],
            documents,
        )
    with clock.span("write"):
        _save(path("masks.npz"), _mask_columns(plan, results))
        np.savez(
            path("tracked.npz"),
            offsets=offsets,
            position=results["position"],
            rank=results["rank"],
            score=results["score"],
            ties=results["ties"],
        )
        _save(path("per-query.npz"), {
            "counts": counts, **{k: v.astype(np.float32) for k, v in per_query.items()}
        })  # fmt: skip
        _save(path("margin-sample.npz"), margins)
    return {
        "coverage": {
            "queries": len(query_ids),
            "positives": int(counts.sum()),
            "documents": int(documents.shape[0]),
            "orderings": len(sources),
            "masks": int(plan.source.size),
            "masks_per_row": {
                ROWS[row]: int(sum(plan.sources[s].row == row for s in plan.source))
                for row in range(len(ROWS))
            },
            "queries_with_excluded_row": int((excluded >= 0).sum()),
            "budgets": list(budgets),
        },
        "checks": {
            "largest_stored_regret_difference": largest_gap if regrets_compared else None,
            "stored_regrets_compared": regrets_compared,
            "tied_tracked_documents": ties_total,
            "excluded_positives": excluded_positives,
        },
        "cost": {"seconds": time.perf_counter() - started},
    }


def _save(target: Path, arrays: Mapping[str, np.ndarray]) -> None:
    np.savez(target, **arrays)  # type: ignore[arg-type]


def _build_batch(
    plan: MaskPlan,
    members: np.ndarray,
    query_ids: Sequence[str],
    query_vector: Callable[[str], np.ndarray],
    positives: Mapping[str, np.ndarray],
    excluded: np.ndarray,
    budgets: Sequence[int],
    documents: np.ndarray,
    device: str,
) -> ScanBatch:
    width = int(plan.tracked[members].max())
    masked = np.zeros((members.size, DIM), dtype=np.float16)
    tracked = np.full((members.size, width), -1, dtype=np.int64)
    excluded_rows = np.empty(members.size, dtype=np.int64)
    for at, mask in enumerate(members):
        source = plan.sources[plan.source[mask]]
        slot = int(plan.budget[mask])
        k = DIM if slot == FULL_BUDGET else int(budgets[slot])
        qid = query_ids[source.query]
        masked[at] = masked_vector(np.asarray(query_vector(qid), dtype=np.float16), source.order, k)
        rows = positives[qid]
        if source.row == _row("own"):
            tracked[at, 0] = rows[source.positive]
        else:
            tracked[at, : rows.size] = rows
        excluded_rows[at] = excluded[source.query]
    rank_only = plan.sources[plan.source[members[0]]].row == _row("own")
    return ScanBatch(
        masked,
        tracked,
        tracked_scores(masked, tracked, documents, device),
        excluded_rows,
        0 if rank_only else DEPTH,
    )


def _collect(
    plan: MaskPlan,
    batch_indices: Sequence[np.ndarray],
    batches: Sequence[ScanBatch],
    query_ids: Sequence[str],
    doc_ids: Sequence[str],
    judgments: Mapping[str, Mapping[str, int]],
    budgets: Sequence[int],
) -> tuple[dict[str, np.ndarray], int, int]:
    """Turn scan counters into mask-ordered arrays, and retrieved lists into the repository's
    metrics, with query ids removed by the scan and the unchanged full training judgments."""
    count = plan.source.size
    position_of = np.zeros(count + 1, dtype=np.int64)
    position_of[1:] = np.cumsum(plan.tracked)
    total = int(position_of[-1])
    position = np.zeros(total, dtype=np.int32)
    ranks = np.zeros(total, dtype=np.int64)
    scores = np.zeros(total, dtype=np.float64)
    ties = np.zeros(total, dtype=np.int32)
    corpus_mean = np.zeros(count)
    corpus_var = np.zeros(count)
    score_at = np.full((count, len(SCORE_RANKS)), np.nan)
    metrics = {name: np.full(count, np.nan, dtype=np.float32) for name in LIST_METRICS}
    list_index = np.full(count, -1, dtype=np.int64)
    kept_lists: list[np.ndarray] = []
    excluded_positives = 0
    tied = 0
    for members, batch in zip(batch_indices, batches, strict=True):
        found = batch.finish()
        rank, doc_ties = found["rank"], found["ties"]
        for at, mask in enumerate(members):
            source = plan.sources[plan.source[mask]]
            start, end = position_of[mask], position_of[mask + 1]
            width = end - start
            position[start:end] = (
                np.array([source.positive]) if source.row == _row("own") else np.arange(width)
            )
            ranks[start:end] = rank[at, :width]
            ties[start:end] = doc_ties[at, :width]
            scores[start:end] = batch.scores[at, :width]
            corpus_mean[mask] = found["corpus_mean"][at]
            corpus_var[mask] = found["corpus_var"][at]
            excluded_positives += int((rank[at, :width] == 0).sum())
            tied += int(doc_ties[at, :width].sum())
            if batch.depth:
                rows = found["top_rows"][at]
                list_index[mask] = len(kept_lists)
                kept_lists.append(rows.astype(np.int32))
                at_ranks = found["top_scores"][at][[r - 1 for r in SCORE_RANKS]]
                score_at[mask] = np.where(np.isinf(at_ranks), np.nan, at_ranks)
                ranking = [doc_ids[r] for r in rows if r >= 0]
                judged = judgments[query_ids[source.query]]
                values = query_metrics(ranking, judged, CUTOFFS)
                for name in LIST_METRICS:
                    metrics[name][mask] = values[name]
    arrays: dict[str, np.ndarray] = {
        "offsets": position_of,
        "position": position,
        "rank": ranks,
        "score": scores,
        "ties": ties,
        "corpus_mean": corpus_mean,
        "corpus_var": corpus_var,
        "score_at": score_at,
        "list_index": list_index,
        "top_rows": (np.stack(kept_lists) if kept_lists else np.empty((0, DEPTH), dtype=np.int32)),
        **metrics,
    }
    return arrays, tied, excluded_positives


def _mask_columns(plan: MaskPlan, results: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    sources = plan.sources
    return {
        "query": np.array([sources[s].query for s in plan.source], dtype=np.int32),
        "row": np.array([sources[s].row for s in plan.source], dtype=np.int8),
        "partition": np.array([sources[s].partition for s in plan.source], dtype=np.int8),
        "positive": np.array([sources[s].positive for s in plan.source], dtype=np.int32),
        "budget": plan.budget,
        "corpus_mean": results["corpus_mean"],
        "corpus_var": results["corpus_var"],
        "score_at": results["score_at"],
        **{name.replace("@", "_at_"): results[name] for name in LIST_METRICS},
    }
