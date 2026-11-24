"""Bounded, CPU-only check of a stored P3-mask-ranking run against its inputs (ADR-0016).

    python scripts/checks/p3_ranking_integrity.py <artifacts> <run_id> --output <dir> \
        [--report-dir <generated report>]

Reads the run and never writes into it. Invariants cover every stored row. Spot checks rebuild a
seeded, fixed set of masks and rescore the whole corpus with NumPy, in code apart from the scan that
produced the run: a few queries per stratum, every arm and budget, one partition per cell. This is
not a rerun of the experiment. With a report, the primary held-out comparisons are recomputed from
the per-query arrays and compared with its claims table. Writes counts and ids of checks only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from hyperdime.cli.analyze_representations import load_training_judgments
from hyperdime.contracts.hashing import NO_DATASET, file_sha256
from hyperdime.contracts.manifests import ArtifactWriter, read_manifest, start_manifest
from hyperdime.data.prepare import qrels_file
from hyperdime.evaluation.metrics import query_metrics
from hyperdime.evaluation.representation_engine import gather
from hyperdime.evaluation.representation_matching import select_stratified
from hyperdime.evaluation.representation_ranking import (
    CELLS,
    CUTOFFS,
    DEPTH,
    FULL_BUDGET,
    LIST_METRICS,
    MEASURES,
    PARTITIONS,
    ROWS,
    SCORE_RANKS,
    expand,
    masked_vector,
    per_query_summary,
    plan_query,
)
from hyperdime.evaluation.representation_shared import TRANSFER_CELLS
from hyperdime.evaluation.statistics import paired_randomization_test
from hyperdime.reports.p3_ranking import (
    PERMUTATIONS,
    PRIMARY_MEASURES,
    RESAMPLES,
    resampled_queries,
)
from hyperdime.space.inputs import UnlabeledInputs, load_unlabeled

DIM = 1024
CHUNK_DOCS = 30_000
PER_STRATUM = 4
MEAN_TOLERANCE = 1e-12
CELL_SIZES = [
    (name, construct, held)
    for name, construct, held in TRANSFER_CELLS
    if name in CELLS and construct is not None and held is not None
]
TRACKED_SAME_AS_FULL = ("rr", "recall10", "recall100", "logrank", "worst", "score")


@dataclass
class Checks:
    """Named pass-or-fail checks of one environment; a failure carries its label."""

    env: str
    passed: int = 0
    failed: list[str] = field(default_factory=list)

    def check(self, name: str, ok: bool) -> None:
        if ok:
            self.passed += 1
        else:
            self.failed.append(f"{self.env}: {name}")


def _same(left: np.ndarray, right: np.ndarray) -> bool:
    """Equal arrays, with NaN equal to NaN."""
    return left.shape == right.shape and np.array_equal(left, right, equal_nan=True)


def _partitions(count: int, construct: int, held: int) -> int:
    if construct + held > count:
        return 0
    total = math.comb(count, construct) * math.comb(count - construct, held)
    return total if total <= PARTITIONS else PARTITIONS


@dataclass
class Environment:
    """The inputs of one environment, rebuilt the way the run built them."""

    inputs: UnlabeledInputs
    judgments: dict[str, dict[str, int]]
    positive: dict[str, np.ndarray]
    grades: dict[str, np.ndarray]
    multi: dict[str, int]


def _environment(root: Path, env: str) -> Environment:
    inputs = load_unlabeled(root, env)
    judgments = load_training_judgments(inputs.dataset_dir, inputs.splits.get("train", []))
    doc_rows = {doc: row for row, doc in enumerate(inputs.doc_ids)}
    positive: dict[str, np.ndarray] = {}
    grades: dict[str, np.ndarray] = {}
    for qid in inputs.splits.get("train", []):
        found = sorted(
            (doc_rows[d], g) for d, g in judgments.get(qid, {}).items() if d in doc_rows and g > 0
        )
        if found:
            positive[qid] = np.array([row for row, _ in found], dtype=np.int64)
            grades[qid] = np.array([g for _, g in found], dtype=np.int64)
    multi = {qid: rows.size for qid, rows in positive.items() if rows.size > 1}
    return Environment(inputs, judgments, positive, grades, multi)


def check_selection(
    c: Checks, e: Environment, ids: list[str], strata: np.ndarray, weights: np.ndarray, seed: int
) -> None:
    """The queries are every eligible training query, in the order the census would give."""
    expected, expected_strata, expected_weights, _ = select_stratified(e.multi, seed, len(e.multi))
    c.check("selected queries are exactly the eligible multi-positive set", ids == expected)
    c.check("strata match the selection rule", _same(strata, expected_strata))
    c.check("weights match the selection rule", _same(weights, expected_weights))
    c.check("no query was capped", bool(np.allclose(weights, 1.0)))
    train = set(e.inputs.splits.get("train", []))
    others = {q for split, members in e.inputs.splits.items() if split != "train" for q in members}
    c.check("selected queries are training queries", set(ids) <= train)
    c.check("selected queries are in no other split", not set(ids) & others)


def check_masks(
    c: Checks,
    m: Mapping[str, np.ndarray],
    counts: np.ndarray,
    budgets: Sequence[int],
    summary: Mapping[str, Any],
) -> None:
    """Every query has exactly the masks its positives allow, each at every budget, once."""
    query, row, part, pos, budget = m["query"], m["row"], m["partition"], m["positive"], m["budget"]
    slots = len(budgets)
    observed = np.zeros((counts.size, len(ROWS)), dtype=np.int64)
    np.add.at(observed, (query, row), 1)
    expected = np.zeros_like(observed)
    expected[:, ROWS.index("full")] = 1
    expected[:, ROWS.index("prefix")] = slots
    expected[:, ROWS.index("shared")] = slots
    expected[:, ROWS.index("own")] = counts * slots
    for name, construct, held in CELL_SIZES:
        expected[:, ROWS.index(name)] = [
            slots * _partitions(int(n), construct, held) for n in counts
        ]
    c.check("masks per query and arm", _same(observed, expected))
    order = np.lexsort((budget, pos, part, row, query))
    keys = np.stack([query, row, part, pos, budget], axis=1)[order]
    c.check("no mask is stored twice", bool((keys[1:] != keys[:-1]).any(axis=1).all()))
    full = row == ROWS.index("full")
    c.check("only the full row has no budget", _same(budget == FULL_BUDGET, full))
    c.check("budget slots are in range", bool(((budget >= 0) & (budget < slots) | full).all()))
    own = row == ROWS.index("own")
    c.check(
        "own masks name one of their positives",
        bool(((pos[own] >= 0) & (pos[own] < counts[query[own]])).all()),
    )
    c.check("only own masks name a positive", bool((pos[~own] == -1).all()))
    in_cell = np.isin(row, [ROWS.index(name) for name in CELLS])
    c.check(
        "partitions are in range", bool(((part[in_cell] >= 0) & (part[in_cell] < PARTITIONS)).all())
    )
    c.check("only cells have partitions", bool((part[~in_cell] == -1).all()))
    per_row = {ROWS[r]: int((row == r).sum()) for r in range(len(ROWS))}
    c.check("summary counts masks per arm", per_row == summary["coverage"]["masks_per_row"])
    c.check("summary counts masks", row.size == summary["coverage"]["masks"])
    c.check("summary counts positives", int(counts.sum()) == summary["coverage"]["positives"])


def check_tracked(
    c: Checks,
    m: Mapping[str, np.ndarray],
    t: Mapping[str, np.ndarray],
    counts: np.ndarray,
    documents: int,
    summary: Mapping[str, Any],
) -> None:
    """Bounds and internal consistency of every stored rank and score."""
    offsets, position, rank, score, ties = (
        t["offsets"],
        t["position"],
        t["rank"],
        t["score"],
        t["ties"],
    )
    own = m["row"] == ROWS.index("own")
    widths = np.where(own, 1, counts[m["query"]])
    c.check("offsets start at zero", int(offsets[0]) == 0)
    c.check("offsets give each mask its documents", _same(np.diff(offsets), widths))
    c.check("offsets end at the stored rows", int(offsets[-1]) == rank.size)
    mask_of = np.repeat(np.arange(widths.size), widths)
    expected_position = np.arange(rank.size) - offsets[mask_of]
    expected_position[own[mask_of]] = m["positive"][own]
    c.check("positions name each mask's documents", _same(position, expected_position))
    c.check("ranks are within the corpus", bool(((rank >= 0) & (rank <= documents)).all()))
    checks = summary["checks"]
    c.check(
        "rank zero only for excluded positives",
        int((rank == 0).sum()) == checks["excluded_positives"],
    )
    c.check("tie counts are not negative", bool((ties >= 0).all()))
    c.check("tie total matches the summary", int(ties.sum()) == checks["tied_tracked_documents"])
    c.check("scores are finite and below two", bool((np.abs(score) < 2).all()))
    grid = score * 2.0**48
    c.check("scores are multiples of 2^-48", bool((grid == np.rint(grid)).all()))
    kept = rank > 0
    ordered = np.lexsort((rank[kept], -score[kept], mask_of[kept]))
    masks, ranks = mask_of[kept][ordered], rank[kept][ordered]
    same_mask = masks[1:] == masks[:-1]
    c.check(
        "within a mask a higher score has a better rank",
        bool((ranks[1:][same_mask] > ranks[:-1][same_mask]).all()),
    )


def check_per_query(
    c: Checks,
    arrays: Mapping[str, np.ndarray],
    counts: np.ndarray,
    budgets: Sequence[int],
    excluded: int,
) -> None:
    """Pairing and eligibility of the per-query arrays the report reads, over every query."""
    last = len(budgets)
    full = ROWS.index("full")
    c.check("stored positive counts", _same(arrays["counts"], counts))
    c.check(
        "array shapes", all(arrays[n].shape == (len(ROWS), last + 1, counts.size) for n in MEASURES)
    )
    for row in ("prefix", "shared"):
        at = ROWS.index(row)
        for name in TRACKED_SAME_AS_FULL:
            base = np.broadcast_to(
                arrays[name][full, last], arrays[f"base_{name}"][at, :last].shape
            )
            c.check(
                f"{row}: baseline {name} is the full query",
                _same(arrays[f"base_{name}"][at, :last], base),
            )
    c.check(
        "only the full row fills the last slot",
        all(np.isnan(arrays["rr"][r, last]).all() for r in range(len(ROWS)) if r != full),
    )
    for name, construct, held in CELL_SIZES:
        at = ROWS.index(name)
        for base in ("base_rr", "base_logrank", "base_recall10", "base_recall100"):
            c.check(
                f"{name}: {base} does not vary with k",
                all(_same(arrays[base][at, s], arrays[base][at, 0]) for s in range(last)),
            )
        if excluded == 0:
            eligible = counts >= construct + held
            c.check(
                f"{name}: defined exactly on eligible queries",
                _same(np.isfinite(arrays["rr"][at, :last]).all(axis=0), eligible)
                and not np.isfinite(arrays["rr"][at, :last, ~eligible]).any(),
            )
            c.check(
                f"{name}: evaluates only held-out positives",
                bool((arrays["evaluated"][at, :last, eligible] == held).all()),
            )
    if excluded == 0:
        shared = ROWS.index("shared")
        c.check(
            "shared evaluates every positive",
            _same(arrays["evaluated"][shared, 0].astype(np.int64), counts),
        )


def spot_queries(
    ids: Sequence[str], strata: np.ndarray, counts: np.ndarray, seed: int, per_stratum: int
) -> list[int]:
    """A fixed sample: the first queries of each stratum by a seeded hash, plus the first query
    eligible for each cell, so that every arm is reached."""
    ranked = sorted(
        range(len(ids)),
        key=lambda at: hashlib.sha256(f"{seed}:integrity:{ids[at]}".encode()).hexdigest(),
    )
    chosen: set[int] = set()
    for stratum in np.unique(strata):
        chosen.update([at for at in ranked if strata[at] == stratum][:per_stratum])
    for _, construct, held in CELL_SIZES:
        eligible = [at for at in ranked if counts[at] >= construct + held]
        if eligible:
            chosen.add(eligible[0])
    return sorted(chosen)


def _scan(
    documents: np.ndarray,
    masked: np.ndarray,
    tracked: Sequence[np.ndarray],
    scores: Sequence[np.ndarray],
    excluded: np.ndarray,
    lists: np.ndarray,
) -> dict[str, Any]:
    """Rank and top lists of every mask, one document chunk and one mask at a time."""
    count = masked.shape[0]
    higher = [np.zeros(t.size, dtype=np.int64) for t in tracked]
    lower = [np.zeros(t.size, dtype=np.int64) for t in tracked]
    equal = [np.zeros(t.size, dtype=np.int64) for t in tracked]
    top: list[tuple[np.ndarray, np.ndarray]] = [
        (np.empty(0), np.empty(0, dtype=np.int64)) for _ in range(count)
    ]
    total, squares = np.zeros(count), np.zeros(count)
    for start in range(0, documents.shape[0], CHUNK_DOCS):
        chunk = np.asarray(documents[start : start + CHUNK_DOCS], dtype=np.float64)
        all_scores = masked @ chunk.T
        total += all_scores.sum(axis=1)
        squares += np.square(all_scores).sum(axis=1)
        for at in range(count):
            here = all_scores[at]
            offset = excluded[at] - start
            if 0 <= offset < here.size:
                here[offset] = -np.inf
            bound = scores[at]
            higher[at] += (here[:, None] > bound[None]).sum(axis=0)
            docs, positions = np.nonzero(here[:, None] == bound[None])
            np.add.at(equal[at], positions, 1)
            np.add.at(
                lower[at], positions, (start + docs < tracked[at][positions]).astype(np.int64)
            )
            if lists[at]:
                take = min(DEPTH, here.size)
                cut = np.partition(here, here.size - take)[here.size - take]
                new = np.flatnonzero((here >= cut) & np.isfinite(here))
                old_scores, old_rows = top[at]
                merged_scores = np.concatenate([old_scores, here[new]])
                merged_rows = np.concatenate([old_rows, start + new])
                keep = np.lexsort((merged_rows, -merged_scores))[:DEPTH]
                top[at] = (merged_scores[keep], merged_rows[keep])
    seen = documents.shape[0]
    return {
        "rank": [1 + h + w for h, w in zip(higher, lower, strict=True)],
        "ties": [e - 1 for e in equal],
        "top": top,
        "corpus_mean": total / seen,
        "corpus_var": squares / seen - (total / seen) ** 2,
    }


def spot_check(
    c: Checks,
    e: Environment,
    ids: list[str],
    counts: np.ndarray,
    budgets: Sequence[int],
    seed: int,
    chosen: list[int],
    m: Mapping[str, np.ndarray],
    t: Mapping[str, np.ndarray],
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Rebuild the masks of ``chosen`` queries and hold them to what the run stored."""
    inputs = e.inputs
    documents = inputs.documents
    doc_rows = {doc: row for row, doc in enumerate(inputs.doc_ids)}
    query = [np.asarray(inputs.query_vectors[inputs.query_rows[ids[at]]]) for at in chosen]
    positive = [e.positive[ids[at]] for at in chosen]
    grades = [e.grades[ids[at]] for at in chosen]
    held_docs = [gather(documents, rows) for rows in positive]
    sources = []
    for local, at in enumerate(chosen):
        sources.extend(
            plan_query(
                local,
                ids[at],
                np.asarray(query[local], dtype=np.float16),
                held_docs[local],
                grades[local],
                seed=seed,
                device="cpu",
            )
        )
    plan = expand(sources, budgets, counts[chosen])
    wanted = np.isin(m["query"], chosen)
    keys = zip(
        m["query"][wanted].tolist(),
        m["row"][wanted].tolist(),
        m["partition"][wanted].tolist(),
        m["positive"][wanted].tolist(),
        m["budget"][wanted].tolist(),
        np.flatnonzero(wanted).tolist(),
        strict=True,
    )
    stored_at = {key[:5]: key[5] for key in keys}
    index = np.empty(plan.source.size, dtype=np.int64)
    missing = 0
    for mask, source_at in enumerate(plan.source):
        s = plan.sources[source_at]
        key = (chosen[s.query], s.row, s.partition, s.positive, int(plan.budget[mask]))
        found = stored_at.get(key)
        if found is None:
            missing += 1
        else:
            index[mask] = found
    c.check("spot: every rebuilt mask is stored", missing == 0)
    c.check("spot: no stored mask is unaccounted for", len(stored_at) == plan.source.size)

    offsets = t["offsets"]
    rebuilt_offsets = np.zeros(index.size + 1, dtype=np.int64)
    rebuilt_offsets[1:] = np.cumsum(offsets[index + 1] - offsets[index])
    stored = [np.arange(offsets[i], offsets[i + 1]) for i in index]
    rows_stored = np.concatenate(stored)
    position = t["position"][rows_stored]
    rank = t["rank"][rows_stored]
    score = t["score"][rows_stored]
    ties = t["ties"][rows_stored]

    masked = np.zeros((index.size, DIM), dtype=np.float64)
    tracked_rows: list[np.ndarray] = []
    tracked_scores: list[np.ndarray] = []
    scores_equal = masks_equal = equivalent = True
    for mask, source_at in enumerate(plan.source):
        s = plan.sources[source_at]
        q = np.asarray(query[s.query], dtype=np.float16)
        slot = int(plan.budget[mask])
        k = DIM if slot == FULL_BUDGET else int(budgets[slot])
        vector = masked_vector(q, s.order, k)
        if s.order is not None:
            chosen_dims = s.order[:k]
            kept = np.zeros(DIM, dtype=bool)
            kept[chosen_dims] = True
            masks_equal &= bool(
                sorted(s.order.tolist()) == list(range(DIM))
                and np.unique(chosen_dims).size == k
                and np.array_equal(vector[kept], q[kept])
                and not vector[~kept].any()
            )
            equivalent &= bool(np.array_equal(masked_vector(q, s.order, DIM), q))
        masked[mask] = vector
        low, high = rebuilt_offsets[mask], rebuilt_offsets[mask + 1]
        positions = position[low:high]
        rows = positive[s.query][positions]
        expected = masked[mask] @ np.asarray(held_docs[s.query][positions], dtype=np.float64).T
        scores_equal &= bool(np.array_equal(expected, score[low:high]))
        tracked_rows.append(rows)
        tracked_scores.append(score[low:high])
    c.check("spot: masks keep exactly k coordinates of the query", masks_equal)
    c.check("spot: k = 1024 is the full query, bit for bit, for every order", equivalent)
    c.check("spot: stored scores equal the exact dot product", scores_equal)

    partitions = disjoint = True
    for s in sources:
        if s.held is None or s.construction is None:
            continue
        count = positive[s.query].size
        built, left = s.construction.tolist(), s.held.tolist()
        partitions &= set(built) <= set(range(count)) and set(left) <= set(range(count))
        disjoint &= not set(built) & set(left)
    c.check("spot: partitions index only training positives of the query", partitions)
    c.check("spot: construction and held-out documents are disjoint", disjoint)
    c.check(
        "spot: a mask ignores documents outside its construction set",
        _ignores_held_out(ids, chosen, query, held_docs, grades, seed),
    )

    picked = _picked(plan)
    excluded = np.array(
        [doc_rows.get(ids[chosen[plan.sources[plan.source[i]].query]], -1) for i in picked],
        dtype=np.int64,
    )
    lists = np.array([plan.sources[plan.source[i]].row != ROWS.index("own") for i in picked])
    scanned = _scan(
        documents,
        masked[picked],
        [tracked_rows[i] for i in picked],
        [tracked_scores[i] for i in picked],
        excluded,
        lists,
    )
    ranks_ok = ties_ok = lists_ok = at_ok = True
    worst_mean = 0.0
    for at, mask in enumerate(picked):
        low, high = rebuilt_offsets[mask], rebuilt_offsets[mask + 1]
        competing = tracked_rows[mask] != excluded[at]
        ranks_ok &= bool(
            np.array_equal(np.where(competing, scanned["rank"][at], 0), rank[low:high])
        )
        ties_ok &= bool(np.array_equal(np.where(competing, scanned["ties"][at], 0), ties[low:high]))
        source = plan.sources[plan.source[mask]]
        stored_index = index[mask]
        worst_mean = max(
            worst_mean,
            abs(scanned["corpus_mean"][at] - m["corpus_mean"][stored_index]),
            abs(scanned["corpus_var"][at] - m["corpus_var"][stored_index]),
        )
        if not lists[at]:
            continue
        top_scores, top_rows = scanned["top"][at]
        expected_at = np.array(
            [top_scores[r - 1] if top_scores.size >= r else np.nan for r in SCORE_RANKS]
        )
        at_ok &= _same(expected_at, m["score_at"][stored_index])
        judged = e.judgments[ids[chosen[source.query]]]
        values = query_metrics([inputs.doc_ids[r] for r in top_rows], judged, CUTOFFS)
        for name in LIST_METRICS:
            lists_ok &= bool(np.float32(values[name]) == m[name.replace("@", "_at_")][stored_index])
    c.check("spot: full-corpus ranks agree exactly", ranks_ok)
    c.check("spot: tie counts agree exactly", ties_ok)
    c.check("spot: list metrics from rescored top lists agree", lists_ok)
    c.check("spot: scores at ranks 1, 10, 100 agree exactly", at_ok)
    c.check("spot: corpus mean and variance agree", worst_mean <= MEAN_TOLERANCE)

    results: dict[str, np.ndarray] = {
        "position": position,
        "rank": rank,
        "score": score,
        "corpus_mean": m["corpus_mean"][index],
        "corpus_var": m["corpus_var"][index],
        "score_at": m["score_at"][index],
        **{name: m[name.replace("@", "_at_")][index] for name in LIST_METRICS},
    }
    rebuilt = per_query_summary(plan, results, rebuilt_offsets, len(chosen), budgets, grades)
    c.check(
        "spot: per-query arrays follow from stored ranks and masks",
        all(
            _same(rebuilt[name].astype(np.float32), arrays[name][:, :, chosen]) for name in MEASURES
        ),
    )
    return {
        "queries": len(chosen),
        "masks_rebuilt": int(index.size),
        "masks_rescored": int(len(picked)),
        "by_arm": {
            ROWS[r]: int(sum(plan.sources[plan.source[i]].row == r for i in picked))
            for r in range(len(ROWS))
        },
        "budgets_rescored": sorted({int(plan.budget[i]) for i in picked}),
        "positives_in_sample": [int(counts[at]) for at in chosen],
        "largest_corpus_statistic_gap": float(worst_mean),
    }


def _picked(plan: Any) -> list[int]:
    """Masks rescored over the corpus: the full query, and per budget the prefix, the shared mask,
    the first positive's own mask, and the first partition of each cell."""
    return [
        mask
        for mask, source_at in enumerate(plan.source)
        if plan.sources[source_at].positive <= 0 and plan.sources[source_at].partition <= 0
    ]


def _ignores_held_out(
    ids: Sequence[str],
    chosen: Sequence[int],
    query: Sequence[np.ndarray],
    docs: Sequence[np.ndarray],
    grades: Sequence[np.ndarray],
    seed: int,
) -> bool:
    """Blanking every document outside a partition's construction set leaves that partition's
    mask unchanged: nothing else reaches it."""

    def cells_of(local: int, vectors: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
        planned = plan_query(
            local,
            ids[chosen[local]],
            np.asarray(query[local], dtype=np.float16),
            vectors,
            grades[local],
            seed=seed,
            device="cpu",
            own=False,
        )
        return {(s.row, s.partition): s for s in planned if s.held is not None}

    for local in range(len(chosen)):
        original = cells_of(local, docs[local])
        for key, source in original.items():
            if key[1] != 0:
                continue
            blank = np.zeros_like(docs[local])
            blank[source.construction] = docs[local][source.construction]
            if not np.array_equal(cells_of(local, blank)[key].order, source.order):
                return False
    return True


def paired_primary(
    environments: Mapping[str, Mapping[str, np.ndarray]],
    strata: Mapping[str, np.ndarray],
    budgets: Sequence[int],
    claims_csv: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Recompute the primary family from per-query arrays, apart from the report's helpers, and
    compare each row with the claims table the report wrote."""
    row = ROWS.index("1-1")
    rows: list[dict[str, Any]] = []
    for offset, (env, arrays) in enumerate(environments.items()):
        resamples = resampled_queries(strata[env], RESAMPLES, offset)
        for slot, k in enumerate(budgets):
            for value, label in PRIMARY_MEASURES:
                delta = arrays[value][row, slot] - arrays[f"base_{value}"][row, slot]
                kept = np.isfinite(delta)
                paired = delta[kept]
                means = np.nanmean(delta[resamples], axis=1)
                better = paired < 0 if value == "logrank" else paired > 0
                worse = paired > 0 if value == "logrank" else paired < 0
                rows.append(
                    {
                        "environment": env,
                        "k": k,
                        "measure": label,
                        "queries": int(kept.sum()),
                        "difference": float(paired.mean()),
                        "low": float(np.quantile(means, 0.025)),
                        "high": float(np.quantile(means, 0.975)),
                        "queries_better": float(better.mean()),
                        "queries_worse": float(worse.mean()),
                        "p": paired_randomization_test(
                            list(paired), [0.0] * paired.size, PERMUTATIONS, seed=0
                        ),
                    }
                )
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p"])
    running = 0.0
    for place, i in enumerate(order):
        running = max(running, min(1.0, (len(rows) - place) * rows[i]["p"]))
        rows[i]["p_holm"] = running
    return rows, _compare_with_claims(rows, claims_csv)


def _printed(text: str, value: float) -> bool:
    """The report's four-decimal number equals ``value``, so that -0.0000 matches 0.0000."""
    return float(text) == float(f"{value:.4f}")


def _compare_with_claims(rows: Sequence[Mapping[str, Any]], claims_csv: Path) -> list[str]:
    with claims_csv.open(encoding="utf-8", newline="") as handle:
        table = [r for r in csv.DictReader(handle) if r["family"].startswith("primary")]
    if len(table) != len(rows):
        return [f"claims table has {len(table)} primary rows, recomputed {len(rows)}"]
    found = {(r["environment"], r["k"], r["measure"]): r for r in table}
    failed = []
    for row in rows:
        key = (row["environment"], str(row["k"]), row["measure"])
        reported = found.get(key)
        expected = reported is not None and (
            reported["queries"] == str(row["queries"])
            and _printed(reported["difference from the full query"], row["difference"])
            and all(
                _printed(text, value)
                for text, value in zip(
                    reported["95% interval (pointwise, exploratory)"].split(" to "),
                    (row["low"], row["high"]),
                    strict=True,
                )
            )
            and _printed(reported["p (sign-flip)"], row["p"])
            and _printed(reported["p (Holm, within family)"], row["p_holm"])
        )
        if not expected:
            failed.append(f"claims differ at {key}")
    return failed


def check_run(root: Path, run: Path, per_stratum: int, report: Path | None) -> dict[str, Any]:
    """All checks of ``run``; ``root`` holds the inputs the run was made from."""
    sources = json.loads((run / "inputs.json").read_text(encoding="utf-8"))
    seed, budgets = int(sources["seed"]), [int(k) for k in sources["budgets"]]
    environments: dict[str, Any] = {}
    failed: list[str] = []
    arrays_of: dict[str, dict[str, np.ndarray]] = {}
    strata_of: dict[str, np.ndarray] = {}
    for env in sources["environments"]:
        c = Checks(env)
        e = _environment(root, env)
        record = sources[env]
        c.check(
            "training judgments hash",
            record["qrels_train_sha256"] == file_sha256(e.inputs.dataset_dir / qrels_file("train")),
        )
        for part, name in (("documents", "documents_sha256"), ("queries", "queries_sha256")):
            c.check(
                f"{part} hash", record[name] == file_sha256(e.inputs.embeddings_dir / f"{part}.npy")
            )
        c.check("embeddings of the run", record["embeddings"] == e.inputs.embeddings_dir.name)
        c.check("dataset of the run", record["dataset"] == e.inputs.dataset_dir.name)
        ids = json.loads((run / env / "selected-queries.json").read_text(encoding="utf-8"))
        summary = json.loads((run / env / "summary.json").read_text(encoding="utf-8"))
        with np.load(run / env / "selection.npz") as found:
            strata, weights = found["strata"], found["weights"]
        counts = np.array([e.multi[qid] for qid in ids], dtype=np.int64)
        check_selection(c, e, ids, strata, weights, seed)
        with np.load(run / env / "masks.npz") as found:
            m = {name: found[name] for name in found.files}
        with np.load(run / env / "tracked.npz") as found:
            t = {name: found[name] for name in found.files}
        with np.load(run / env / "per-query.npz") as found:
            arrays = {name: found[name] for name in found.files}
        check_masks(c, m, counts, budgets, summary)
        check_tracked(c, m, t, counts, e.inputs.documents.shape[0], summary)
        check_per_query(c, arrays, counts, budgets, summary["checks"]["excluded_positives"])
        chosen = spot_queries(ids, strata, counts, seed, per_stratum)
        spot = spot_check(c, e, ids, counts, budgets, seed, chosen, m, t, arrays)
        environments[env] = {
            "checks_passed": c.passed,
            "failed": c.failed,
            "queries": len(ids),
            "positives": int(counts.sum()),
            "documents": int(e.inputs.documents.shape[0]),
            "masks": int(m["row"].size),
            "strata": summary["coverage"]["selection"],
            "excluded_positives": summary["checks"]["excluded_positives"],
            "tied_tracked_documents": summary["checks"]["tied_tracked_documents"],
            "spot": spot,
        }
        failed.extend(c.failed)
        arrays_of[env] = arrays
        strata_of[env] = strata
    found_paired: dict[str, Any] = {}
    if report is not None:
        rows, differing = paired_primary(
            arrays_of, strata_of, budgets, report / "tables/claims.csv"
        )
        failed.extend(differing)
        found_paired = {"primary": rows, "differing_from_report": differing}
    return {"run": run.name, "environments": environments, "paired": found_paired, "failed": failed}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("run")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--per-stratum", type=int, default=PER_STRATUM)
    args = parser.parse_args(argv)
    run = args.root / "evaluation/P3-mask-ranking/all" / args.run
    script_hash = file_sha256(Path(__file__))
    manifest = start_manifest(
        "p3-ranking-integrity",
        {"run": args.run, "per_stratum": args.per_stratum, "script_sha256": script_hash},
        seed=0,
        dataset_hash=NO_DATASET,
    )
    found = check_run(args.root, run, args.per_stratum, args.report_dir)
    found["script_sha256"] = script_hash
    found["run_commit"] = read_manifest(run).git_commit
    with ArtifactWriter(
        args.output, "evaluation/P3-mask-ranking-integrity/all", manifest
    ) as writer:
        text = json.dumps(found, indent=2, sort_keys=True, allow_nan=False) + "\n"
        writer.path("integrity.json").write_text(text, encoding="utf-8")
        print(writer.commit())
    print(
        json.dumps(
            {
                "failed": found["failed"],
                "passed": {
                    env: value["checks_passed"] for env, value in found["environments"].items()
                },
            }
        )
    )
    return 1 if found["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
