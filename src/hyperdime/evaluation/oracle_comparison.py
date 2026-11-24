"""Exact positive masks beside Learning-to-Select on the same training queries (ADR-0017).

Every arm of a query is scored on the same corpus by the exact float64 scan of ADR-0016, and no
document is excluded by id. The all-positive oracle is a privileged ceiling for the mean similarity
of the query's positives and nothing else: it is not a ranking optimum. Its masks saw the documents
they are scored on, and the L2S numbers are in-sample, because oracle targets may only come from
training judgments. Training judgments only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from hyperdime.evaluation.metrics import query_metrics
from hyperdime.evaluation.representation import budget_masks, query_agreement, relevance_gains
from hyperdime.evaluation.representation_controls import query_stream
from hyperdime.evaluation.representation_engine import Clock, gather, stable_order
from hyperdime.evaluation.representation_ranking import (
    BATCH_MASKS,
    BATCH_TRACKED,
    CHUNK_DOCS,
    CUTOFFS,
    DEPTH,
    DIM,
    LIST_METRICS,
    SCORE_RANKS,
    ScanBatch,
    masked_vector,
    scan_corpus,
    tracked_scores,
)
from hyperdime.evaluation.representation_shared import (
    PARTITIONS,
    TRANSFER_CELLS,
    masses,
    optimum_curves,
    partitions,
    scores_under,
    shared_order,
)

ARMS = ("full", "prefix", "oracle", "held", "l2s", "own")
TRACKING_ALL = ("full", "prefix", "oracle", "l2s")  # arms that track every positive of the query
POSITIVE_MEASURES = ("rr", "logrank", "recall10", "recall100", "score", "regret")
HELD_MEASURES = ("held_rr", "held_logrank", "held_recall10", "held_recall100", "held_score")
COMPETITOR_MEASURES = tuple(f"at{rank}" for rank in SCORE_RANKS)
RELATION_MEASURES = (
    "pair_jaccard", "sign_consensus", "positive_consensus", "oracle_regret", "held_regret",
    "held_privileged_regret", "held_oracle_jaccard",
)  # fmt: skip
ONE_ONE = next(at for at, cell in enumerate(TRANSFER_CELLS) if cell[0] == "1-1")


def _arm(name: str) -> int:
    return ARMS.index(name)


@dataclass(frozen=True)
class Source:
    """One mask ordering of a query."""

    query: int
    arm: int
    partition: int = -1
    positive: int = -1
    order: np.ndarray | None = None


def pair_order(query: np.ndarray, document: np.ndarray) -> np.ndarray:
    """One positive's dimensions by descending ``q * d``: the top ``k`` is its pair-optimal mask."""
    values = torch.from_numpy(np.asarray(query, np.float64) * np.asarray(document, np.float64))
    return stable_order(values[None])[0].numpy().astype(np.int16)


def oracle_order(query: np.ndarray, docs: np.ndarray, grades: np.ndarray) -> np.ndarray:
    """Dimensions by descending relevance-weighted mean of ``q * d`` over the query's positives."""
    values = torch.from_numpy(np.asarray(query, np.float64) * np.asarray(docs, np.float64))
    gains = torch.from_numpy(relevance_gains(grades)).to(torch.float64)
    return shared_order(values[None], gains[None])[0].numpy().astype(np.int16)


def one_one_partitions(count: int, seed: int, qid: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """ADR-0015's 1-1 partitions of a query's positives, from the same stream."""
    return partitions(count, 1, 1, query_stream(seed, 400 + ONE_ONE, qid), PARTITIONS)


def plan_query(
    index: int,
    qid: str,
    query: np.ndarray,
    docs: np.ndarray,
    grades: np.ndarray,
    importance: np.ndarray,
    seed: int,
) -> tuple[list[Source], list[tuple[np.ndarray, np.ndarray]], bool]:
    """Orderings of one query, its 1-1 partitions, and whether a lone positive's mask is the
    oracle's; a single positive's own mask is not planned, since it would repeat the oracle's."""
    count = docs.shape[0]
    oracle = oracle_order(query, docs, grades)
    l2s = stable_order(torch.from_numpy(importance)[None])[0].numpy().astype(np.int16)
    sources = [
        Source(index, _arm("full")),
        Source(index, _arm("prefix"), order=np.arange(DIM, dtype=np.int16)),
        Source(index, _arm("oracle"), order=oracle),
        Source(index, _arm("l2s"), order=l2s),
    ]
    if count == 1:
        return sources, [], bool(np.array_equal(pair_order(query, docs[0]), oracle))
    splits = one_one_partitions(count, seed, qid)
    for slot, (construct, _) in enumerate(splits):
        order = oracle_order(query, docs[construct], grades[construct])
        sources.append(Source(index, _arm("held"), partition=slot, order=order))
    sources.extend(
        Source(index, _arm("own"), positive=at, order=pair_order(query, docs[at]))
        for at in range(count)
    )
    return sources, splits, True


def relationship(
    query: np.ndarray,
    docs: np.ndarray,
    grades: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    budgets: Sequence[int],
) -> dict[str, np.ndarray]:
    """How one multi-positive query's positives relate at each budget."""
    values = torch.from_numpy(np.asarray(query, np.float64) * np.asarray(docs, np.float64))
    optimum, _ = optimum_curves(values)
    mass = masses(values)
    positions = np.array([k - 1 for k in budgets])
    gains = torch.from_numpy(relevance_gains(grades)).to(torch.float64)
    oracle = shared_order(values[None], gains[None])
    own = stable_order(values).numpy()
    agreement = query_agreement(values.numpy(), own, grades, budgets)
    found = {name: np.full(len(budgets), np.nan) for name in RELATION_MEASURES}
    found["pair_jaccard"] = agreement["mean_jaccard"]
    all_positive = (values > 0).all(dim=0).numpy()
    all_negative = (values < 0).all(dim=0).numpy()
    oracle_regret = (
        optimum[:, positions] - scores_under(values[None], oracle)[0][:, positions]
    ) / mass[:, None]
    found["oracle_regret"] = oracle_regret.mean(dim=0).numpy()
    for slot, k in enumerate(budgets):
        kept = budget_masks(oracle.numpy(), k)[0]
        found["sign_consensus"][slot] = (all_positive | all_negative)[kept].mean()
        found["positive_consensus"][slot] = all_positive[kept].mean()
    if not splits:
        return found
    built = shared_order(
        values[torch.from_numpy(np.stack([c for c, _ in splits]))],
        torch.from_numpy(np.stack([relevance_gains(grades[c]) for c, _ in splits])).to(
            torch.float64
        ),
    )
    held_rows = np.stack([h for _, h in splits])
    held_values = values[torch.from_numpy(held_rows)]
    held_scores = scores_under(held_values, built)
    privileged = scores_under(held_values, oracle.expand(len(splits), -1))
    held_optimum = optimum[held_rows]
    held_mass = mass[held_rows]
    held_regret = (held_optimum - held_scores)[:, :, positions] / held_mass[:, :, None]
    privileged_regret = (held_optimum - privileged)[:, :, positions] / held_mass[:, :, None]
    found["held_regret"] = held_regret.mean(dim=1).mean(dim=0).numpy()
    found["held_privileged_regret"] = privileged_regret.mean(dim=1).mean(dim=0).numpy()
    for slot, k in enumerate(budgets):
        held = budget_masks(built.numpy(), k)
        reference = budget_masks(oracle.numpy(), k)[0]
        intersection = (held & reference).sum(axis=1)
        found["held_oracle_jaccard"][slot] = (intersection / (2 * k - intersection)).mean()
    return found


def _batches(tracked: np.ndarray, depth: np.ndarray, batch_masks: int) -> list[np.ndarray]:
    """Masks by retrieval need, then tracked count; masks times tracked documents stay bounded."""
    order = np.lexsort((tracked, depth))
    found: list[np.ndarray] = []
    at = 0
    while at < order.size:
        take = min(batch_masks, order.size - at)
        members = order[at : at + take]
        while take > 1 and (
            take * int(tracked[members].max()) > BATCH_TRACKED or np.unique(depth[members]).size > 1
        ):
            take = max(1, take // 2)
            members = order[at : at + take]
        found.append(members)
        at += take
    return found


def _scan(
    sources: Sequence[Source],
    mask_source: np.ndarray,
    mask_slot: np.ndarray,
    query_vector: Callable[[int], np.ndarray],
    positives: Sequence[np.ndarray],
    colliding: np.ndarray,
    documents: np.ndarray,
    budgets: Sequence[int],
    device: str,
    chunk_docs: int,
    batch_masks: int,
    clock: Clock,
    log: Callable[[str], None] | None,
) -> dict[str, np.ndarray]:
    """Scan the corpus under every mask; nothing is excluded by id."""
    own = _arm("own")
    tracked_count = np.fromiter(
        (1 if sources[s].arm == own else positives[sources[s].query].size for s in mask_source),
        dtype=np.int32,
        count=mask_source.size,
    )
    depth = np.fromiter(
        (0 if sources[s].arm == own else DEPTH for s in mask_source),
        dtype=np.int32,
        count=mask_source.size,
    )
    groups = _batches(tracked_count, depth, batch_masks)
    batches: list[ScanBatch] = []
    collision_score = np.full(mask_source.size, np.nan)
    with clock.span("thresholds"):
        for members in groups:
            width = int(tracked_count[members].max())
            masked = np.zeros((members.size, DIM), dtype=np.float16)
            tracked = np.full((members.size, width), -1, dtype=np.int64)
            for at, mask in enumerate(members):
                source = sources[mask_source[mask]]
                vector = np.asarray(query_vector(source.query), dtype=np.float16)
                masked[at] = masked_vector(vector, source.order, int(budgets[mask_slot[mask]]))
                rows = positives[source.query]
                if source.arm == own:
                    tracked[at, 0] = rows[source.positive]
                else:
                    tracked[at, : rows.size] = rows
            same_id = colliding[[sources[mask_source[m]].query for m in members]]
            if (same_id >= 0).any():
                exact = tracked_scores(masked, same_id[:, None], documents, device)[:, 0]
                collision_score[members] = np.where(same_id >= 0, exact, np.nan)
            batches.append(
                ScanBatch(
                    masked,
                    tracked,
                    tracked_scores(masked, tracked, documents, device),
                    np.full(members.size, -1, dtype=np.int64),
                    int(depth[members[0]]),
                )
            )
    scan_corpus(documents, batches, device, chunk_docs, clock, log)
    offsets = np.zeros(mask_source.size + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(tracked_count)
    rank = np.zeros(int(offsets[-1]), dtype=np.int64)
    score = np.zeros(int(offsets[-1]), dtype=np.float64)
    ties = np.zeros(int(offsets[-1]), dtype=np.int32)
    score_at = np.full((mask_source.size, len(SCORE_RANKS)), np.nan)
    top_rows = np.full((mask_source.size, DEPTH), -1, dtype=np.int32)
    for members, batch in zip(groups, batches, strict=True):
        done = batch.finish()
        for at, mask in enumerate(members):
            start, end = offsets[mask], offsets[mask + 1]
            rank[start:end] = done["rank"][at, : end - start]
            score[start:end] = batch.scores[at, : end - start]
            ties[start:end] = done["ties"][at, : end - start]
            if batch.depth:
                values = done["top_scores"][at][[r - 1 for r in SCORE_RANKS]]
                score_at[mask] = np.where(np.isinf(values), np.nan, values)
                top_rows[mask] = done["top_rows"][at]
    return {
        "offsets": offsets,
        "rank": rank,
        "score": score,
        "ties": ties,
        "score_at": score_at,
        "top_rows": top_rows,
        "collision_score": collision_score,
    }


def _collisions(
    sources: Sequence[Source],
    mask_source: np.ndarray,
    results: Mapping[str, np.ndarray],
    rows_of: Sequence[np.ndarray],
    colliding: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Which tracked ranks each same-id document charged, and that it really competed: one scoring
    above rank 100 must be in the retrieved list, one below it must not."""
    offsets, score = results["offsets"], results["score"]
    charged = np.zeros(score.size, dtype=bool)
    masks_with = inconsistent = 0
    last = len(SCORE_RANKS) - 1
    for mask, source_at in enumerate(mask_source):
        source = sources[source_at]
        row = int(colliding[source.query])
        if row < 0:
            continue
        masks_with += 1
        value = results["collision_score"][mask]
        rows = rows_of[source.query]
        tracked = rows[[source.positive]] if source.arm == _arm("own") else rows
        here = score[offsets[mask] : offsets[mask + 1]]
        charged[offsets[mask] : offsets[mask + 1]] = (value > here) | (
            (value == here) & (row < tracked)
        )
        if source.arm == _arm("own"):
            continue
        listed = bool((results["top_rows"][mask] == row).any())
        at_last = results["score_at"][mask, last]
        if np.isnan(at_last) or value > at_last:
            inconsistent += int(not listed)
        elif value < at_last:
            inconsistent += int(listed)
    return charged, {
        "masks_with_same_id_document": masks_with,
        "same_id_list_inconsistencies": inconsistent,
        "tracked_ranks_charged_by_same_id_document": int(charged.sum()),
    }


@dataclass
class Comparison:
    """Per-query arrays ``[arm, budget, query]``, with what the checks found."""

    arrays: dict[str, np.ndarray]
    relation: dict[str, np.ndarray]
    tracked: dict[str, np.ndarray]
    masks: dict[str, np.ndarray]
    checks: dict[str, Any]


def _position_measures(ranks: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "rr": float((1 / ranks).mean()),
        "logrank": float(np.log(ranks).mean()),
        "recall10": float((ranks <= 10).mean()),
        "recall100": float((ranks <= 100).mean()),
        "score": float(scores.mean()),
    }


def _empty_arrays(queries: int, budgets: int) -> dict[str, np.ndarray]:
    names = (
        *LIST_METRICS, *POSITIVE_MEASURES, *HELD_MEASURES, *COMPETITOR_MEASURES, "jaccard_oracle",
    )  # fmt: skip
    return {name: np.full((len(ARMS), budgets, queries), np.nan) for name in names}


def compare_masks(
    query_ids: Sequence[str],
    query_vector: Callable[[str], np.ndarray],
    documents: np.ndarray,
    doc_ids: Sequence[str],
    positives: Mapping[str, np.ndarray],
    grades: Mapping[str, np.ndarray],
    judgments: Mapping[str, Mapping[str, int]],
    importance: np.ndarray,
    *,
    seed: int,
    budgets: Sequence[int],
    device: str,
    clock: Clock,
    chunk_docs: int = CHUNK_DOCS,
    batch_masks: int = BATCH_MASKS,
    log: Callable[[str], None] | None = None,
) -> Comparison:
    """Rank the corpus under every arm of every query; the last budget must be 1024."""
    if documents.dtype != np.float16:
        raise ValueError("exact float64 scoring needs float16 documents")
    if budgets[-1] != DIM:
        raise ValueError("the grid ends at the full query")
    full_slot = len(budgets) - 1
    query_ids = list(query_ids)
    all_sources: list[Source] = []
    per_query_splits: list[list[tuple[np.ndarray, np.ndarray]]] = []
    position_of: list[np.ndarray] = []
    relation = {name: np.full((len(budgets), len(query_ids)), np.nan) for name in RELATION_MEASURES}
    single_identity = 0
    singles = 0
    rows_of = [positives[qid] for qid in query_ids]
    row_of_doc = {doc: row for row, doc in enumerate(doc_ids)}
    colliding = np.array([row_of_doc.get(qid, -1) for qid in query_ids], dtype=np.int64)
    with clock.span("plan"):
        for index, qid in enumerate(query_ids):
            vector = np.asarray(query_vector(qid), dtype=np.float16)
            docs = gather(documents, positives[qid])
            sources, splits, identity = plan_query(
                index, qid, vector, docs, grades[qid], importance[index], seed
            )
            all_sources.extend(sources)
            per_query_splits.append(splits)
            position_of.append(rows_of[index])
            if docs.shape[0] == 1:
                singles += 1
                single_identity += int(identity)
            else:
                found = relationship(vector, docs, grades[qid], splits, budgets)
                for name, values in found.items():
                    relation[name][:, index] = values
    mask_source_list: list[int] = []
    mask_slot_list: list[int] = []
    for at, source in enumerate(all_sources):
        slots = [full_slot] if source.arm == _arm("full") else range(len(budgets))
        for slot in slots:
            mask_source_list.append(at)
            mask_slot_list.append(slot)
    mask_source = np.array(mask_source_list, dtype=np.int64)
    mask_slot = np.array(mask_slot_list, dtype=np.int32)
    vectors = [np.asarray(query_vector(qid), dtype=np.float16) for qid in query_ids]
    results = _scan(
        all_sources, mask_source, mask_slot, vectors.__getitem__, rows_of, colliding, documents,
        budgets, device, chunk_docs, batch_masks, clock, log,
    )  # fmt: skip
    with clock.span("metrics"):
        arrays, checks, list_values = _aggregate(
            all_sources, mask_source, mask_slot, results, query_ids, doc_ids, judgments, rows_of,
            [grades[qid] for qid in query_ids], per_query_splits, vectors, documents, budgets,
        )  # fmt: skip
        charged, collision_checks = _collisions(
            all_sources, mask_source, results, rows_of, colliding
        )
    columns: dict[str, np.ndarray] = {
        "query": np.array([all_sources[s].query for s in mask_source], dtype=np.int32),
        "arm": np.array([all_sources[s].arm for s in mask_source], dtype=np.int8),
        "partition": np.array([all_sources[s].partition for s in mask_source], dtype=np.int8),
        "positive": np.array([all_sources[s].positive for s in mask_source], dtype=np.int32),
        "budget": mask_slot,
        "score_at": results["score_at"],
        **{name.replace("@", "_at_"): values for name, values in list_values.items()},
    }
    checks.update(
        {
            "single_positive_queries": singles,
            "single_positive_masks_identical_to_oracle": single_identity,
            "masks": int(mask_source.size),
            "documents_excluded_by_id": 0,
            **collision_checks,
        }
    )
    return Comparison(
        arrays,
        relation,
        {
            "offsets": results["offsets"],
            "rank": results["rank"],
            "score": results["score"],
            "ties": results["ties"],
            "collision_charged": charged,
        },
        columns,
        checks,
    )


def _aggregate(
    sources: Sequence[Source],
    mask_source: np.ndarray,
    mask_slot: np.ndarray,
    results: Mapping[str, np.ndarray],
    query_ids: Sequence[str],
    doc_ids: Sequence[str],
    judgments: Mapping[str, Mapping[str, int]],
    rows_of: Sequence[np.ndarray],
    grade_of: Sequence[np.ndarray],
    splits_of: Sequence[Sequence[tuple[np.ndarray, np.ndarray]]],
    vectors: Sequence[np.ndarray],
    documents: np.ndarray,
    budgets: Sequence[int],
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, np.ndarray]]:
    """Turn scan results into per-query measures and check the invariants."""
    arrays = _empty_arrays(len(query_ids), len(budgets))
    full_slot = len(budgets) - 1
    offsets = results["offsets"]
    rank, score = results["rank"], results["score"]
    list_values = {name: np.full(mask_source.size, np.nan) for name in LIST_METRICS}
    by_key: dict[tuple[int, int, int], list[int]] = {}
    for mask, source_at in enumerate(mask_source):
        source = sources[source_at]
        by_key.setdefault((source.query, source.arm, int(mask_slot[mask])), []).append(mask)
    bad_bounds = 0
    corpus = documents.shape[0]
    violations = 0
    full_mismatch = list_mismatch = 0
    jaccard_cache: dict[tuple[int, int], np.ndarray] = {}

    def mask_of(order: np.ndarray | None, k: int) -> np.ndarray:
        found = np.zeros(DIM, dtype=bool)
        found[np.arange(DIM) if order is None else order[:k]] = True
        return found

    for (query, arm, slot), masks in by_key.items():
        name = ARMS[arm]
        rows = rows_of[query]
        grades = grade_of[query]
        gains = relevance_gains(grades)
        ranks = [rank[offsets[m] : offsets[m + 1]] for m in masks]
        scores = [score[offsets[m] : offsets[m + 1]] for m in masks]
        for values in ranks:
            bad_bounds += int(((values < 1) | (values > corpus)).sum())
        if name != "own":
            ranked_metrics: list[dict[str, float]] = []
            for m in masks:
                top = results["top_rows"][m]
                ranking = [doc_ids[r] for r in top if r >= 0]
                values_ = query_metrics(ranking, judgments[query_ids[query]], CUTOFFS)
                for metric in LIST_METRICS:
                    list_values[metric][m] = values_[metric]
                ranked_metrics.append(values_)
            for metric in LIST_METRICS:
                arrays[metric][arm, slot, query] = float(
                    np.mean([v[metric] for v in ranked_metrics])
                )
            for at, rank_value in enumerate(SCORE_RANKS):
                reached = results["score_at"][masks, at]
                if np.isfinite(reached).any():  # a corpus can be shorter than this rank
                    arrays[f"at{rank_value}"][arm, slot, query] = float(np.nanmean(reached))
        if name in TRACKING_ALL:
            measures = _position_measures(ranks[0], scores[0])
            for key, value in measures.items():
                arrays[key][arm, slot, query] = value
        elif name == "own":
            ranks_all = np.concatenate(ranks)
            scores_all = np.concatenate(scores)
            for key, value in _position_measures(ranks_all, scores_all).items():
                arrays[key][arm, slot, query] = value
        if name == "held" or name in TRACKING_ALL:
            splits = splits_of[query]
            if splits:
                held: list[dict[str, float]] = []
                for partition, (_, left_out) in enumerate(splits):
                    at_mask = masks[partition] if name == "held" else masks[0]
                    here_ranks = rank[offsets[at_mask] : offsets[at_mask + 1]]
                    here_scores = score[offsets[at_mask] : offsets[at_mask + 1]]
                    held.append(_position_measures(here_ranks[left_out], here_scores[left_out]))
                for key in ("rr", "logrank", "recall10", "recall100", "score"):
                    arrays[f"held_{key}"][arm, slot, query] = float(np.mean([h[key] for h in held]))
    for (query, arm, slot), masks in by_key.items():
        name = ARMS[arm]
        if name not in ("prefix", "l2s", "held"):
            continue
        ceiling = by_key.get((query, _arm("oracle"), slot))
        if ceiling is None:
            continue
        k = int(budgets[slot])
        cached = jaccard_cache.setdefault(
            (query, slot), mask_of(sources[mask_source[ceiling[0]]].order, k)
        )
        overlaps = []
        for mask in masks:
            selected = mask_of(sources[mask_source[mask]].order, k)
            common = int((selected & cached).sum())
            overlaps.append(common / (2 * k - common))
        arrays["jaccard_oracle"][arm, slot, query] = float(np.mean(overlaps))
    for query, rows in enumerate(rows_of):
        gains = relevance_gains(grade_of[query])
        values = np.asarray(vectors[query], np.float64) * np.asarray(
            gather(documents, rows), np.float64
        )
        mass = float(np.abs(values).sum(axis=1).mean())
        for slot in range(len(budgets)):
            top = by_key[(query, _arm("oracle"), slot)][0]
            reference = score[offsets[top] : offsets[top + 1]]
            ceiling_sum = float((gains * reference).sum())
            for other in ("prefix", "l2s"):
                mask = by_key[(query, _arm(other), slot)][0]
                here = score[offsets[mask] : offsets[mask + 1]]
                total = float((gains * here).sum())
                violations += int(total > ceiling_sum)
                arrays["regret"][_arm(other), slot, query] = (
                    (ceiling_sum - total) / float(gains.sum()) / mass
                )
            arrays["regret"][_arm("oracle"), slot, query] = 0.0
        if rows.size == 1:
            for key in (*POSITIVE_MEASURES[:-1], *LIST_METRICS, *COMPETITOR_MEASURES):
                arrays[key][_arm("own")][:, query] = arrays[key][_arm("oracle")][:, query]
        full_mask = by_key[(query, _arm("full"), full_slot)][0]
        reference_rank = rank[offsets[full_mask] : offsets[full_mask + 1]]
        reference_list = results["top_rows"][full_mask]
        for other in ("prefix", "oracle", "l2s", "held"):
            for mask in by_key.get((query, _arm(other), full_slot), []):
                full_mismatch += int(
                    not np.array_equal(rank[offsets[mask] : offsets[mask + 1]], reference_rank)
                )
                list_mismatch += int(not np.array_equal(results["top_rows"][mask], reference_list))
        for mask in by_key.get((query, _arm("own"), full_slot), []):
            at = sources[mask_source[mask]].positive
            full_mismatch += int(rank[offsets[mask]] != reference_rank[at])
    checks = {
        "ranks_out_of_bounds": bad_bounds,
        "oracle_ceiling_violations": violations,
        "k1024_rank_mismatches": full_mismatch,
        "k1024_list_mismatches": list_mismatch,
        "tied_tracked_documents": int(results["ties"].sum()),
        "boundary_note": "L2S ties at a mask boundary go to the lower dimension",
    }
    return arrays, checks, list_values


def boundary_ties(importance: Tensor, budgets: Sequence[int]) -> int:
    """Queries and budgets whose top-``k`` cut falls between two equal importances."""
    values = importance.sort(dim=1, descending=True).values
    return sum(int((values[:, k - 1] == values[:, k]).sum().item()) for k in budgets if k < DIM)


def save(target: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write arrays compressed; the arrays' own types are kept."""
    np.savez_compressed(target, **arrays)  # type: ignore[arg-type]
