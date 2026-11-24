"""What the id-equals-query exclusion did to a stored P3-mask-ranking run (ADR-0016).

    python scripts/checks/p3_ranking_exclusion_sensitivity.py <artifacts> <run_id> --output <dir>

Where a query's id equals a document id, the run removed that document from the competition. On
corpora with numeric ids this removes an unrelated passage. Given the stored rank and exact score of
each tracked positive, the passage's own score under each mask says exactly whether it would have
beaten that positive, so every rank-based measure can be corrected with no rescan. Top lists are not
stored, so list metrics are not corrected: only how often the passage would have entered them.

Reads the run and writes elsewhere; CPU only. Counts and ids of checks only.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from p3_ranking_integrity import _environment, _same

from hyperdime.contracts.hashing import NO_DATASET, file_sha256
from hyperdime.contracts.manifests import ArtifactWriter, start_manifest
from hyperdime.evaluation.representation_engine import gather
from hyperdime.evaluation.representation_ranking import (
    DIM,
    FULL_BUDGET,
    LIST_METRICS,
    MEASURES,
    ROWS,
    expand,
    masked_vector,
    per_query_summary,
    plan_query,
)
from hyperdime.reports.p3_ranking import (
    PRIMARY_FAMILY,
    PRIMARY_MEASURES,
    Environment,
    _load,
    claims,
    contrast,
)

TOP = (10, 100)
NOT_CORRECTED = (*LIST_METRICS, "score", "margin10", "z", "rank10_gain", "rank100_gain")
CORRECTED = tuple(
    name
    for name in MEASURES
    if name.removeprefix("base_") not in NOT_CORRECTED
    and name not in ("corpus_mean_shift", "evaluated")
)


def corrected_ranks(
    rank: np.ndarray,
    score: np.ndarray,
    row: np.ndarray,
    excluded_score: np.ndarray,
    excluded_row: np.ndarray,
) -> np.ndarray:
    """Ranks had the excluded document competed: one worse where it beats the positive, by a higher
    score or, on an exact tie, a lower corpus row. Documents out of the competition keep rank 0."""
    beats = (excluded_score > score) | ((excluded_score == score) & (excluded_row < row))
    found: np.ndarray = np.where(rank > 0, rank + beats.astype(np.int64), rank)
    return found


def _plan_order_matches(plan: Any, m: Mapping[str, np.ndarray]) -> bool:
    source = plan.source
    return bool(
        np.array_equal(m["budget"], plan.budget)
        and np.array_equal(m["query"], [plan.sources[s].query for s in source])
        and np.array_equal(m["row"], [plan.sources[s].row for s in source])
        and np.array_equal(m["partition"], [plan.sources[s].partition for s in source])
        and np.array_equal(m["positive"], [plan.sources[s].positive for s in source])
    )


def _evaluated(source: Any, width: int) -> np.ndarray:
    """Positions the run evaluated for this mask, within its positives."""
    if source.row == ROWS.index("own"):
        return np.array([source.positive])
    if source.held is not None:
        held: np.ndarray = source.held
        return held
    return np.arange(width)


def sensitivity(
    root: Path, run: Path, env: str, seed: int, budgets: Sequence[int]
) -> dict[str, Any]:
    """Corrected per-query rank measures of one environment, with how many ranks moved."""
    e = _environment(root, env)
    inputs = e.inputs
    ids = json.loads((run / env / "selected-queries.json").read_text(encoding="utf-8"))
    doc_rows = {doc: row for row, doc in enumerate(inputs.doc_ids)}
    excluded = np.array([doc_rows.get(qid, -1) for qid in ids], dtype=np.int64)
    counts = np.array([e.multi[qid] for qid in ids], dtype=np.int64)
    passage_is_positive = int(
        sum(row >= 0 and row in e.positive[qid] for qid, row in zip(ids, excluded, strict=True))
    )
    summary: dict[str, Any] = {
        "queries": len(ids),
        "positives": int(counts.sum()),
        "queries_with_excluded_document": int((excluded >= 0).sum()),
        "excluded_document_is_a_training_positive": passage_is_positive,
    }
    if not (excluded >= 0).any():
        return {"summary": summary, "corrected": None, "exposure": [], "replays_stored": True}
    with np.load(run / env / "masks.npz") as found:
        m = {name: found[name] for name in found.files}
    with np.load(run / env / "tracked.npz") as found:
        t = {name: found[name] for name in found.files}
    with np.load(run / env / "per-query.npz") as found:
        stored = {name: found[name] for name in found.files}
    grades = [e.grades[qid] for qid in ids]
    sources = []
    vectors = [
        np.asarray(inputs.query_vectors[inputs.query_rows[q]], dtype=np.float16) for q in ids
    ]
    for at, qid in enumerate(ids):
        sources.extend(plan_query(at, qid, vectors[at], gather(inputs.documents, e.positive[qid]),
                                  grades[at], seed=seed, device="cpu"))  # fmt: skip
    plan = expand(sources, budgets, counts)
    if not _plan_order_matches(plan, m):
        raise RuntimeError(f"{env}: rebuilt masks are not in the stored order")
    passage = np.zeros((len(ids), DIM), dtype=np.float64)
    affected = np.flatnonzero(excluded >= 0)
    passage[affected] = np.asarray(gather(inputs.documents, excluded[affected]), dtype=np.float64)
    offsets, position, rank, score = t["offsets"], t["position"], t["rank"], t["score"]
    mask_scores = np.full(plan.source.size, -np.inf)
    rows = np.zeros(rank.size, dtype=np.int64)
    for mask, source_at in enumerate(plan.source):
        s = plan.sources[source_at]
        slot = int(plan.budget[mask])
        k = DIM if slot == FULL_BUDGET else int(budgets[slot])
        low, high = offsets[mask], offsets[mask + 1]
        rows[low:high] = e.positive[ids[s.query]][position[low:high]]
        if excluded[s.query] >= 0:
            mask_scores[mask] = float(
                np.asarray(masked_vector(vectors[s.query], s.order, k), dtype=np.float64)
                @ passage[s.query]
            )
    mask_of = np.repeat(np.arange(plan.source.size), np.diff(offsets))
    corrected = corrected_ranks(
        rank, score, rows, mask_scores[mask_of], excluded[m["query"][mask_of]]
    )
    base: dict[str, np.ndarray] = {
        "position": position,
        "score": score,
        "corpus_mean": m["corpus_mean"],
        "corpus_var": m["corpus_var"],
        "score_at": m["score_at"],
        **{name: m[name.replace("@", "_at_")] for name in LIST_METRICS},
    }
    replay = per_query_summary(plan, {**base, "rank": rank}, offsets, len(ids), budgets, grades)
    same = all(_same(replay[name].astype(np.float32), stored[name]) for name in MEASURES)
    fixed = per_query_summary(plan, {**base, "rank": corrected}, offsets, len(ids), budgets, grades)
    exposure, full_changed = _exposure(plan, m, offsets, rank, corrected, mask_scores, budgets)
    summary["positives_with_a_changed_rank_under_the_full_query"] = full_changed
    return {
        "summary": summary,
        "corrected": {name: fixed[name].astype(np.float32) for name in CORRECTED},
        "exposure": exposure,
        "replays_stored": same,
    }


def _exposure(
    plan: Any,
    m: Mapping[str, np.ndarray],
    offsets: np.ndarray,
    rank: np.ndarray,
    corrected: np.ndarray,
    mask_scores: np.ndarray,
    budgets: Sequence[int],
) -> tuple[list[dict[str, Any]], int]:
    """Per arm and budget: evaluated positives whose rank moved or crossed a cutoff, and masks whose
    stored top list the passage would have entered. ``>=`` the stored score bounds the entry."""
    groups: dict[tuple[int, int], dict[str, Any]] = {}
    full_changed: set[tuple[int, int]] = set()
    for mask, source_at in enumerate(plan.source):
        s = plan.sources[source_at]
        slot = int(plan.budget[mask])
        group = groups.setdefault(
            (s.row, slot),
            {"masks": 0, "evaluated": 0, "moved": 0, "cross10": 0, "cross100": 0, "queries": set(),
             "all": set(), "into10": 0, "into100": 0},
        )  # fmt: skip
        group["masks"] += 1
        group["all"].add(s.query)
        low, high = int(offsets[mask]), int(offsets[mask + 1])
        evaluate = _evaluated(s, high - low)
        at = np.array([0]) if s.row == ROWS.index("own") else evaluate
        was, now = rank[low:high][at], corrected[low:high][at]
        keep = was > 0
        was, now = was[keep], now[keep]
        group["evaluated"] += int(was.size)
        moved = now != was
        group["moved"] += int(moved.sum())
        group["cross10"] += int(((was <= 10) & (now > 10)).sum())
        group["cross100"] += int(((was <= 100) & (now > 100)).sum())
        if moved.any():
            group["queries"].add(s.query)
        if s.row == ROWS.index("full"):
            full_changed.update((s.query, int(i)) for i in at[keep][moved])
        if s.row != ROWS.index("own"):
            group["into10"] += int(mask_scores[mask] >= m["score_at"][mask, 1])
            group["into100"] += int(mask_scores[mask] >= m["score_at"][mask, 2])
    rows = []
    for (row, slot), group in sorted(groups.items()):
        rows.append({
            "arm": ROWS[row],
            "k": 1024 if slot == FULL_BUDGET else int(budgets[slot]),
            "masks": group["masks"],
            "queries": len(group["all"]),
            "evaluated_positives": group["evaluated"],
            "ranks_moved": group["moved"],
            "queries_with_a_moved_rank": len(group["queries"]),
            "left_top_10": group["cross10"],
            "left_top_100": group["cross100"],
            "masks_passage_at_or_above_rank_10": group["into10"],
            "masks_passage_at_or_above_rank_100": group["into100"],
        })  # fmt: skip
    return rows, len(full_changed)


def _p_values(
    environments: Sequence[Environment],
) -> dict[tuple[str, str, str], tuple[float, float]]:
    table = claims(list(environments))
    header = table[0]
    place = {name: header.index(name) for name in header}
    found = {}
    for row in table[1:]:
        if row[place["family"]] != PRIMARY_FAMILY:
            continue
        key = (row[place["environment"]], row[place["k"]], row[place["measure"]])
        found[key] = (
            float(row[place["p (sign-flip)"]]),
            float(row[place["p (Holm, within family)"]]),
        )
    return found


def primary_comparison(
    run: Path, order: Sequence[str], fixed: Mapping[str, Mapping[str, np.ndarray]]
) -> list[dict[str, Any]]:
    """The primary family as stored and as corrected, on the same resamples and permutations."""
    stored = [_load(run, env, offset) for offset, env in enumerate(order)]
    corrected = []
    for e in stored:
        arrays = dict(e.arrays)
        for name, values in (fixed.get(e.name) or {}).items():
            arrays[name] = values.astype(np.float64)
        corrected.append(Environment(e.name, arrays, e.weights, e.resamples, e.budgets, e.summary))
    p_stored, p_fixed = _p_values(stored), _p_values(corrected)
    cell = ROWS.index("1-1")
    rows = []
    for before, after in zip(stored, corrected, strict=True):
        for slot, k in enumerate(before.budgets):
            for value, label in PRIMARY_MEASURES:
                old = contrast(before, before.arrays[value][cell, slot],
                               before.arrays[f"base_{value}"][cell, slot])  # fmt: skip
                new = contrast(after, after.arrays[value][cell, slot],
                               after.arrays[f"base_{value}"][cell, slot])  # fmt: skip
                key = (before.name, str(k), label)
                pa, ha = p_stored[key]
                pb, hb = p_fixed[key]
                rows.append({
                    "environment": before.name, "k": k, "measure": label,
                    "queries": int(old["queries"]),
                    "stored_difference": old["delta"], "corrected_difference": new["delta"],
                    "change_in_difference": new["delta"] - old["delta"],
                    "stored_low": old["low"], "stored_high": old["high"],
                    "corrected_low": new["low"], "corrected_high": new["high"],
                    "stored_p": pa, "corrected_p": pb,
                    "stored_p_holm": ha, "corrected_p_holm": hb,
                    "sign_changed": bool(np.sign(old["delta"]) != np.sign(new["delta"])),
                    "interval_excludes_zero_changed": bool(
                        (old["low"] > 0 or old["high"] < 0) != (new["low"] > 0 or new["high"] < 0)
                    ),
                    "holm_0_05_changed": bool((ha < 0.05) != (hb < 0.05)),
                })  # fmt: skip
    return rows


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def check_run(root: Path, run: Path) -> dict[str, Any]:
    """Sensitivity of every environment of ``run``; ``root`` holds its inputs."""
    sources = json.loads((run / "inputs.json").read_text(encoding="utf-8"))
    seed, budgets = int(sources["seed"]), [int(k) for k in sources["budgets"]]
    environments, fixed, exposure = {}, {}, []
    for env in sources["environments"]:
        found = sensitivity(root, run, env, seed, budgets)
        environments[env] = {**found["summary"], "replays_stored": found["replays_stored"]}
        if found["corrected"] is not None:
            fixed[env] = found["corrected"]
        exposure.extend({"environment": env, **row} for row in found["exposure"])
    comparison = primary_comparison(run, sources["environments"], fixed)
    return {
        "run": run.name,
        "environments": environments,
        "primary": comparison,
        "exposure": exposure,
        "corrected_measures": list(CORRECTED),
        "not_corrected": list(NOT_CORRECTED),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("run")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    script_hash = file_sha256(Path(__file__))
    manifest = start_manifest(
        "p3-ranking-exclusion-sensitivity",
        {"run": args.run, "script_sha256": script_hash},
        seed=0,
        dataset_hash=NO_DATASET,
    )
    found = check_run(args.root, args.root / "evaluation/P3-mask-ranking/all" / args.run)
    found["script_sha256"] = script_hash
    with ArtifactWriter(
        args.output, "evaluation/P3-mask-ranking-exclusion/all", manifest
    ) as writer:
        writer.path("sensitivity.json").write_text(
            json.dumps(found, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        writer.path("primary-comparison.csv").write_text(
            _csv_text(found["primary"]), encoding="utf-8"
        )
        if found["exposure"]:
            writer.path("exposure.csv").write_text(_csv_text(found["exposure"]), encoding="utf-8")
        print(writer.commit())
    replayed = all(env["replays_stored"] for env in found["environments"].values())
    print(json.dumps({"replays_stored": replayed, "environments": found["environments"]}))
    return 0 if replayed else 1


if __name__ == "__main__":
    sys.exit(main())
