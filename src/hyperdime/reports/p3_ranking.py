"""Report of the mask ranking diagnostic, generated from one run; no number is typed here."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hyperdime.artifacts import latest_artifact
from hyperdime.contracts.manifests import read_manifest
from hyperdime.evaluation.representation_ranking import CELLS, ROWS
from hyperdime.evaluation.statistics import holm, paired_randomization_test
from hyperdime.reports.e0 import _markdown
from hyperdime.reports.p3_svg import Series, lines

RESAMPLES = 2000
PERMUTATIONS = 10_000
SEED = 0
COHORT_MINIMUM = {"2-2": 4, "5-5": 10, "10-10": 20, "20-20": 40}
SHOWN = ("prefix", "shared", "own", *CELLS)
RANKING_ROWS = ("prefix", "shared", *CELLS)
LIST_METRICS = (
    ("ndcg@10", "nDCG@10"),
    ("mrr@10", "MRR@10"),
    ("mrr@100", "MRR@100"),
    ("recall@100", "Recall@100"),
)
POSITION_MEASURES = (
    ("rr", "reciprocal rank"),
    ("recall10", "in top 10"),
    ("recall100", "in top 100"),
    ("logrank", "log rank (lower is better)"),
    ("worst", "worst rank in the set"),
)
HELD_MEASURES = (
    ("held_ndcg10", "nDCG@10, restricted to held-out labels"),
    ("secondary_rr", "reciprocal rank, construction positives skipped (secondary)"),
)
CLAIM_METRICS = ("ndcg@10", "mrr@10", "recall@100")
PRIMARY_FAMILY = "primary: held-out positives"
ALL_LABEL_FAMILY = "secondary: all-label, includes construction positives"
FAMILIES = (PRIMARY_FAMILY, ALL_LABEL_FAMILY)
PRIMARY_MEASURES = (
    ("rr", "held-out reciprocal rank"),
    ("logrank", "held-out log rank"),
    ("recall10", "held-out in top 10"),
    ("recall100", "held-out in top 100"),
)


def _num(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


@dataclass
class Environment:
    """Per-query arrays of one environment, with the resampling every interval shares."""

    name: str
    arrays: dict[str, np.ndarray]
    weights: np.ndarray
    resamples: np.ndarray
    budgets: list[int]
    summary: dict[str, Any]


def resampled_queries(strata: np.ndarray, count: int, seed: int) -> np.ndarray:
    """``[RESAMPLES, Q]`` query positions, resampled within each stratum of the selection. One
    matrix per environment: each interval uses the same resamples, so contrasts stay paired."""
    rng = np.random.default_rng(seed)
    columns = []
    for stratum in np.unique(strata):
        members = np.flatnonzero(strata == stratum)
        columns.append(members[rng.integers(0, members.size, (count, members.size))])
    found: np.ndarray = np.concatenate(columns, axis=1)
    return found


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted mean over finite values; NaN if there are none."""
    finite = np.isfinite(values) & (weights > 0)
    if not finite.any():
        return float("nan")
    return float((values[finite] * weights[finite]).sum() / weights[finite].sum())


def interval(delta: np.ndarray, weights: np.ndarray, resamples: np.ndarray) -> tuple[float, float]:
    """Percentile 95% interval of a weighted mean over queries; pointwise and exploratory."""
    if not np.isfinite(delta).any():
        return (float("nan"), float("nan"))
    ok = np.isfinite(delta)
    w = np.where(ok, weights, 0.0)
    d = np.nan_to_num(delta, nan=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = (w[resamples] * d[resamples]).sum(axis=1) / w[resamples].sum(axis=1)
    means = means[np.isfinite(means)]
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def contrast(e: Environment, values: np.ndarray, base: np.ndarray) -> dict[str, float]:
    """A row's mean, its baseline on the same queries, and the paired difference over queries."""
    delta = values - base
    keep = np.isfinite(delta)
    low, high = interval(delta, e.weights, e.resamples)
    return {
        "queries": float(keep.sum()),
        "mean": weighted_mean(np.where(keep, values, np.nan), e.weights),
        "base": weighted_mean(np.where(keep, base, np.nan), e.weights),
        "delta": weighted_mean(delta, e.weights),
        "low": low,
        "high": high,
    }


def _span(found: dict[str, float]) -> str:
    return f"{_num(found['low'])} to {_num(found['high'])}"


def _measure(e: Environment, name: str, row: str, slot: int) -> np.ndarray:
    found: np.ndarray = e.arrays[name][ROWS.index(row), slot]
    return found


def _mean(e: Environment, name: str, row: str, slot: int) -> float:
    return weighted_mean(_measure(e, name, row, slot), e.weights)


def _change(e: Environment, name: str, row: str, slot: int) -> float:
    """Weighted mean of a measure minus its value under the full query."""
    return weighted_mean(
        _measure(e, name, row, slot) - _measure(e, f"base_{name}", row, slot), e.weights
    )


def _list_table(environments: list[Environment]) -> list[list[str]]:
    table = [["environment", "arm", "k", "metric", "queries", "mean", "full query, same queries",
              "difference", "95% interval (pointwise)"]]  # fmt: skip
    for e in environments:
        last = len(e.budgets)
        for name, label in LIST_METRICS:
            base = _measure(e, name, "full", last)
            table.append([e.name, "full query", "1024", label, str(int(np.isfinite(base).sum())),
                          _num(weighted_mean(base, e.weights)), "", "", ""])  # fmt: skip
            for row in RANKING_ROWS:
                for slot, k in enumerate(e.budgets):
                    found = contrast(e, _measure(e, name, row, slot), base)
                    if not found["queries"]:
                        continue
                    table.append([e.name, row, str(k), label, str(int(found["queries"])),
                                  _num(found["mean"]), _num(found["base"]), _num(found["delta"]),
                                  f"{_num(found['low'])} to {_num(found['high'])}"])  # fmt: skip
    return table


def _position_table(
    environments: list[Environment], *, document_weighted: bool = False
) -> list[list[str]]:
    table = [["environment", "arm", "k", "positives evaluated", "measure", "queries", "arm value",
              "full query, same documents", "difference", "95% interval (pointwise)"]]  # fmt: skip
    measures = (*POSITION_MEASURES, *HELD_MEASURES)
    for e in environments:
        for row in SHOWN:
            for slot, k in enumerate(e.budgets):
                evaluated = _measure(e, "evaluated", row, slot)
                if not np.isfinite(evaluated).any():
                    continue
                weights = (
                    e.weights * np.nan_to_num(evaluated, nan=0.0)
                    if document_weighted
                    else e.weights
                )
                weighted = Environment(e.name, e.arrays, weights, e.resamples, e.budgets, e.summary)
                for name, label in measures:
                    here = _measure(e, name, row, slot)
                    if not np.isfinite(here).any():
                        continue
                    found = contrast(weighted, here, _measure(e, f"base_{name}", row, slot))
                    table.append([e.name, row, str(k), _set_label(row), label,
                                  str(int(found["queries"])), _num(found["mean"]),
                                  _num(found["base"]), _num(found["delta"]),
                                  f"{_num(found['low'])} to {_num(found['high'])}"])  # fmt: skip
    return table


def _set_label(row: str) -> str:
    if row == "own":
        return "the mask's own positive"
    return "held-out" if row in CELLS else "all known positives"


def _movement_table(environments: list[Environment]) -> list[list[str]]:
    table = [["environment", "arm", "k", "queries", "mean fraction of positives improved",
              "mean fraction worsened", "worst positive's rank change (median)",
              "queries whose worst positive worsened"]]  # fmt: skip
    for e in environments:
        for row in SHOWN:
            for slot, k in enumerate(e.budgets):
                up = _measure(e, "improved", row, slot)
                if not np.isfinite(up).any():
                    continue
                down = _measure(e, "worsened", row, slot)
                worst = _measure(e, "worst", row, slot)
                base = _measure(e, "base_worst", row, slot)
                keep = np.isfinite(worst - base)
                moved = worst[keep] - base[keep]
                table.append([e.name, row, str(k), str(int(np.isfinite(up).sum())),
                              _num(weighted_mean(up, e.weights)),
                              _num(weighted_mean(down, e.weights)),
                              _num(float(np.median(moved)), 1) if moved.size else "n/a",
                              _num(weighted_mean(
                                  (worst > base).astype(float) + np.where(keep, 0.0, np.nan),
                                  e.weights,
                              ))])  # fmt: skip
    return table


def _competition_table(environments: list[Environment]) -> list[list[str]]:
    table = [["environment", "arm", "k", "positive score change",
              "score change at rank 10 of the list", "score change at rank 100",
              "corpus mean score change", "positive-minus-rank-10 margin change",
              "positive z-score change", "mean rank change (log)"]]  # fmt: skip
    for e in environments:
        for row in SHOWN:
            for slot, k in enumerate(e.budgets):
                score = _measure(e, "score", row, slot)
                if not np.isfinite(score).any():
                    continue

                table.append([e.name, row, str(k), _num(_change(e, "score", row, slot)),
                              _num(_mean(e, "rank10_gain", row, slot)),
                              _num(_mean(e, "rank100_gain", row, slot)),
                              _num(_mean(e, "corpus_mean_shift", row, slot)),
                              _num(_change(e, "margin10", row, slot)),
                              _num(_change(e, "z", row, slot), 3),
                              _num(_change(e, "logrank", row, slot), 3)])  # fmt: skip
    return table


def _cohort_table(environments: list[Environment]) -> list[list[str]]:
    """Cells of different construction size on the same queries: only those with enough positives
    for the largest cell of the row."""
    table = [["environment", "queries with at least", "queries", "cell", "k", "measure",
              "cell value", "1-1 value, same queries", "difference from 1-1",
              "95% interval (pointwise)"]]  # fmt: skip
    measures = (("rr", "held-out reciprocal rank"), ("logrank", "held-out log rank"),
                ("ndcg@10", "nDCG@10, full judgments"))  # fmt: skip
    for e in environments:
        counts = e.arrays["counts"]
        for largest, minimum in COHORT_MINIMUM.items():
            cohort = counts >= minimum
            if not cohort.any():
                continue
            members = CELLS[: CELLS.index(largest) + 1]
            for cell in members:
                for slot, k in enumerate(e.budgets):
                    for name, label in measures:
                        here = _measure(e, name, cell, slot).copy()
                        base = _measure(e, name, "1-1", slot).copy()
                        here[~cohort] = np.nan
                        base[~cohort] = np.nan
                        found = contrast(e, here, base)
                        if not found["queries"]:
                            continue
                        table.append([e.name, str(minimum), str(int(cohort.sum())), cell, str(k),
                                      label, _num(found["mean"]), _num(found["base"]),
                                      _num(found["delta"]),
                                      _span(found)])  # fmt: skip
    return table


def claims(environments: list[Environment]) -> list[list[str]]:
    """Paired sign-flip tests of the two families frozen before the run (ADR-0016), each
    Holm-adjusted across every environment and budget. Nothing is adjusted across families."""
    entries: dict[str, dict[str, list[str]]] = {name: {} for name in FAMILIES}
    tests: dict[str, dict[str, float]] = {name: {} for name in FAMILIES}

    def add(family: str, e: Environment, slot: int, arm: str, label: str, value: str) -> None:
        values = _measure(e, value, arm, slot)
        if value in dict(LIST_METRICS):
            bases = _measure(e, value, "full", len(e.budgets))
        else:
            bases = _measure(e, f"base_{value}", arm, slot)
        delta = values - bases
        keep = np.isfinite(delta)
        if not keep.any():
            return
        weights = e.weights[keep] / e.weights[keep].mean()
        scaled = list(weights * delta[keep])
        key = f"{e.name}:{arm}:{e.budgets[slot]}:{label}"
        tests[family][key] = paired_randomization_test(
            scaled, [0.0] * len(scaled), PERMUTATIONS, seed=SEED
        )
        low, high = interval(delta, e.weights, e.resamples)
        entries[family][key] = [
            family, e.name, arm, str(e.budgets[slot]), label, str(int(keep.sum())),
            _num(weighted_mean(delta, e.weights)), f"{_num(low)} to {_num(high)}",
        ]  # fmt: skip

    for e in environments:
        for slot in range(len(e.budgets)):
            for value, label in PRIMARY_MEASURES:
                add(PRIMARY_FAMILY, e, slot, "1-1", label, value)
            for arm in ("shared", "1-1"):
                for metric in CLAIM_METRICS:
                    add(ALL_LABEL_FAMILY, e, slot, arm, metric, metric)
    table = [["family", "environment", "arm", "k", "measure", "queries",
              "difference from the full query", "95% interval (pointwise, exploratory)",
              "p (sign-flip)", "p (Holm, within family)"]]  # fmt: skip
    for family in FAMILIES:
        adjusted = holm(tests[family]) if tests[family] else {}
        for key, row in entries[family].items():
            table.append([*row, f"{tests[family][key]:.4f}", f"{adjusted[key]:.4f}"])
    return table


def _memorization_table(environments: list[Environment]) -> list[list[str]]:
    """All-label gains beside where they came from: the documents that built the mask, against the
    held-out positives of the same partitions."""
    table = [["environment", "cell", "k", "queries", "all-label nDCG@10 change",
              "held-out-label nDCG@10 change", "construction positive reciprocal rank change",
              "held-out positive reciprocal rank change", "construction positive in top 10 change",
              "held-out positive in top 10 change"]]  # fmt: skip
    for e in environments:
        last = len(e.budgets)
        for cell in CELLS:
            for slot, k in enumerate(e.budgets):
                held = _measure(e, "rr", cell, slot)
                if not np.isfinite(held).any():
                    continue
                table.append([e.name, cell, str(k), str(int(np.isfinite(held).sum())),
                              _num(weighted_mean(
                                  _measure(e, "ndcg@10", cell, slot)
                                  - _measure(e, "ndcg@10", "full", last), e.weights)),
                              _num(_change(e, "held_ndcg10", cell, slot)),
                              _num(_change(e, "construction_rr", cell, slot)),
                              _num(_change(e, "rr", cell, slot)),
                              _num(_change(e, "construction_recall10", cell, slot)),
                              _num(_change(e, "recall10", cell, slot))])  # fmt: skip
    return table


def _margin_table(run: Path, environments: list[Environment]) -> list[list[str]]:
    table = [["environment", "query position", "arm", "k", "rank of the worst evaluated positive",
              "its rank, full query", "its score, full query", "its score, mask",
              "competitors' mean score, full query", "competitors' mean score, mask",
              "competitors"]]  # fmt: skip
    for e in environments:
        with np.load(run / e.name / "margin-sample.npz") as found:
            rows = found["rows"]
        for row in rows:
            table.append([e.name, str(int(row[0])), ROWS[int(row[1])], str(int(row[2])),
                          str(int(row[3])), str(int(row[4])), _num(float(row[5]), 5),
                          _num(float(row[6]), 5), _num(float(row[7]), 5), _num(float(row[8]), 5),
                          str(int(row[9]))])  # fmt: skip
    return table


def _coverage_table(environments: list[Environment]) -> list[list[str]]:
    table = [["environment", "training queries", "with a positive", "eligible (2+ positives)",
              "ranked", "capped", "documents ranked", "positives", "orderings", "masks",
              "queries sharing an id with a document", "positives out of the competition",
              "ties met by tracked positives"]]  # fmt: skip
    for e in environments:
        c, checks = e.summary["coverage"], e.summary["checks"]
        table.append([e.name, str(c["training_queries"]), str(c["queries_with_positive"]),
                      str(c["eligible_multi_positive"]), str(c["queries"]), str(c["capped"]),
                      str(c["documents"]), str(c["positives"]), str(c["orderings"]),
                      str(c["masks"]),
                      str(c["queries_with_excluded_row"]), str(checks["excluded_positives"]),
                      str(checks["tied_tracked_documents"])])  # fmt: skip
    return table


def _cost_table(environments: list[Environment]) -> list[list[str]]:
    table = [["environment", "device", "seconds", "masks", "documents", "mask-documents per second",
              "plan", "thresholds", "read", "scan", "metrics", "write", "peak device MiB",
              "peak RSS MiB", "artifact MiB"]]  # fmt: skip
    for e in environments:
        c = e.summary["cost"]
        coverage = e.summary["coverage"]
        phase = c["phase_seconds"]
        masks, documents = coverage["masks"], coverage["documents"]
        table.append([e.name, c["device_memory"]["name"], _num(c["total_seconds"], 1), str(masks),
                      str(documents),
                      f"{masks * documents / max(phase.get('scan', 0.0), 1e-9) / 1e6:.0f}M",
                      *[_num(phase.get(p, 0.0), 1) for p in
                        ("plan", "thresholds", "read", "scan", "metrics", "write")],
                      _num(c["device_memory"].get("peak_allocated_mib"), 0),
                      _num(c["peak_rss_mib"], 0),
                      _num(c["artifact_bytes"] / 2**20, 1)])  # fmt: skip
    return table


def _check_table(environments: list[Environment]) -> list[list[str]]:
    table = [["environment", "stored regrets compared", "largest difference from the shared run",
              "source"]]  # fmt: skip
    for e in environments:
        checks = e.summary["checks"]
        table.append([e.name, str(checks["stored_regrets_compared"]),
                      _num(checks["largest_stored_regret_difference"], 8),
                      e.summary.get("shared_mask_run", "")])  # fmt: skip
    return table


def _figures(environments: list[Environment]) -> dict[str, str]:
    found: dict[str, str] = {}
    for e in environments:
        k = [float(x) for x in e.budgets]
        last = len(e.budgets)
        base = weighted_mean(_measure(e, "ndcg@10", "full", last), e.weights)
        series = [
            Series("Full query", k, [base] * len(k), neutral=True),
            *[
                Series(row, k, [
                    weighted_mean(_measure(e, "ndcg@10", row, slot), e.weights)
                    for slot in range(len(k))
                ])
                for row in ("prefix", "shared", "1-1")
            ],
        ]  # fmt: skip
        found[f"ndcg-{e.name}"] = lines(
            f"nDCG@10 under masks ({e.name})",
            "Training judgments, multi-positive queries, query-balanced. The shared and 1-1 masks "
            "saw relevant documents, and are privileged. Budgets are the grid, not every k.",
            "Retained dimensions k",
            "nDCG@10",
            series,
            markers=True,
        )
    return found


def _csv(table: list[list[str]]) -> str:
    out = io.StringIO()
    csv.writer(out, lineterminator="\n").writerows(table)
    return out.getvalue()


def _load(run: Path, env: str, seed: int) -> Environment:
    summary = json.loads((run / env / "summary.json").read_text(encoding="utf-8"))
    with np.load(run / env / "per-query.npz") as found:
        arrays = {name: found[name].astype(np.float64) for name in found.files}
    with np.load(run / env / "selection.npz") as found:
        strata, weights = found["strata"], found["weights"]
    return Environment(
        env,
        arrays,
        weights,
        resampled_queries(strata, RESAMPLES, seed),
        summary["coverage"]["budgets"],
        summary,
    )


def write(root: Path, target: Path) -> Path:
    """Write README, tables, and figures for the latest run; the same run gives the same bytes."""
    run = latest_artifact(root, "evaluation/P3-mask-ranking", "all")
    inputs = json.loads((run / "inputs.json").read_text(encoding="utf-8"))
    environments = [
        _load(run, env, SEED + offset) for offset, env in enumerate(inputs["environments"])
    ]
    for e in environments:
        e.summary["shared_mask_run"] = inputs[e.name].get("shared_mask_run") or ""
    manifest = read_manifest(run)
    tables = {
        "list-metrics": _list_table(environments),
        "positions": _position_table(environments),
        "positions-document-weighted": _position_table(environments, document_weighted=True),
        "movement": _movement_table(environments),
        "competition": _competition_table(environments),
        "common-cohort": _cohort_table(environments),
        "claims": claims(environments),
        "memorization": _memorization_table(environments),
        "margin-sample": _margin_table(run, environments),
        "coverage": _coverage_table(environments),
        "checks": _check_table(environments),
        "cost": _cost_table(environments),
    }
    figures = _figures(environments)
    (target / "tables").mkdir(parents=True, exist_ok=True)
    (target / "figures").mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        (target / "tables" / f"{name}.csv").write_text(_csv(table), encoding="utf-8")
    for name, svg in figures.items():
        (target / "figures" / f"{name}.svg").write_text(svg, encoding="utf-8")
    lines_out = [
        "# P3: rankings under masks that maximize positive similarity",
        "",
        "Privileged offline diagnostic, training judgments only, in-domain. Nothing here is a "
        "retrieval method, a ranking upper bound, or evidence about unseen queries.",
        "",
        f"- Run `{run.name}` (commit `{manifest.git_commit[:12]}`, dirty `{manifest.git_dirty}`, "
        f"device `{manifest.device}`)",
        f"- Budgets {inputs['budgets']}, a grid: representation curves cover every k, these "
        "rankings do not. The full query is k = 1024.",
        "- Exact float64 scoring of every corpus document for every mask; ties go to the lower "
        "row; ids equal to the query id are excluded, as in `rank_documents`.",
        "",
        "## Reading these tables",
        "",
        "The shared mask saw the positives it is scored on. Construction-only masks see only their "
        "construction positives, and are scored on held-out positives of the same training query. "
        "Each positive's own mask maximizes that positive's similarity: a privileged diagnostic, "
        "and never an upper bound on its rank. Construction positives remain relevant and remain "
        "in every primary ranking. Judgments are incomplete, so competitors are other documents, "
        "not confirmed irrelevant ones. Masks maximize mean similarity; they are neither minimax "
        "nor ranking optimal.",
        "",
        "Every arm here, the full query and prefix-k included, breaks exact ties the same way: the "
        "lower corpus row wins, and scores are exact. The repository's float32 `exact_search` "
        "leaves such ties to `topk`, so its full-query numbers can differ on queries with "
        "duplicate documents. Earlier results stand as published and are not recomputed here.",
        "",
        "Intervals are pointwise bootstrap intervals over queries and are exploratory. Only the "
        "two families below carry a multiplicity adjustment, each within itself.",
        "",
        "## Pre-declared families",
        "",
        "Primary: does a construction-only mask move held-out positives? Secondary: all-label "
        "metrics, in which a mask can gain by lifting the very documents that built it. A gain "
        "in the secondary family without a gain in the primary one is memorization of "
        "construction documents, not better retrieval of relevant documents.",
        "",
        *_markdown(tables["claims"]),
        "",
        "## All-label gains against held-out positives",
        "",
        "Same partitions, query-balanced. The construction columns show how far the documents "
        "that built the mask moved; the held-out columns show what happened to the others.",
        "",
        *_markdown(tables["memorization"]),
        "",
        "## Ranking metrics, full training judgments",
        "",
        "These count construction positives as relevant. Read them with the table above.",
        "",
        *_markdown(tables["list-metrics"]),
        "",
        "## Where relevant documents went",
        "",
        "Primary ranks, query-balanced. The held-out nDCG is a restricted-label diagnostic, and "
        "the secondary rank skips construction positives only to separate crowding from "
        "displacement.",
        "",
        *_markdown(tables["positions"]),
        "",
        "Document-weighted: every evaluated positive counts once. Larger queries dominate.",
        "",
        *_markdown(tables["positions-document-weighted"]),
        "",
        "## Gains and losses per query",
        "",
        *_markdown(tables["movement"]),
        "",
        "## Similarity against competition",
        "",
        "Differences from the full query, on the same documents. A positive whose score rises but "
        "whose rank does not, means other documents rose as much or more.",
        "",
        *_markdown(tables["competition"]),
        "",
        "## Construction size on common cohorts",
        "",
        "Cells are compared on the same queries: those with enough positives for the largest cell "
        "shown. Partitions are drawn separately per cell; this describes high-count queries only.",
        "",
        *_markdown(tables["common-cohort"]),
        "",
        "## Small sample of competing scores",
        "",
        "A seeded sample of queries, fixed before results: the worst-ranked evaluated positive "
        "against the mask's top non-positive documents, scored under both queries. Diagnostic "
        "only.",
        "",
        *_markdown(tables["margin-sample"]),
        "",
        "## Coverage, checks, and cost",
        "",
        *_markdown(tables["coverage"]),
        "",
        *_markdown(tables["checks"]),
        "",
        *_markdown(tables["cost"]),
        "",
    ]
    readme = target / "README.md"
    readme.write_text("\n".join(lines_out), encoding="utf-8")
    return readme
