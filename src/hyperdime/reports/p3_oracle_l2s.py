"""P3-oracle-l2s report: exact positive masks beside L2S on the same training queries (ADR-0017)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hyperdime.artifacts import latest_artifact
from hyperdime.contracts.manifests import read_manifest
from hyperdime.evaluation.oracle_comparison import ARMS, RELATION_MEASURES
from hyperdime.reports.e0 import _markdown
from hyperdime.reports.p3_ranking import (
    RESAMPLES,
    SEED,
    _csv,
    _num,
    interval,
    resampled_queries,
    weighted_mean,
)

COHORTS = (("single-positive", 1), ("multi-positive", 2))
LIST_METRICS = ("ndcg@10", "mrr@10", "mrr@100", "recall@100")
POSITION = ("rr", "logrank", "recall10", "recall100")
HELD = ("held_rr", "held_logrank", "held_recall10", "held_recall100")
COMPARISONS = (("oracle", "l2s"), ("l2s", "full"), ("oracle", "full"), ("prefix", "full"))
HELD_COMPARISONS = (("held", "l2s"), ("l2s", "full"), ("held", "full"), ("oracle", "held"))
LABELS = {
    "full": "full query",
    "prefix": "prefix-k",
    "oracle": "all-positive oracle (privileged)",
    "held": "held-out transfer (1-1)",
    "l2s": "L2S (in-sample)",
    "own": "pair-optimal, target only (secondary)",
}
TINY = 1e-12


@dataclass
class Cohort:
    environment: str
    name: str
    members: np.ndarray
    weights: np.ndarray
    resamples: np.ndarray


@dataclass
class Environment:
    name: str
    arrays: dict[str, np.ndarray]
    relation: dict[str, np.ndarray]
    budgets: list[int]
    cohorts: list[Cohort]
    summary: dict[str, Any]


def _at(arrays: Mapping[str, np.ndarray], name: str, arm: str, slot: int) -> np.ndarray:
    found: np.ndarray = arrays[name][ARMS.index(arm), slot]
    return found


def _values(e: Environment, name: str, arm: str, slot: int) -> np.ndarray:
    """A measure of one arm at a budget; the full query is the same at every budget."""
    return _at(e.arrays, name, arm, len(e.budgets) - 1 if arm == "full" else slot)


def _load(run: Path, env: str, seed: int, budgets: list[int]) -> Environment:
    directory = run / env
    with np.load(directory / "per-query.npz") as data:
        arrays = {name: data[name] for name in data.files}
    with np.load(directory / "relationship.npz") as data:
        relation = {name: data[name] for name in data.files}
    with np.load(directory / "selection.npz") as data:
        strata, weights = data["strata"], data["weights"]
        sampled = data["sampled"] if "sampled" in data.files else np.zeros_like(strata)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    cohorts = []
    for name, stratum in COHORTS:
        members = np.flatnonzero(strata == stratum)
        count = members.size
        cohorts.append(
            Cohort(
                env,
                name,
                members,
                weights[members],
                resampled_queries(sampled[members].astype(np.int64), RESAMPLES, seed + stratum)
                if count
                else np.empty((RESAMPLES, 0), dtype=np.int64),
            )
        )
    return Environment(env, arrays, relation, budgets, cohorts, summary)


def _summary(values: np.ndarray, cohort: Cohort) -> str:
    if not cohort.members.size or not np.isfinite(values).any():
        return "n/a"
    low, high = interval(values, cohort.weights, cohort.resamples)
    return f"{weighted_mean(values, cohort.weights):.4f} ({low:.4f} to {high:.4f})"


def _delta_row(a: np.ndarray, b: np.ndarray, cohort: Cohort) -> tuple[str, str, str, str, str, str]:
    delta = a - b
    keep = np.isfinite(delta)
    if not keep.any():
        return ("0", "n/a", "n/a", "n/a", "n/a", "n/a")
    low, high = interval(np.where(keep, delta, np.nan), cohort.weights, cohort.resamples)
    wins = np.sum(cohort.weights[keep & (delta > TINY)]) / np.sum(cohort.weights[keep])
    losses = np.sum(cohort.weights[keep & (delta < -TINY)]) / np.sum(cohort.weights[keep])
    return (
        str(int(keep.sum())),
        _num(weighted_mean(delta, cohort.weights)),
        _num(low),
        _num(high),
        _num(float(wins), 3),
        _num(float(losses), 3),
    )


HEADER = ("queries", "difference", "2.5%", "97.5%", "share better", "share worse")


def _metrics_table(environments: list[Environment]) -> list[list[str]]:
    table = [["environment", "cohort", "k", "ratio", "arm", *LIST_METRICS]]
    for e in environments:
        for cohort in e.cohorts:
            for slot, k in enumerate(e.budgets):
                for arm in ("full", "prefix", "oracle", "held", "l2s"):
                    if arm == "full" and slot != len(e.budgets) - 1:
                        continue
                    cells = [
                        _summary(_values(e, m, arm, slot)[cohort.members], cohort)
                        for m in LIST_METRICS
                    ]
                    if all(c == "n/a" for c in cells):
                        continue
                    table.append(
                        [e.name, cohort.name, str(k), f"{k / 1024:.0%}", LABELS[arm], *cells]
                    )
    return table


def _paired(
    environments: list[Environment],
    measures: tuple[str, ...],
    comparisons: tuple[tuple[str, str], ...],
    multi_only: bool = False,
) -> list[list[str]]:
    table = [["environment", "cohort", "k", "measure", "comparison", *HEADER]]
    for e in environments:
        for cohort in e.cohorts:
            if multi_only and cohort.name != "multi-positive":
                continue
            for slot, k in enumerate(e.budgets[:-1]):
                for measure in measures:
                    for a, b in comparisons:
                        here = _values(e, measure, a, slot)[cohort.members]
                        base = _values(e, measure, b, slot)[cohort.members]
                        table.append(
                            [
                                e.name, cohort.name, str(k), measure,
                                f"{LABELS[a]} minus {LABELS[b]}",
                                *_delta_row(here, base, cohort),
                            ]
                        )  # fmt: skip
    return table


def _positions(environments: list[Environment]) -> list[list[str]]:
    table = [
        ["environment", "cohort", "k", "arm", *POSITION, "positive score change",
         "at rank 1 change", "at rank 10 change", "at rank 100 change"]
    ]  # fmt: skip
    for e in environments:
        for cohort in e.cohorts:
            for slot, k in enumerate(e.budgets[:-1]):
                for arm in ("prefix", "oracle", "l2s", "own"):
                    if not np.isfinite(_values(e, "rr", arm, slot)[cohort.members]).any():
                        continue
                    row = [
                        _summary(_values(e, name, arm, slot)[cohort.members], cohort)
                        for name in POSITION
                    ]
                    deltas = []
                    for name in ("score", "at1", "at10", "at100"):
                        here = _values(e, name, arm, slot)[cohort.members]
                        base = _values(e, name, "full", slot)[cohort.members]
                        deltas.append(_summary(here - base, cohort))
                    table.append([e.name, cohort.name, str(k), LABELS[arm], *row, *deltas])
    return table


def _agreement(environments: list[Environment]) -> list[list[str]]:
    table = [["environment", "cohort", "k", "arm", "Jaccard with oracle mask",
              "score regret to ceiling / mass"]]  # fmt: skip
    for e in environments:
        for cohort in e.cohorts:
            for slot, k in enumerate(e.budgets):
                for arm in ("prefix", "l2s", "held"):
                    jaccard = _values(e, "jaccard_oracle", arm, slot)[cohort.members]
                    if not np.isfinite(jaccard).any():
                        continue
                    regret = _values(e, "regret", arm, slot)[cohort.members]
                    table.append([
                        e.name, cohort.name, str(k), LABELS[arm], _summary(jaccard, cohort),
                        _summary(regret, cohort),
                    ])  # fmt: skip
    return table


def _relationship(environments: list[Environment]) -> list[list[str]]:
    table = [["environment", "k", "queries", *RELATION_MEASURES]]
    for e in environments:
        cohort = next(c for c in e.cohorts if c.name == "multi-positive")
        if not cohort.members.size:
            continue
        for slot, k in enumerate(e.budgets):
            table.append(
                [
                    e.name, str(k), str(cohort.members.size),
                    *(_summary(e.relation[name][slot, cohort.members], cohort)
                      for name in RELATION_MEASURES),
                ]
            )  # fmt: skip
    return table


def _coverage(environments: list[Environment]) -> list[list[str]]:
    table = [["environment", "training queries", "with a positive", "evaluated", "single-positive",
              "multi-positive", "positives", "documents", "capped", "weights all 1"]]  # fmt: skip
    for e in environments:
        c = e.summary["coverage"]
        table.append([
            e.name, str(c["training_queries"]), str(c["queries_with_positive"]), str(c["queries"]),
            str(c["single_positive"]), str(c["multi_positive"]), str(c["positives"]),
            str(c["documents"]), str(c["capped"]), str(c["weights_all_one"]),
        ])  # fmt: skip
    return table


def _provenance(environments: list[Environment]) -> list[list[str]]:
    table = [["environment", "check", "passed"]]
    for e in environments:
        for name, passed in sorted(e.summary["provenance"]["checks"].items()):
            table.append([e.name, name, str(passed)])
    return table


def _checks(environments: list[Environment]) -> list[list[str]]:
    table = [["environment", "check", "value"]]
    for e in environments:
        for name, value in sorted(e.summary["checks"].items()):
            table.append([e.name, name, json.dumps(value, sort_keys=True)])
        table.append([e.name, "total seconds", _num(e.summary["cost"]["total_seconds"], 1)])
        table.append([e.name, "device", e.summary["cost"]["device"]])
    return table


def write(root: Path, target: Path) -> Path:
    """Write README and tables for the latest run; the same run gives the same bytes."""
    run = latest_artifact(root, "evaluation/P3-oracle-l2s", "all")
    inputs = json.loads((run / "inputs.json").read_text(encoding="utf-8"))
    environments = [
        _load(run, env, SEED + offset, inputs["budgets"])
        for offset, env in enumerate(inputs["environments"])
    ]
    manifest = read_manifest(run)
    tables = {
        "provenance": _provenance(environments),
        "coverage": _coverage(environments),
        "metrics": _metrics_table(environments),
        "paired": _paired(environments, LIST_METRICS, COMPARISONS),
        "paired-held": _paired(environments, HELD, HELD_COMPARISONS, multi_only=True),
        "positions": _positions(environments),
        "agreement": _agreement(environments),
        "relationship": _relationship(environments),
        "checks": _checks(environments),
    }
    (target / "tables").mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        (target / "tables" / f"{name}.csv").write_text(_csv(table), encoding="utf-8")
    capped = inputs["max_queries"] is not None
    lines = [
        "# P3: exact positive masks beside Learning-to-Select",
        "",
        "Privileged, paired, in-domain diagnostic on training queries and training judgments. "
        "L2S numbers are **in-sample**: L2S was fitted on these queries. Nothing here measures "
        "performance on unseen queries.",
        "",
        f"- Run `{run.name}` (commit `{manifest.git_commit[:12]}`, dirty `{manifest.git_dirty}`, "
        f"device `{manifest.device}`), environments {inputs['environments']}, query bound "
        f"{inputs['max_queries']}{' (a bounded sample)' if capped else ''}.",
        f"- Decile grid: {inputs['budgets']}, with the 30% point treated like every other.",
        "- Exact float64 scoring of every corpus document for every arm. Ties go to the lower row. "
        "No document is excluded by id.",
        "",
        "## Reading these tables",
        "",
        "- The **all-positive oracle** saw the positives. It is a privileged ceiling for their "
        "relevance-weighted mean similarity. It is not a ranking ceiling: a mask that raises "
        "positives can raise other documents more, and the tables show where it falls below the "
        "full query or L2S.",
        "- **Held-out transfer** is built from one positive and scored on another of the same "
        "query (multi-positive queries only). All-label metrics for this arm count the document "
        "that built it; use `paired-held` for what happened to the others.",
        "- L2S was fitted on the held-out positives as well as the others. Held-out transfer was "
        "not, so `held minus L2S` measures an in-sample fit against an out-of-construction mask, "
        "not two equally informed methods.",
        "- The **pair-optimal** mask depends on the target document, so it is a secondary "
        "diagnostic of that target's rank. For single-positive queries it is the oracle's mask.",
        "- Single-positive and multi-positive queries are separate cohorts and are never pooled.",
        "- Intervals are pointwise bootstrap intervals over queries and are exploratory. Competing "
        "documents are other documents, not confirmed irrelevant ones.",
        "",
        "## Provenance of the L2S checkpoint",
        "",
        *_markdown(tables["provenance"]),
        "",
        "## Coverage",
        "",
        *_markdown(tables["coverage"]),
        "",
        "## Official metrics, full training judgments",
        "",
        *_markdown(tables["metrics"]),
        "",
        "## Paired differences, with queries better and worse",
        "",
        *_markdown(tables["paired"]),
        "",
        "## Held-out positives of multi-positive queries",
        "",
        "Same partitions for every arm. Measures use only the held-out positives.",
        "",
        *_markdown(tables["paired-held"]),
        "",
        "## Positive ranks, and how scores moved against the full query",
        "",
        *_markdown(tables["positions"]),
        "",
        "## Agreement with the oracle mask",
        "",
        *_markdown(tables["agreement"]),
        "",
        "## Relationship among a query's positives (multi-positive)",
        "",
        "`pair_jaccard` is the mean overlap of pair-optimal masks. `sign_consensus` is the share "
        "of the oracle mask's coordinates on which all positives agree in sign, "
        "`positive_consensus` the share on which all are positive. `oracle_regret` is the regret "
        "of the all-positive mask against each positive's own optimum, `held_regret` of a mask "
        "built from a disjoint positive, each over mass.",
        "",
        *_markdown(tables["relationship"]),
        "",
        "## Checks and cost",
        "",
        *_markdown(tables["checks"]),
        "",
    ]
    readme = target / "README.md"
    readme.write_text("\n".join(lines), encoding="utf-8")
    return readme
