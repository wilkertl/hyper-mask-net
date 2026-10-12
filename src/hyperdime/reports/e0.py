"""The E0 report and gate G1, generated from result files only (F3 research D6, research R7).

Nothing numeric is typed here: every value comes from published evaluation and selector
artifacts, and every table row names the run it came from.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperdime.artifacts import latest_artifact
from hyperdime.contracts.manifests import read_manifest
from hyperdime.contracts.schemas import EvaluationResult, load_records
from hyperdime.evaluation.statistics import holm, paired_bootstrap_ci, paired_randomization_test

METHODS = ("full", "prefix", "random", "global-importance", "l2s", "selected", "oracle")
HEADLINE_K = 307
G1_K = 256
ALPHA = 0.05


@dataclass
class EnvironmentResults:
    environment: str
    run: str
    commit: str
    results: dict[str, dict[tuple[str, int], EvaluationResult]]  # query set -> (method, k)
    checks: dict[str, Any]
    official: dict[str, Any]
    candidates: dict[str, Any] | None
    selector_runs: dict[str, str]


def load(root: Path, environments: Sequence[str]) -> list[EnvironmentResults]:
    loaded = []
    for environment in environments:
        directory = latest_artifact(root, "evaluation", environment)
        results: dict[str, dict[tuple[str, int], EvaluationResult]] = {}
        for path in sorted(directory.glob("results-*.jsonl")):
            name = path.stem.removeprefix("results-")
            results[name] = {(r.selector, r.k): r for r in load_records(EvaluationResult, path)}
        sources = json.loads((directory / "inputs.json").read_text(encoding="utf-8"))
        selectors = root / "selectors"
        official = _json(
            selectors / "official" / environment / sources["official"] / "selection.json"
        )
        candidates = (
            _json(selectors / "candidates" / environment / sources["candidates"] / "selection.json")
            if "candidates" in sources
            else None
        )
        loaded.append(
            EnvironmentResults(
                environment,
                directory.name,
                read_manifest(directory).git_commit,
                results,
                _json(directory / "checks.json"),
                official,
                candidates,
                dict(sources),
            )
        )
    return loaded


def g1(environments: Sequence[EnvironmentResults], regenerates: bool) -> list[dict[str, Any]]:
    """Research R7's four criteria, each with its value per environment and whether it is met."""
    rows = []
    kl = {e.environment: e.official["cells"][e.official["chosen"]] for e in environments}
    rows.append(
        {
            "criterion": "(a) validation KL of the selector below the uniform predictor's",
            "value": "; ".join(
                f"{env} {c['val_kl']:.6f} vs {c['uniform_kl']:.6f}" for env, c in kl.items()
            ),
            "met": all(c["val_kl"] < c["uniform_kl"] for c in kl.values()),
        }
    )
    identical = {
        f"{e.environment}/{name}": all(check["k1024_identical"].values())
        for e in environments
        for name, check in e.checks.items()
    }
    rows.append(
        {
            "criterion": "(b) k = 1024 gives the unmasked rankings for every method and query set",
            "value": f"{sum(identical.values())}/{len(identical)} query sets",
            "met": all(identical.values()),
        }
    )
    ordering = {}
    for e in environments:
        test = e.results["test"]
        values = [test[(m, G1_K)].metrics["ndcg@10"] for m in ("oracle", "l2s", "random")]
        ordering[e.environment] = values
    rows.append(
        {
            "criterion": f"(c) nDCG@10 at k = {G1_K}: oracle >= selector >= random",
            "value": "; ".join(
                f"{env} {o:.4f} >= {s:.4f} >= {r:.4f}" for env, (o, s, r) in ordering.items()
            ),
            "met": all(o >= s >= r for o, s, r in ordering.values()),
        }
    )
    rows.append(
        {
            "criterion": "(d) the report regenerates byte-identically from result files",
            "value": "two renders from freshly loaded files compared",
            "met": regenerates,
        }
    )
    return rows


def significance(environments: Sequence[EnvironmentResults]) -> list[dict[str, Any]]:
    """Paired tests at the 30% budget on each test set, against full and against the selector;
    Holm-adjusted over the whole table."""
    rows: list[dict[str, Any]] = []
    for e in environments:
        test = e.results["test"]
        references = {"full": test[("full", 1024)], "l2s": test[("l2s", HEADLINE_K)]}
        for method in METHODS:
            if method == "full" or (method, HEADLINE_K) not in test:
                continue
            result = test[(method, HEADLINE_K)]
            for against, reference in references.items():
                if method == against:
                    continue
                qids = sorted(result.per_query)
                xs = [result.per_query[q]["ndcg@10"] for q in qids]
                ys = [reference.per_query[q]["ndcg@10"] for q in qids]
                low, high = paired_bootstrap_ci(xs, ys)
                rows.append(
                    {
                        "environment": e.environment,
                        "method": method,
                        "against": against,
                        "diff": sum(xs) / len(xs) - sum(ys) / len(ys),
                        "ci_low": low,
                        "ci_high": high,
                        "p": paired_randomization_test(xs, ys),
                    }
                )
    adjusted = holm({str(i): row["p"] for i, row in enumerate(rows)})
    for i, row in enumerate(rows):
        row["p_holm"] = adjusted[str(i)]
    return rows


def render(
    environments: Sequence[EnvironmentResults], regenerates: bool
) -> tuple[str, dict[str, str]]:
    tables: dict[str, list[list[str]]] = {}
    methods = [
        m
        for m in METHODS
        if any((m, HEADLINE_K) in e.results["test"] or m == "full" for e in environments)
    ]
    header = ["environment", "query set", "run", *(f"{m}" for m in methods)]
    l2s_rows = []
    for e in environments:
        test = e.results["test"]
        cells = [
            _fmt(test[("full", 1024)].metrics["ndcg@10"]) if m == "full"
            else _fmt(test[(m, HEADLINE_K)].metrics["ndcg@10"]) if (m, HEADLINE_K) in test else "—"
            for m in methods
        ]  # fmt: skip
        l2s_rows.append([e.environment, "test", e.run, *cells])
    tables["l2s_style"] = [header, *l2s_rows]
    sig = significance(environments)
    tables["significance"] = [
        ["environment", "method", "against", "diff", "ci_low", "ci_high", "p", "p_holm"],
        *[_significance_row(r) for r in sig],
    ]
    dime_sets = []
    for e in environments:
        for name in sorted(e.results):
            if name == "test":
                continue
            dime_sets.append((e, name))
            budgets = sorted({k for (_, k) in e.results[name]})
            for metric in ("ndcg@10", "ap"):
                rows = [["method", *(f"k={k}" for k in budgets)]]
                for m in methods:
                    if m == "full":
                        value = _fmt(e.results[name][("full", 1024)].metrics[metric])
                        rows.append([m, *([value] * len(budgets))])
                    elif (m, budgets[0]) in e.results[name]:
                        rows.append(
                            [m, *(_fmt(e.results[name][(m, k)].metrics[metric]) for k in budgets)]
                        )
                tables[f"dime_{name}_{metric.replace('@', '')}"] = rows
    criteria = g1(environments, regenerates)
    tables["g1"] = [
        ["criterion", "value", "met"],
        *[[c["criterion"], c["value"], _yes(c["met"])] for c in criteria],
    ]
    selection_rows = [
        ["environment", "official cell", "official validation nDCG@10", "selected", "family scores"]
    ]
    for e in environments:
        chosen = e.official["chosen"]
        scores = e.candidates["family_scores"] if e.candidates else {}
        selection_rows.append([
            e.environment, chosen, _fmt(e.official["cells"][chosen]["validation_ndcg@10"]),
            e.candidates["selected"] if e.candidates else "official (no candidates run)",
            ", ".join(f"{f} {_fmt(s)}" for f, s in sorted(scores.items())),
        ])  # fmt: skip
    tables["selection"] = selection_rows

    lines = [
        "# E0: Learning-to-Select reproduction and gate G1",
        "",
        "Generated by `hyperdime report E0` from the evaluation and selector artifacts named",
        "below; no number is typed by hand. Protocol: in-domain. Scoring: exact inner product over",
        "the full corpus. Learning-to-Select and DIME numbers are cited from their papers, never",
        "copied here.",
        "",
        *[
            f"- `{e.environment}`: evaluation `{e.run}` (commit `{e.commit[:12]}`), selectors "
            + ", ".join(f"{kind} `{run}`" for kind, run in sorted(e.selector_runs.items()))
            for e in environments
        ],
        "",
        "## Gate G1 (research R7)",
        "",
        *_markdown(tables["g1"]),
        "",
        f"**G1: {'pass' if all(c['met'] for c in criteria) else 'fail'}**",
        "",
        f"## Learning-to-Select format: nDCG@10 at k = {HEADLINE_K} (30%), full at 1024",
        "",
        *_markdown(tables["l2s_style"]),
        "",
        f"## Paired tests at k = {HEADLINE_K}",
        "",
        "Paired randomization tests, Holm-adjusted over the table, with 95% bootstrap intervals.",
        "",
        *_markdown(tables["significance"]),
        "",
        "## Selector selection (validation only)",
        "",
        *_markdown(tables["selection"]),
    ]
    for e, name in dime_sets:
        for metric in ("ndcg@10", "ap"):
            key = f"dime_{name}_{metric.replace('@', '')}"
            lines += [
                "",
                f"## DIME format: {name} ({e.environment}), {metric} by budget",
                "",
                *_markdown(tables[key]),
            ]
    csvs = {name: _csv(rows) for name, rows in tables.items()}
    return "\n".join(lines) + "\n", csvs


def write(root: Path, environments: Sequence[str], output: Path) -> Path:
    """Render twice from freshly loaded files; criterion (d) is whether the renders match."""
    first = render(load(root, environments), True)
    second = render(load(root, environments), True)
    text, csvs = render(load(root, environments), first == second)
    (output / "tables").mkdir(parents=True, exist_ok=True)
    (output / "README.md").write_text(text, encoding="utf-8")
    for name, content in csvs.items():
        (output / "tables" / f"{name}.csv").write_text(content, encoding="utf-8")
    return output / "README.md"


def _significance_row(row: dict[str, Any]) -> list[str]:
    signed = [_fmt(row[key], "+") for key in ("diff", "ci_low", "ci_high")]
    return [
        row["environment"],
        row["method"],
        row["against"],
        *signed,
        _fmt(row["p"]),
        _fmt(row["p_holm"]),
    ]


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: float, sign: str = "") -> str:
    return f"{value:{sign}.4f}"


def _yes(flag: bool) -> str:
    return "yes" if flag else "no"


def _markdown(rows: Sequence[Sequence[str]]) -> list[str]:
    head, *body = rows
    return [
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
        *("| " + " | ".join(r) + " |" for r in body),
    ]


def _csv(rows: Sequence[Sequence[str]]) -> str:
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    return buffer.getvalue()
