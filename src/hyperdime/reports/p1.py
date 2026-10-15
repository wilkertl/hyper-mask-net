"""The P1 report: what the corpus's geometry alone does, generated from result files only.

Reads the latest `probe-geometry` run and the S1 run it names; the variance method is paired per
query with S1's selectors and baselines (F6 spec). Protocol: in-domain. Scoring: exact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hyperdime.artifacts import latest_artifact
from hyperdime.contracts.manifests import read_manifest
from hyperdime.contracts.schemas import EvaluationResult, load_records
from hyperdime.evaluation.statistics import holm, paired_bootstrap_ci, paired_randomization_test
from hyperdime.reports import s1
from hyperdime.reports.e0 import _csv, _fmt, _markdown, _yes
from hyperdime.reports.e1 import note

DIM = 1024
PROMISE = 0.5
AGAINST = {"L2S": "l2s:{env}", "S1": "retrieval:{env}", "global": "global-importance"}


def load(root: Path) -> tuple[str, str, s1.S1Results, dict[str, dict[str, Any]]]:
    """The P1 run's name and commit, the S1 results with ``variance`` added, and the geometry."""
    directory = latest_artifact(root, "evaluation/P1", "all")
    sources = json.loads((directory / "inputs.json").read_text(encoding="utf-8"))
    results = s1.load(root)
    if results.run != sources["s1"]:
        raise ValueError(
            f"P1 was scored against S1 run {sources['s1']}, not the latest {results.run}"
        )
    geometry = {}
    for env in results.environments:
        geometry[env] = json.loads((directory / f"geometry-{env}.json").read_text(encoding="utf-8"))
        for path in sorted(directory.glob(f"results-{env}-*.jsonl")):
            key = (env, path.stem.removeprefix(f"results-{env}-"))
            for r in load_records(EvaluationResult, path):
                if r.selector == "variance":
                    results.results[key][("variance", r.k)] = r
    return directory.name, read_manifest(directory).git_commit, results, geometry


def comparisons(results: s1.S1Results) -> list[dict[str, Any]]:
    """Variance against each reference on the test sets, mean over 10%-90%; Holm over the three
    environments for each reference."""
    rows = []
    which = s1.SUMMARIES[0]
    for label, template in AGAINST.items():
        found: list[dict[str, Any]] = []
        for env in results.environments:
            key = (env, "test")
            new = s1.summary(results, key, "variance", which)
            old = s1.summary(results, key, template.format(env=env), which)
            queries = sorted(new)
            a, b = [new[q] for q in queries], [old[q] for q in queries]
            low, high = paired_bootstrap_ci(a, b)
            found.append({
                "environment": env, "against": label, "variance": sum(a) / len(a),
                "reference": sum(b) / len(b), "diff": (sum(a) - sum(b)) / len(a),
                "ci_low": low, "ci_high": high, "p": paired_randomization_test(a, b),
            })  # fmt: skip
        adjusted = holm({r["environment"]: r["p"] for r in found})
        rows += [{**r, "p_holm": adjusted[r["environment"]]} for r in found]
    return rows


def promise(results: s1.S1Results) -> list[dict[str, Any]]:
    """Share of L2S's gain over global importance that the variance method recovers (SC-001)."""
    rows = []
    which = s1.SUMMARIES[0]
    for env in results.environments:
        key = (env, "test")
        means = {
            name: _mean(s1.summary(results, key, method, which))
            for name, method in (
                ("variance", "variance"), ("l2s", f"l2s:{env}"), ("global", "global-importance"),
            )
        }  # fmt: skip
        gain = means["l2s"] - means["global"]
        share = (means["variance"] - means["global"]) / gain if gain else float("nan")
        rows.append({"environment": env, **means, "share": share, "recovers": share >= PROMISE})
    return rows


def best_budget(results: s1.S1Results, env: str, method: str) -> int:
    """The budget (full 1024 included) with the highest mean test nDCG@10."""
    key = (env, "test")
    scores = {k: _mean(s1.per_query(results, key, method, k)) for k in results.budgets}
    scores[DIM] = _mean(s1.per_query(results, key, "full", DIM))
    return max(scores, key=lambda k: (scores[k], -k))


def render(root: Path) -> tuple[str, dict[str, str]]:
    run, commit, results, geometry = load(root)
    rows = comparisons(results)
    shares = promise(results)
    recovered = sum(1 for r in shares if r["recovers"])
    needed = (2 * len(shares) + 2) // 3
    dims = [["environment", "documents", "participation ratio", "share of 1024", "energy in mean",
             "best budget L2S", "best budget S1", "best budget variance"]]  # fmt: skip
    ratios, bests = {}, {}
    for env in results.environments:
        g = geometry[env]
        ratios[env], bests[env] = g["participation_ratio"], best_budget(results, env, f"l2s:{env}")
        dims.append([
            env, str(g["num_documents"]), f"{g['participation_ratio']:.1f}",
            _fmt(g["participation_ratio"] / DIM), _fmt(g["mean_energy"]), str(bests[env]),
            str(best_budget(results, env, f"retrieval:{env}")),
            str(best_budget(results, env, "variance")),
        ])  # fmt: skip
    order = sorted(results.environments, key=lambda e: ratios[e])
    matches = [bests[e] for e in order] == sorted(bests[e] for e in order)
    tables = {
        "p1": [
            ["reading", "value", "rule", "holds"],
            [
                "SC-001 variance recovers >= half of L2S's gain over global importance",
                f"{recovered} of {len(shares)} environments",
                f">= {needed} of {len(shares)}",
                _yes(recovered >= needed),
            ],
            [
                "SC-002 participation ratio orders L2S's best budget",
                " < ".join(f"{e} {ratios[e]:.1f} ({bests[e]})" for e in order),
                "same order (descriptive, three environments)",
                _yes(matches),
            ],
        ],  # fmt: skip
        "comparison": [
            [
                "test",
                "against",
                "variance",
                "reference",
                "diff",
                "ci_low",
                "ci_high",
                "p",
                "p_holm",
                "note",
            ],
            *[
                [
                    r["environment"],
                    r["against"],
                    _fmt(r["variance"]),
                    _fmt(r["reference"]),
                    _fmt(r["diff"], "+"),
                    _fmt(r["ci_low"], "+"),
                    _fmt(r["ci_high"], "+"),
                    _fmt(r["p"]),
                    _fmt(r["p_holm"]),
                    note(r["environment"]),
                ]
                for r in rows
            ],
        ],  # fmt: skip
        "promise": [
            [
                "test",
                "variance",
                "L2S",
                "global importance",
                "share of L2S's gain",
                "recovers half",
            ],
            *[
                [
                    r["environment"],
                    _fmt(r["variance"]),
                    _fmt(r["l2s"]),
                    _fmt(r["global"]),
                    _fmt(r["share"], "+"),
                    _yes(r["recovers"]),
                ]
                for r in shares
            ],
        ],  # fmt: skip
        "dimensionality": dims,
        "progression": _progression(results),
    }
    lines = [
        "# P1: what the corpus's geometry alone does",
        "",
        "Protocol: in-domain. Scoring: exact inner product over the full corpus, hard top-k mask.",
        "`variance` keeps the k dimensions with the largest `q_d² · Var(x_d)`, with the variance",
        "from document embeddings only (no judgment); it is paired per query with the S1 run's",
        "selectors (seeds averaged) and baselines.",
        "",
        f"- probe-geometry `{run}` (commit `{commit[:12]}`), S1 compare-selectors `{results.run}`",
        "",
        "## Readings",
        "",
        *_markdown(tables["p1"]),
        "",
        "## Variance against the selectors and global importance (mean nDCG@10 over 10%-90%)",
        "",
        "Holm over the environments, separately for each reference.",
        "",
        *_markdown(tables["comparison"]),
        "",
        *_markdown(tables["promise"]),
        "",
        "## Effective dimensionality and the best budget",
        "",
        "Participation ratio `(Σλ)² / Σλ²` of the document covariance; energy in the mean",
        "`‖μ‖² / E‖x‖²`. Three environments describe; they do not test a relation.",
        "",
        *_markdown(tables["dimensionality"]),
        "",
        "## nDCG@10 by budget",
        "",
        *_markdown(tables["progression"]),
    ]
    return "\n".join(lines) + "\n", {name: _csv(table) for name, table in tables.items()}


def _progression(results: s1.S1Results) -> list[list[str]]:
    header = [f"{round(100 * k / DIM)}% (k={k})" for k in [*results.budgets, DIM]]
    table = [["test", "query set", "method", *header]]
    for key in sorted(results.results):
        env = key[0]
        full = _mean(s1.per_query(results, key, "full", DIM))
        table.append([*key, "full", *[_fmt(full)] * len(header)])
        for method in ("prefix", "random", "global-importance", "variance", f"l2s:{env}",
                       f"retrieval:{env}"):  # fmt: skip
            values = [_mean(s1.per_query(results, key, method, k)) for k in results.budgets]
            table.append([*key, method, *(_fmt(v) for v in values), _fmt(full)])
    return table


def write(root: Path, output: Path) -> Path:
    text, csvs = render(root)
    (output / "tables").mkdir(parents=True, exist_ok=True)
    (output / "README.md").write_text(text, encoding="utf-8")
    for name, content in csvs.items():
        (output / "tables" / f"{name}.csv").write_text(content, encoding="utf-8")
    return output / "README.md"


def _mean(values: dict[str, float]) -> float:
    return sum(values.values()) / len(values)
