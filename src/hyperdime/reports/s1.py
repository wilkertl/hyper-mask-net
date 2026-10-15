"""The S1 report and its drafted decision, generated from result files only (F5 research D7–D9).

Nothing numeric is typed here: every value comes from the latest `compare-selectors` run, the
retrieval grid, and the seeded selectors it names. Protocol: in-domain. Scoring: exact.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperdime.artifacts import latest_artifact
from hyperdime.contracts.manifests import read_manifest
from hyperdime.contracts.schemas import Criterion, EvaluationResult, GateDecision, load_records
from hyperdime.evaluation.statistics import holm, paired_bootstrap_ci, paired_randomization_test
from hyperdime.evaluation.transfer import ALPHA
from hyperdime.reports.e0 import _csv, _fmt, _markdown, _yes
from hyperdime.reports.e1 import NOTES, note

HEADLINE_K = 307
DIM = 1024
MIN_SEEDS = 3
BASELINES = ("prefix", "random", "global-importance", "global-selector")
SUMMARIES = ("mean 10%-90%", f"k = {HEADLINE_K}")
TAG = "s1-selector-objective"
DEFAULT_ADR = Path("docs/decisions/ADR-0009-s1-selector-objective.md")
_STATUS = re.compile(r"^- \*\*Status:\*\* (\w+)", re.MULTILINE)


@dataclass
class S1Results:
    run: str
    commit: str
    finished: str
    environments: list[str]
    seeds: list[int]
    results: dict[tuple[str, str], dict[tuple[str, int], EvaluationResult]]  # (env, set) -> (m, k)
    overlap: dict[str, Any]
    fit: dict[str, Any]
    sources: dict[str, Any]
    grids: dict[str, dict[str, Any]]  # env -> grid selection.json
    histories: dict[str, dict[int, list[dict[str, float]]]]  # env -> seed -> history

    @property
    def budgets(self) -> list[int]:
        return sorted({k for found in self.results.values() for _, k in found if k < DIM})


def load(root: Path) -> S1Results:
    directory = latest_artifact(root, "evaluation/S1", "all")
    manifest = read_manifest(directory)
    sources = _json(directory / "inputs.json")
    environments = sorted(e for e in sources if e != "global")
    results: dict[tuple[str, str], dict[tuple[str, int], EvaluationResult]] = {}
    for path in sorted(directory.glob("results-*.jsonl")):
        records = load_records(EvaluationResult, path)
        test = records[0].environment
        results[(test, path.stem.removeprefix(f"results-{test}-"))] = {
            (r.selector, r.k): r for r in records
        }
    seeds = sorted(int(s.removeprefix("seed-")) for s in sources[environments[0]]["retrieval"])
    if len(seeds) < MIN_SEEDS:
        raise ValueError(f"S1 needs at least {MIN_SEEDS} seeds per environment; found {seeds}")
    selectors = root / "selectors"
    grids = {
        env: _json(selectors / "retrieval-grid" / env / sources[env]["grid"] / "selection.json")
        for env in environments
    }
    histories = {
        env: {
            seed: next(
                iter(
                    _json(
                        selectors
                        / "retrieval-seeded"
                        / env
                        / f"seed-{seed}"
                        / sources[env]["retrieval"][f"seed-{seed}"]
                        / "history.json"
                    ).values()
                )
            )  # fmt: skip
            for seed in seeds
        }
        for env in environments
    }
    return S1Results(
        directory.name,
        manifest.git_commit,
        f"{manifest.finished_at:%Y-%m-%d}" if manifest.finished_at else "",
        environments,
        seeds,
        results,
        _json(directory / "overlap.json"),
        _json(directory / "fit.json"),
        sources,
        grids,
        histories,
    )


def unchanged_starts(s1: S1Results) -> list[str]:
    """Environments whose selected cell starts from L2S and whose every seed keeps its step-0
    checkpoint: there, the retrieval selector is the L2S selector itself."""
    found = []
    for env in s1.environments:
        grid = s1.grids[env]
        if grid["cells"][grid["selected"]]["cell"]["init"] != "l2s":
            continue
        best = [
            max(h, key=lambda row: row["val_rerank_ndcg10"])["step"]
            for h in s1.histories[env].values()
        ]
        if all(step == 0 for step in best):
            found.append(env)
    return found


def unchanged_note(env: str) -> str:
    return (
        "the selected cell starts from the L2S selector and every seed keeps its step-0 "
        "checkpoint (no training step improved validation), so `retrieval:"
        f"{env}` is the L2S selector itself and its difference from L2S is zero by construction."
    )


def per_query(
    s1: S1Results, key: tuple[str, str], method: str, k: int, metric: str = "ndcg@10"
) -> dict[str, float]:
    """Per-query metric of ``method`` at budget ``k``; ``l2s:<env>`` and ``retrieval:<env>`` are
    averaged over seeds per query (as in E1)."""
    names = [method]
    if method.split(":")[0] in ("l2s", "retrieval"):
        names = [f"{method}:s{s}" for s in s1.seeds]
    found = s1.results[key]
    queries = found[(names[0], k)].per_query
    return {q: sum(found[(n, k)].per_query[q][metric] for n in names) / len(names) for q in queries}


def summary(
    s1: S1Results, key: tuple[str, str], method: str, which: str, metric: str = "ndcg@10"
) -> dict[str, float]:
    """Per query: the mean over the 10%-90% budgets, or the value at the headline budget."""
    if which == SUMMARIES[1]:
        return per_query(s1, key, method, HEADLINE_K, metric)
    by_k = [per_query(s1, key, method, k, metric) for k in s1.budgets]
    return {q: sum(values[q] for values in by_k) / len(by_k) for q in by_k[0]}


def compare(
    s1: S1Results, key: tuple[str, str], which: str, metric: str = "ndcg@10"
) -> dict[str, Any]:
    """Retrieval-trained selector against L2S on one query set, paired over queries."""
    env = key[0]
    new = summary(s1, key, f"retrieval:{env}", which, metric)
    old = summary(s1, key, f"l2s:{env}", which, metric)
    queries = sorted(new)
    a, b = [new[q] for q in queries], [old[q] for q in queries]
    low, high = paired_bootstrap_ci(a, b)
    return {
        "environment": env,
        "query_set": key[1],
        "summary": which,
        "l2s": sum(b) / len(b),
        "retrieval": sum(a) / len(a),
        "diff": (sum(a) - sum(b)) / len(a),
        "ci_low": low,
        "ci_high": high,
        "p": paired_randomization_test(a, b),
    }


def comparisons(s1: S1Results) -> list[dict[str, Any]]:
    """Test sets: one row per environment and summary, Holm over environments per summary."""
    rows = []
    for which in SUMMARIES:
        found = [compare(s1, (env, "test"), which) for env in s1.environments]
        adjusted = holm({r["environment"]: r["p"] for r in found})
        rows += [{**r, "p_holm": adjusted[r["environment"]]} for r in found]
    return rows


def criteria(s1: S1Results) -> list[dict[str, Any]]:
    """SC-001 and SC-002 on the mean over the 10%-90% budgets (spec, research D7)."""
    rows = [r for r in comparisons(s1) if r["summary"] == SUMMARIES[0]]
    better = sum(1 for r in rows if r["diff"] > 0 and r["p_holm"] <= ALPHA)
    worse = sum(1 for r in rows if r["diff"] < 0 and r["p_holm"] <= ALPHA)
    required = (2 * len(rows) + 2) // 3
    return [
        {
            "name": "SC-001 better than L2S (mean nDCG@10 over 10%-90%, Holm, alpha 0.05)",
            "value": float(better),
            "shown": f"{better} of {len(rows)} environments",
            "threshold": f">= {required} of {len(rows)}",
            "met": better >= required,
        },
        {
            "name": "SC-002 worse than L2S in no environment (same test)",
            "value": float(worse),
            "shown": f"{worse} of {len(rows)} environments",
            "threshold": "0",
            "met": worse == 0,
        },
    ]


def _methods(s1: S1Results, env: str) -> list[str]:
    families = sorted(s1.grids[env]["family_best"])
    return [
        *BASELINES,
        f"l2s:{env}",
        f"retrieval:{env}",
        *(f"family:{f}:{env}" for f in families),
    ]


def progression(s1: S1Results) -> list[list[str]]:
    header = [f"{round(100 * k / DIM)}% (k={k})" for k in [*s1.budgets, DIM]]
    rows = [["test", "query set", "method", *header, "note"]]
    for key in sorted(s1.results):
        env, query_set = key
        full = _mean(per_query(s1, key, "full", DIM))
        rows.append([env, query_set, "full", *[_fmt(full)] * len(header), "—"])
        for method in _methods(s1, env):
            values = [_mean(per_query(s1, key, method, k)) for k in s1.budgets]
            rows.append([env, query_set, method, *(_fmt(v) for v in values), _fmt(full), note(env)])
    return rows


def vs_full(s1: S1Results) -> list[list[str]]:
    rows = [["test", "query set", "budget", "full", "l2s - full", "retrieval - full"]]
    for key in sorted(s1.results):
        full = _mean(per_query(s1, key, "full", DIM))
        for k in s1.budgets:
            l2s = _mean(per_query(s1, key, f"l2s:{key[0]}", k))
            new = _mean(per_query(s1, key, f"retrieval:{key[0]}", k))
            rows.append([*key, str(k), _fmt(full), _fmt(l2s - full, "+"), _fmt(new - full, "+")])
    return rows


def _comparison_rows(rows: Sequence[dict[str, Any]], holm_column: bool) -> list[list[str]]:
    head = ["test", "query set", "summary", "L2S", "retrieval", "diff", "ci_low", "ci_high", "p"]
    table = [head + (["p_holm"] if holm_column else []) + ["note"]]
    for r in rows:
        table.append([
            r["environment"], r["query_set"], r["summary"], _fmt(r["l2s"]), _fmt(r["retrieval"]),
            _fmt(r["diff"], "+"), _fmt(r["ci_low"], "+"), _fmt(r["ci_high"], "+"), _fmt(r["p"]),
            *([_fmt(r["p_holm"])] if holm_column else []), note(r["environment"]),
        ])  # fmt: skip
    return table


def trec(s1: S1Results) -> list[list[str]]:
    rows = []
    for key in sorted(k for k in s1.results if k[1] != "test"):
        for which in SUMMARIES:
            for metric in ("ndcg@10", "ap"):
                row = compare(s1, key, which, metric)
                rows.append({**row, "summary": f"{which}, {metric}"})
    return _comparison_rows(rows, holm_column=False)


def families(s1: S1Results) -> list[list[str]]:
    """Each (variant, init) family's validation-best cell on test, seed 0 (context, FR-005)."""
    rows = [["test", "family", "cell", *SUMMARIES]]
    for env in s1.environments:
        key = (env, "test")
        rows.append([env, "l2s (3 seeds)", "—", *(
            _fmt(_mean(summary(s1, key, f"l2s:{env}", w))) for w in SUMMARIES
        )])  # fmt: skip
        for family, cell in sorted(s1.grids[env]["family_best"].items()):
            method = f"family:{family}:{env}"
            rows.append(
                [env, family, cell, *(_fmt(_mean(summary(s1, key, method, w))) for w in SUMMARIES)]
            )
    return rows


def grid(s1: S1Results) -> list[list[str]]:
    rows = [["environment", "cell", "validation mean nDCG@10", "soft mass", "selectable", "role"]]
    for env in s1.environments:
        record = s1.grids[env]
        best = set(record["family_best"].values())
        for key, cell in record["cells"].items():
            role = "selected" if key == record["selected"] else "family best" if key in best else ""
            rows.append([
                env, key, _fmt(cell["validation_mean"]), _fmt(cell["soft_mass"]),
                _yes(cell["selectable"]), role,
            ])  # fmt: skip
    return rows


def overlap(s1: S1Results) -> list[list[str]]:
    rows = [["environment", "budget", "retrieval vs L2S", "L2S seeds", "retrieval seeds"]]
    for env in s1.environments:
        for k, values in sorted(s1.overlap[env].items(), key=lambda item: int(item[0])):
            rows.append([
                env, k, _fmt(values["retrieval_l2s"]), _fmt(values["l2s_seeds"]),
                _fmt(values["retrieval_seeds"]),
            ])  # fmt: skip
    return rows


def fit(s1: S1Results) -> list[list[str]]:
    rows = [["environment", "selector", "mean gain over global importance", "p", "p_holm", "fits"]]
    for env in s1.environments:
        for selector in ("l2s", "retrieval"):
            r = s1.fit[selector][env]
            rows.append([
                env, selector, _fmt(r["mean_gain"], "+"), _fmt(r["p_value"]), _fmt(r["p_holm"]),
                _yes(r["fits"]),
            ])  # fmt: skip
    return rows


def curves(s1: S1Results) -> list[list[str]]:
    rows = [["environment", "seed", "steps", "best step", "best validation rerank nDCG@10",
             "final soft mass"]]  # fmt: skip
    for env in s1.environments:
        for seed, history in sorted(s1.histories[env].items()):
            top = max(history, key=lambda h: h["val_rerank_ndcg10"])
            rows.append([
                env, str(seed), str(int(history[-1]["step"])), str(int(top["step"])),
                _fmt(top["val_rerank_ndcg10"]), _fmt(history[-1]["soft_mass"]),
            ])  # fmt: skip
    return rows


def render(s1: S1Results) -> tuple[str, dict[str, str]]:
    computed = criteria(s1)
    tables = {
        "s1": [
            ["criterion", "value", "threshold", "met"],
            *[[c["name"], c["shown"], c["threshold"], _yes(c["met"])] for c in computed],
        ],
        "comparison": _comparison_rows(comparisons(s1), holm_column=True),
        "progression": progression(s1),
        "vs_full": vs_full(s1),
        "trec": trec(s1),
        "families": families(s1),
        "grid": grid(s1),
        "overlap": overlap(s1),
        "fit": fit(s1),
        "curves": curves(s1),
    }
    cells = ", ".join(f"{env} `{s1.grids[env]['selected']}`" for env in s1.environments)
    lines = [
        "# S1: a selector trained for retrieval against Learning-to-Select",
        "",
        "Protocol: in-domain. Scoring: exact inner product over the full corpus, hard top-k mask.",
        f"Budgets: {', '.join(str(k) for k in s1.budgets)} (10%-90%); 100% is the full query.",
        f"Selectors: the retrieval grid's selected cell per environment ({cells}) and E1's",
        f"`wide-grid` L2S selectors, seeds {', '.join(str(s) for s in s1.seeds)}; seed scores",
        "are averaged per query before every paired test (F5 research D7).",
        "",
        f"- compare-selectors `{s1.run}` (commit `{s1.commit[:12]}`)",
        "",
        "## Criteria",
        "",
        *_markdown(tables["s1"]),
        "",
        "## Against L2S on the test sets",
        "",
        "Holm over the environments, separately for each summary.",
        "",
        *_markdown(tables["comparison"]),
        "",
        "## nDCG@10 by budget",
        "",
        *_markdown(tables["progression"]),
        "",
        "## Against full 1024",
        "",
        *_markdown(tables["vs_full"]),
        "",
        "## TREC DL sets (context, not in the criteria)",
        "",
        *_markdown(tables["trec"]),
        "",
        "## Families (validation-best cell of each variant and start, seed 0; context)",
        "",
        *_markdown(tables["families"]),
        "",
        "## Where the difference comes from",
        "",
        "Top-k overlap on test queries between the two selectors (seed-matched) and between",
        "seeds of each (noise floor):",
        "",
        *_markdown(tables["overlap"]),
        "",
        "Ranking fit on validation queries (top-k agreement with the oracle mask, gain over",
        "global importance, F4 research D2), Holm over environments per selector:",
        "",
        *_markdown(tables["fit"]),
        "",
        "Training of the seeded retrieval selectors:",
        "",
        *_markdown(tables["curves"]),
        "",
        "## Notes",
        "",
        *[f"- **{env}**: {text}" for env, text in sorted(NOTES.items()) if env in s1.environments],
        *[f"- **{env}**: {unchanged_note(env)}" for env in unchanged_starts(s1)],
        "",
        "Every grid cell is in `tables/grid.csv`.",
    ]
    return "\n".join(lines) + "\n", {name: _csv(rows) for name, rows in tables.items()}


def write(root: Path, output: Path) -> Path:
    text, csvs = render(load(root))
    (output / "tables").mkdir(parents=True, exist_ok=True)
    (output / "README.md").write_text(text, encoding="utf-8")
    for name, content in csvs.items():
        (output / "tables" / f"{name}.csv").write_text(content, encoding="utf-8")
    return output / "README.md"


def decision(s1: S1Results, adr: Path) -> GateDecision:
    computed = criteria(s1)
    headline = [r for r in comparisons(s1) if r["summary"] == SUMMARIES[1]]
    context = {
        "headline": ", ".join(
            f"{r['environment']} {_fmt(r['diff'], '+')} (Holm p {_fmt(r['p_holm'])})"
            for r in headline
        ),
        "selected": ", ".join(f"{env} {s1.grids[env]['selected']}" for env in s1.environments),
        **{f"note_{env}": NOTES[env] for env in s1.environments if env in NOTES},
        **{f"unchanged_{env}": unchanged_note(env) for env in unchanged_starts(s1)},
    }
    return GateDecision(
        gate="S1",
        outcome="pass" if all(c["met"] for c in computed) else "fail",
        criteria=[
            Criterion(name=c["name"], value=c["value"], threshold=c["threshold"], met=c["met"])
            for c in computed
        ],
        context=context,
        evidence=[
            s1.run,
            *sorted(
                {run for env in s1.environments for run in s1.sources[env]["retrieval"].values()}
            ),
        ],  # fmt: skip
        adr=adr.as_posix(),
        tag=TAG,
    )


def proposed_adr(s1: S1Results, record: GateDecision) -> str:
    computed = criteria(s1)
    lines = [
        "# ADR-0009: Experiment S1 — a selector trained for retrieval",
        "",
        "- **Status:** proposed",
        f"- **Date:** {s1.finished}",
        "- **Issue:** F5 (`specs/006-selector-objective`), experiment S1",
        "",
        "## Context",
        "",
        "S1 asks whether the same linear selector, trained on masked retrieval instead of the",
        "oracle's importance distribution, retrieves better than Learning-to-Select over the",
        "10%-90% budgets (ADR-0008). This draft is generated by `hyperdime decide S1` from the",
        "result files of `reports/S1/README.md`.",
        "",
        "## Evidence",
        "",
        f"From compare-selectors `{s1.run}` (commit `{s1.commit[:12]}`); every value is in",
        "`reports/S1/README.md`.",
        "",
        *[
            f"- **{c['name']}**: {c['shown']}; threshold {c['threshold']}; "
            f"{'met' if c['met'] else 'not met'}."
            for c in computed
        ],  # fmt: skip
        "",
        "Context, not decisive:",
        "",
        *[f"- **{key}**: {value}" for key, value in sorted(record.context.items())],
        "",
        "## Decision",
        "",
        f"The S1 criteria read **{record.outcome}**. The decision is the researcher's: accept",
        "this reading, or amend it here with the reasons.",
        "",
        "## Consequences",
        "",
        "- If S1 passes: the retrieval-trained selector replaces L2S as the selector later work",
        "  builds on.",
        "- If S1 fails: the next candidates are higher capacity or a budget policy, each",
        "  justified by the S1 diagnostics (ADR-0008).",
    ]
    return "\n".join(lines) + "\n"


def write_adr(s1: S1Results, record: GateDecision, path: Path) -> Path:
    """Write the proposed ADR; an existing ADR is overwritten only while it is still proposed."""
    if path.exists():
        found = _STATUS.search(path.read_text(encoding="utf-8"))
        if found is None or found.group(1) != "proposed":
            raise ValueError(f"{path} is no longer proposed; it is not overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(proposed_adr(s1, record), encoding="utf-8")
    return path


def _mean(values: dict[str, float]) -> float:
    return sum(values.values()) / len(values)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
