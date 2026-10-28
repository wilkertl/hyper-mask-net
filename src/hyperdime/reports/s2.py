"""The S2 report and its drafted decision, generated from result files only (F7 spec).

The geometric selector (S1's retrieval selector started from the corpus's geometric prior) is
compared with the S1 selector (SC-001), the label-free ``variance`` method it starts from (SC-002),
and L2S (context). Reuses the S1 report's loading and pairing. Protocol: in-domain. Exact scoring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hyperdime.contracts.schemas import Criterion, GateDecision
from hyperdime.evaluation.statistics import holm
from hyperdime.evaluation.transfer import ALPHA
from hyperdime.reports import s1
from hyperdime.reports.e0 import _csv, _fmt, _markdown, _yes
from hyperdime.reports.e1 import NOTES, note

DIM = 1024
CANDIDATE = "geometric"
AGAINST = {"S1 selector": "retrieval", "variance": "variance", "L2S": "l2s"}
TAG = "s2-geometric-prior"
DEFAULT_ADR = Path("docs/decisions/ADR-0010-s2-geometric-prior.md")


def load(root: Path) -> s1.S1Results:
    return s1.load(root, experiment="S2", candidate=CANDIDATE)


def comparisons(
    results: s1.S1Results,
    query_set: str = "test",
    candidate: str = CANDIDATE,
    against: dict[str, str] = AGAINST,
) -> list[dict[str, Any]]:
    """The geometric selector against each reference, per environment and summary; Holm over
    the environments for each reference and summary (test sets only)."""
    rows = []
    for label, reference in against.items():
        for which in s1.SUMMARIES:
            found = [
                {
                    **s1.compare(
                        results, (env, query_set), which, new_name=candidate, old_name=reference
                    ),
                    "against": label,
                }  # fmt: skip
                for env in results.environments
                if (env, query_set) in results.results
            ]
            adjusted = holm({r["environment"]: r["p"] for r in found})
            rows += [{**r, "p_holm": adjusted[r["environment"]]} for r in found]
    return rows


def criteria(results: s1.S1Results) -> list[dict[str, Any]]:
    rows = [r for r in comparisons(results) if r["summary"] == s1.SUMMARIES[0]]
    better = [
        r for r in rows if r["against"] == "S1 selector" and r["diff"] > 0 and r["p_holm"] <= ALPHA
    ]
    worse = [
        r for r in rows if r["against"] == "variance" and r["diff"] < 0 and r["p_holm"] <= ALPHA
    ]
    count = len(results.environments)
    required = (2 * count + 2) // 3
    return [
        {
            "name": "SC-001 better than the S1 selector (mean nDCG@10, 10%-90%, Holm, alpha 0.05)",
            "value": float(len(better)),
            "shown": f"{len(better)} of {count} environments",
            "threshold": f">= {required} of {count}",
            "met": len(better) >= required,
        },
        {
            "name": "SC-002 worse than the variance method in no environment (same test)",
            "value": float(len(worse)),
            "shown": f"{len(worse)} of {count} environments",
            "threshold": "0",
            "met": not worse,
        },
    ]


def _rows(
    rows: list[dict[str, Any]], holm_column: bool, candidate: str = CANDIDATE
) -> list[list[str]]:
    head = ["test", "query set", "against", "summary", "reference", candidate, "diff", "ci_low",
            "ci_high", "p"]  # fmt: skip
    table = [head + (["p_holm"] if holm_column else []) + ["note"]]
    for r in rows:
        table.append([
            r["environment"], r["query_set"], r["against"], r["summary"], _fmt(r["old"]),
            _fmt(r["new"]), _fmt(r["diff"], "+"), _fmt(r["ci_low"], "+"), _fmt(r["ci_high"], "+"),
            _fmt(r["p"]), *([_fmt(r["p_holm"])] if holm_column else []), note(r["environment"]),
        ])  # fmt: skip
    return table


def progression(
    results: s1.S1Results, selectors: tuple[str, ...] = ("retrieval", CANDIDATE)
) -> list[list[str]]:
    """Every method by budget; ``selectors`` are the seeded selectors after L2S, the candidate
    last, followed by the candidate grid's family bests."""
    header = [f"{round(100 * k / DIM)}% (k={k})" for k in [*results.budgets, DIM]]
    table = [["test", "query set", "method", *header, "note"]]
    for key in sorted(results.results):
        env = key[0]
        full = _mean(s1.per_query(results, key, "full", DIM))
        table.append([*key, "full", *[_fmt(full)] * len(header), "—"])
        methods = [
            *s1.BASELINES,
            "variance",
            f"l2s:{env}",
            *(f"{selector}:{env}" for selector in selectors),
        ]
        methods += [f"family:{f}:{env}" for f in sorted(results.grids[env]["family_best"])]
        for method in methods:
            values = [_mean(s1.per_query(results, key, method, k)) for k in results.budgets]
            table.append([*key, method, *(_fmt(v) for v in values), _fmt(full), note(env)])
    return table


def vs_full(
    results: s1.S1Results, names: tuple[str, ...] = ("variance", "retrieval", CANDIDATE)
) -> list[list[str]]:
    table = [["test", "query set", "budget", "full", *(f"{n} - full" for n in names)]]
    for key in sorted(results.results):
        full = _mean(s1.per_query(results, key, "full", DIM))
        for k in results.budgets:
            diffs = [_mean(s1.per_query(results, key, s1.method_name(n, key[0]), k)) - full
                     for n in names]  # fmt: skip
            table.append([*key, str(k), _fmt(full), *(_fmt(d, "+") for d in diffs)])
    return table


def overlap(results: s1.S1Results, candidate: str = CANDIDATE) -> list[list[str]]:
    table = [["environment", "budget", f"{candidate} vs L2S", "L2S seeds", f"{candidate} seeds"]]
    for env in results.environments:
        for k, values in sorted(results.overlap[env].items(), key=lambda item: int(item[0])):
            table.append([env, k, _fmt(values[f"{candidate}_l2s"]), _fmt(values["l2s_seeds"]),
                          _fmt(values[f"{candidate}_seeds"])])  # fmt: skip
    return table


def fit(
    results: s1.S1Results, selectors: tuple[str, ...] = ("l2s", "retrieval", CANDIDATE)
) -> list[list[str]]:
    table = [["environment", "selector", "mean gain over global importance", "p", "p_holm", "fits"]]
    for env in results.environments:
        for selector in selectors:
            r = results.fit[selector][env]
            table.append([env, selector, _fmt(r["mean_gain"], "+"), _fmt(r["p_value"]),
                          _fmt(r["p_holm"]), _yes(r["fits"])])  # fmt: skip
    return table


def render(results: s1.S1Results) -> tuple[str, dict[str, str]]:
    computed = criteria(results)
    trec_rows = [r for key in sorted({k[1] for k in results.results} - {"test"})
                 for r in comparisons(results, key) if r["against"] != "L2S"]  # fmt: skip
    tables = {
        "s2": [["criterion", "value", "threshold", "met"],
               *[[c["name"], c["shown"], c["threshold"], _yes(c["met"])] for c in computed]],
        "comparison": _rows(comparisons(results), holm_column=True),
        "progression": progression(results),
        "vs_full": vs_full(results),
        "trec": _rows(trec_rows, holm_column=False),
        "families": s1.families(results),
        "grid": s1.grid(results),
        "overlap": overlap(results),
        "fit": fit(results),
        "curves": s1.curves(results),
    }  # fmt: skip
    cells = ", ".join(f"{env} `{results.grids[env]['selected']}`" for env in results.environments)
    unchanged = s1.unchanged_starts(results)
    lines = [
        "# S2: a retrieval selector started from the corpus's geometric prior",
        "",
        "Protocol: in-domain. Scoring: exact inner product over the full corpus, hard top-k mask.",
        "The geometric selector scores `log(q_d² · Var(x_d) + ε) + (W q + b)_d`, with the corpus",
        "variance from document embeddings only and `W, b` starting at zero (at step 0 it is the",
        "`variance` method); it is trained with S1's retrieval objective and grid rules.",
        f"Selected cells: {cells}; seeds {', '.join(str(s) for s in results.seeds)}, averaged",
        "per query before every paired test.",
        "",
        f"- compare-selectors `{results.run}` (commit `{results.commit[:12]}`)",
        "",
        "## Criteria",
        "",
        *_markdown(tables["s2"]),
        "",
        "## Against the S1 selector, the variance method, and L2S on the test sets",
        "",
        "Holm over the environments, separately for each reference and summary.",
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
        "## Families (validation-best cell of each variant, seed 0; context)",
        "",
        *_markdown(tables["families"]),
        "",
        "## Where the difference comes from",
        "",
        *_markdown(tables["overlap"]),
        "",
        *_markdown(tables["fit"]),
        "",
        *_markdown(tables["curves"]),
        "",
        "## Notes",
        "",
        *[
            f"- **{env}**: {text}"
            for env, text in sorted(NOTES.items())
            if env in results.environments
        ],
        *[f"- **{env}**: {s1.unchanged_note(env, CANDIDATE)}" for env in unchanged],
        "",
        "Every grid cell is in `tables/grid.csv`.",
    ]
    return "\n".join(lines) + "\n", {name: _csv(table) for name, table in tables.items()}


def write(root: Path, output: Path) -> Path:
    text, csvs = render(load(root))
    (output / "tables").mkdir(parents=True, exist_ok=True)
    (output / "README.md").write_text(text, encoding="utf-8")
    for name, content in csvs.items():
        (output / "tables" / f"{name}.csv").write_text(content, encoding="utf-8")
    return output / "README.md"


def decision(results: s1.S1Results, adr: Path) -> GateDecision:
    computed = criteria(results)
    headline = [r for r in comparisons(results) if r["summary"] == s1.SUMMARIES[0]]
    context = {
        f"against_{label.replace(' ', '_').lower()}": ", ".join(
            f"{r['environment']} {_fmt(r['diff'], '+')} (Holm p {_fmt(r['p_holm'])})"
            for r in headline if r["against"] == label
        )
        for label in AGAINST
    }  # fmt: skip
    context["selected"] = ", ".join(
        f"{e} {results.grids[e]['selected']}" for e in results.environments
    )
    context.update({f"note_{e}": NOTES[e] for e in results.environments if e in NOTES})
    context.update(
        {f"unchanged_{e}": s1.unchanged_note(e, CANDIDATE) for e in s1.unchanged_starts(results)}
    )
    return GateDecision(
        gate="S2",
        outcome="pass" if all(c["met"] for c in computed) else "fail",
        criteria=[
            Criterion(name=c["name"], value=c["value"], threshold=c["threshold"], met=c["met"])
            for c in computed
        ],  # fmt: skip
        context=context,
        evidence=[
            results.run,
            *sorted(
                {
                    run
                    for env in results.environments
                    for run in results.sources[env][CANDIDATE].values()
                }
            ),
        ],  # fmt: skip
        adr=adr.as_posix(),
        tag=TAG,
    )


def proposed_adr(results: s1.S1Results, record: GateDecision) -> str:
    lines = [
        "# ADR-0010: Experiment S2 — a retrieval selector started from the geometric prior",
        "",
        "- **Status:** proposed",
        f"- **Date:** {results.finished}",
        "- **Issue:** F7 (`specs/008-geometric-prior`), experiment S2",
        "",
        "## Context",
        "",
        "P1 found that the label-free `variance` method approximates the full query best",
        "(MS MARCO) while learned selectors reshape the ranking toward relevance (SciFact). S2",
        "asks whether S1's retrieval selector, started from that geometric prior, gets both. This",
        "draft is generated",
        "by `hyperdime decide S2` from the result files of `reports/S2/README.md`.",
        "",
        "## Evidence",
        "",
        f"From compare-selectors `{results.run}` (commit `{results.commit[:12]}`); every value",
        "is in `reports/S2/README.md`.",
        "",
        *[
            f"- **{c['name']}**: {c['shown']}; threshold {c['threshold']}; "
            f"{'met' if c['met'] else 'not met'}."
            for c in criteria(results)
        ],  # fmt: skip
        "",
        "Context, not decisive:",
        "",
        *[f"- **{key}**: {value}" for key, value in sorted(record.context.items())],
        "",
        "## Decision",
        "",
        f"The S2 criteria read **{record.outcome}**. The decision is the researcher's: accept",
        "this reading, or amend it here with the reasons.",
        "",
        "## Consequences",
        "",
        "- If S2 passes: the geometric selector replaces the S1 selector as the one later work",
        "  builds on, and the prior makes it usable on a new corpus before any label exists.",
        "- If S2 fails: the prior and the learned correction are kept as separate methods, chosen",
        "  per environment.",
    ]
    return "\n".join(lines) + "\n"


def write_adr(results: s1.S1Results, record: GateDecision, path: Path) -> Path:
    if path.exists():
        found = s1._STATUS.search(path.read_text(encoding="utf-8"))
        if found is None or found.group(1) != "proposed":
            raise ValueError(f"{path} is no longer proposed; it is not overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(proposed_adr(results, record), encoding="utf-8")
    return path


def _mean(values: dict[str, float]) -> float:
    return sum(values.values()) / len(values)
