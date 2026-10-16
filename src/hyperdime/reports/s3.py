"""The S3 report and its drafted decision, generated from result files only (F8 spec).

The weighted selector (S2's selector with one learned weight on the geometric prior) is compared
with the S1 selector, the S2 selector, and the ``variance`` method (SC-001: never worse than any of
them) and with the S1 selector (SC-002: better in most environments). Reuses the S1 and S2 report
helpers. Protocol: in-domain. Scoring: exact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hyperdime.contracts.schemas import Criterion, GateDecision
from hyperdime.evaluation.transfer import ALPHA
from hyperdime.reports import s1, s2
from hyperdime.reports.e0 import _csv, _fmt, _markdown, _yes
from hyperdime.reports.e1 import NOTES

CANDIDATE = "weighted"
AGAINST = {
    "S1 selector": "retrieval",
    "S2 selector": "geometric",
    "variance": "variance",
    "L2S": "l2s",
}
NOT_WORSE = ("S1 selector", "S2 selector", "variance")
TAG = "s3-prior-weight"
DEFAULT_ADR = Path("docs/decisions/ADR-0011-s3-prior-weight.md")


def load(root: Path) -> s1.S1Results:
    return s1.load(root, experiment="S3", candidate=CANDIDATE)


def comparisons(results: s1.S1Results, query_set: str = "test") -> list[dict[str, Any]]:
    return s2.comparisons(results, query_set, candidate=CANDIDATE, against=AGAINST)


def criteria(results: s1.S1Results) -> list[dict[str, Any]]:
    rows = [r for r in comparisons(results) if r["summary"] == s1.SUMMARIES[0]]
    significant = [r for r in rows if r["p_holm"] <= ALPHA]
    worse = [r for r in significant if r["against"] in NOT_WORSE and r["diff"] < 0]
    better = [r for r in significant if r["against"] == "S1 selector" and r["diff"] > 0]
    count = len(results.environments)
    required = (2 * count + 2) // 3
    return [
        {
            "name": "SC-001 worse than none of the S1 selector, S2 selector, variance (Holm, 0.05)",
            "value": float(len(worse)),
            "shown": ", ".join(f"{r['environment']} vs {r['against']}" for r in worse) or "none",
            "threshold": "none",
            "met": not worse,
        },
        {
            "name": "SC-002 better than the S1 selector (mean nDCG@10, 10%-90%, Holm, 0.05)",
            "value": float(len(better)),
            "shown": f"{len(better)} of {count} environments",
            "threshold": f">= {required} of {count}",
            "met": len(better) >= required,
        },
    ]


def weights(results: s1.S1Results) -> list[list[str]]:
    """The learned prior weight at each seed's kept (best validation) checkpoint."""
    table = [["environment", "seed", "best step", "prior weight at best step", "final weight"]]
    for env in results.environments:
        for seed, history in sorted(results.histories[env].items()):
            top = max(history, key=lambda h: h["val_rerank_ndcg10"])
            table.append([
                env, str(seed), str(int(top["step"])), _fmt(top["prior_weight"]),
                _fmt(history[-1]["prior_weight"]),
            ])  # fmt: skip
    return table


def render(results: s1.S1Results) -> tuple[str, dict[str, str]]:
    computed = criteria(results)
    trec_rows = [r for key in sorted({k[1] for k in results.results} - {"test"})
                 for r in comparisons(results, key) if r["against"] != "L2S"]  # fmt: skip
    selectors = ("retrieval", "geometric", CANDIDATE)
    tables = {
        "s3": [["criterion", "value", "threshold", "met"],
               *[[c["name"], c["shown"], c["threshold"], _yes(c["met"])] for c in computed]],
        "comparison": s2._rows(comparisons(results), holm_column=True, candidate=CANDIDATE),
        "weights": weights(results),
        "progression": s2.progression(results, selectors),
        "vs_full": s2.vs_full(results, ("variance", *selectors)),
        "trec": s2._rows(trec_rows, holm_column=False, candidate=CANDIDATE),
        "families": s1.families(results),
        "grid": s1.grid(results),
        "overlap": s2.overlap(results, CANDIDATE),
        "fit": s2.fit(results, ("l2s", *selectors)),
        "curves": s1.curves(results),
    }  # fmt: skip
    cells = ", ".join(f"{env} `{results.grids[env]['selected']}`" for env in results.environments)
    unchanged = s1.unchanged_starts(results)
    lines = [
        "# S3: a selector that learns how much to trust the geometric prior",
        "",
        "Protocol: in-domain. Scoring: exact inner product over the full corpus, hard top-k mask.",
        "The weighted selector scores `a · log(q_d² · Var(x_d) + ε) + (W q + b)_d`: S2's selector",
        "with one learned scalar `a` starting at 1 (at step 0 it is the `variance` method); it is",
        "trained with S1's retrieval objective and S2's grid rules.",
        f"Selected cells: {cells}; seeds {', '.join(str(s) for s in results.seeds)}, averaged",
        "per query before every paired test.",
        "",
        f"- compare-selectors `{results.run}` (commit `{results.commit[:12]}`)",
        "",
        "## Criteria",
        "",
        *_markdown(tables["s3"]),
        "",
        "## Against the S1 and S2 selectors, the variance method, and L2S on the test sets",
        "",
        "Holm over the environments, separately for each reference and summary.",
        "",
        *_markdown(tables["comparison"]),
        "",
        "## The learned prior weight",
        "",
        *_markdown(tables["weights"]),
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
        *[f"- **{e}**: {text}" for e, text in sorted(NOTES.items()) if e in results.environments],
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
    envs = results.environments
    context["selected"] = ", ".join(f"{e} {results.grids[e]['selected']}" for e in envs)
    context["prior_weights"] = ", ".join(
        f"{row[0]} s{row[1]} {row[3]}" for row in weights(results)[1:]
    )
    context.update({f"note_{e}": NOTES[e] for e in envs if e in NOTES})
    for env in s1.unchanged_starts(results):
        context[f"unchanged_{env}"] = s1.unchanged_note(env, CANDIDATE)
    return GateDecision(
        gate="S3",
        outcome="pass" if all(c["met"] for c in computed) else "fail",
        criteria=[
            Criterion(name=c["name"], value=c["value"], threshold=c["threshold"], met=c["met"])
            for c in computed
        ],
        context=context,
        evidence=[
            results.run,
            *sorted({run for e in envs for run in results.sources[e][CANDIDATE].values()}),
        ],  # fmt: skip
        adr=adr.as_posix(),
        tag=TAG,
    )


def proposed_adr(results: s1.S1Results, record: GateDecision) -> str:
    lines = [
        "# ADR-0011: Experiment S3 — a learned weight on the geometric prior",
        "",
        "- **Status:** proposed",
        f"- **Date:** {results.finished}",
        "- **Issue:** #59, F8 (`specs/009-prior-weight`), experiment S3",
        "",
        "## Context",
        "",
        "S2 (ADR-0010) showed the geometric prior helps where approximating the full query",
        "matters and hurts where the ranking must be reshaped. S3 asks whether one learned weight",
        "on the prior lets a single selector be never worse than the S1 selector, the S2 selector,",
        "and the `variance` method. This draft is generated by `hyperdime decide S3` from the",
        "result files of `reports/S3/README.md`.",
        "",
        "## Evidence",
        "",
        f"From compare-selectors `{results.run}` (commit `{results.commit[:12]}`); every value",
        "is in `reports/S3/README.md`.",
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
        f"The S3 criteria read **{record.outcome}**. The decision is the researcher's: accept",
        "this reading, or amend it here with the reasons.",
        "",
        "## Consequences",
        "",
        "- If S3 passes: the weighted selector is the one later work builds on (one method for",
        "  every environment), and it is rerun on the added environments (#60).",
        "- If S3 fails: the next candidate is a per-budget weight, since the prior's value",
        "  changes with the budget (S2).",
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
