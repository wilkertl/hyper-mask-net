"""The S4 report and its drafted decision, generated from result files only (F9 spec).

The budget selector (one learned prior weight per budget) is compared with its aligned control
(S3's selector trained on the same 8 cells under the same aligned checkpoint rule), the S2
selector, and the ``variance`` method by a one-sided non-inferiority test with the margin fixed
before any run (SC-001), and with the aligned control by a two-sided test (SC-002). The original
S3, S1, and L2S selectors are context. Reuses the S1 and S2 report helpers. Protocol: in-domain.
Scoring: exact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from hyperdime.contracts.schemas import Criterion, GateDecision
from hyperdime.evaluation.statistics import (
    holm,
    noninferiority_test,
    paired_bootstrap_ci,
    paired_randomization_test,
)
from hyperdime.evaluation.transfer import ALPHA
from hyperdime.reports import s1, s2
from hyperdime.reports.e0 import _csv, _fmt, _markdown, _yes
from hyperdime.reports.e1 import NOTES, note

CANDIDATE = "budget"
CONTROL = "weighted-aligned"
TRACKS = (CONTROL, CANDIDATE)
# The non-inferiority margin in nDCG@10 on the mean over budgets, fixed before any S4 run
# (F9 research D7), and the one-sided level it is tested at.
MARGIN = 0.005
NONINFERIORITY_ALPHA = 0.025
AGAINST = {
    "aligned S3": CONTROL,
    "S3 selector": "weighted",
    "S2 selector": "geometric",
    "variance": "variance",
    "S1 selector": "retrieval",
    "L2S": "l2s",
}
NOT_WORSE = ("aligned S3", "S2 selector", "variance")
SECONDARY_METRICS = ("mrr@10", "recall@100")
SELECTORS = ("retrieval", "geometric", "weighted", CONTROL, CANDIDATE)
TAG = "s4-budget-prior"
DEFAULT_ADR = Path("docs/decisions/ADR-0012-s4-budget-prior.md")
REUSE_NOTE = (
    "S1-S3's results on these test sets informed the S4 proposal, so S4's reading on them is "
    "exploratory; the confirmation reruns this spec unchanged on new environments (#60)."
)


@dataclass
class S4Results:
    base: s1.S1Results
    control_grids: dict[str, dict[str, Any]]  # env -> the control grid's selection.json
    seeded: dict[str, dict[str, dict[int, dict[str, Any]]]]  # track -> env -> seed -> record
    diagnostics: dict[str, Any]  # track -> env -> seed -> budget -> values
    ablation: dict[str, Any]  # track -> env -> seed -> budget -> {kept, no-prior}


def load(root: Path) -> S4Results:
    """The latest S4 run, both tracks' grids and seeded records; refuses fewer than three seeds
    per environment and track."""
    base = s1.load(root, experiment="S4", candidate=CANDIDATE)
    selectors = root / "selectors"
    seeded: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    for track in TRACKS:
        for env in base.environments:
            runs = base.sources[env][track]
            if len(runs) < s1.MIN_SEEDS:
                raise ValueError(
                    f"S4 needs at least {s1.MIN_SEEDS} seeds per environment and track; "
                    f"{env} {track} has {len(runs)}"
                )
            seeded.setdefault(track, {})[env] = {
                int(tag.removeprefix("seed-")): _json(
                    selectors / f"{track}-seeded" / env / tag / run / "selection.json"
                )
                for tag, run in runs.items()
            }
    control_grids = {
        env: _json(
            selectors / f"{CONTROL}-grid" / env / base.sources[env][f"{CONTROL}_grid"]
            / "selection.json"
        )
        for env in base.environments
    }  # fmt: skip
    directory = root / "evaluation" / "S4" / "all" / base.run
    return S4Results(
        base,
        control_grids,
        seeded,
        _json(directory / "diagnostics.json"),
        _json(directory / "ablation.json"),
    )


def comparisons(
    results: S4Results, query_set: str = "test", metric: str = "ndcg@10"
) -> list[dict[str, Any]]:
    """The budget selector against each reference, and the aligned control against S3, per
    environment and summary; Holm over the environments for each pair and summary."""
    base = results.base
    pairs = [(CANDIDATE, label, reference) for label, reference in AGAINST.items()]
    pairs.append((CONTROL, "S3 selector", "weighted"))
    rows = []
    for candidate, label, reference in pairs:
        for which in s1.SUMMARIES:
            found = [
                {
                    **s1.compare(base, (env, query_set), which, metric, candidate, reference),
                    "candidate": candidate,
                    "against": label,
                    "metric": metric,
                }
                for env in base.environments
                if (env, query_set) in base.results
            ]
            adjusted = holm({r["environment"]: r["p"] for r in found})
            rows += [{**r, "p_holm": adjusted[r["environment"]]} for r in found]
    return rows


def noninferiority(results: S4Results) -> list[dict[str, Any]]:
    """SC-001's one-sided tests of ``mean(S4 - reference) <= -MARGIN`` on the mean nDCG@10 over
    budgets, test sets; Holm over the environments for each reference."""
    base = results.base
    rows: list[dict[str, Any]] = []
    for label in NOT_WORSE:
        found: list[dict[str, Any]] = []
        for env in base.environments:
            key = (env, "test")
            new = s1.summary(base, key, s1.method_name(CANDIDATE, env), s1.SUMMARIES[0])
            old = s1.summary(base, key, s1.method_name(AGAINST[label], env), s1.SUMMARIES[0])
            queries = sorted(new)
            a, b = [new[q] for q in queries], [old[q] for q in queries]
            low, high = paired_bootstrap_ci(a, b)
            found.append({
                "environment": env, "against": label, "diff": (sum(a) - sum(b)) / len(a),
                "ci_low": low, "ci_high": high, "p": noninferiority_test(a, b, MARGIN),
            })  # fmt: skip
        adjusted = holm({r["environment"]: r["p"] for r in found})
        for r in found:
            r["p_holm"] = adjusted[r["environment"]]
            r["met"] = r["p_holm"] <= NONINFERIORITY_ALPHA
        rows += found
    return rows


def criteria(results: S4Results) -> list[dict[str, Any]]:
    tests = noninferiority(results)
    failed = [r for r in tests if not r["met"]]
    headline = [r for r in comparisons(results) if r["summary"] == s1.SUMMARIES[0]]
    better = [
        r for r in headline
        if r["candidate"] == CANDIDATE and r["against"] == "aligned S3"
        and r["diff"] > 0 and r["p_holm"] <= ALPHA
    ]  # fmt: skip
    count = len(results.base.environments)
    return [
        {
            "name": f"SC-001 non-inferior to aligned S3, S2 selector, variance (margin {MARGIN}, "
            f"one-sided, Holm, {NONINFERIORITY_ALPHA})",
            "value": float(len(failed)),
            "shown": ", ".join(f"{r['environment']} vs {r['against']}" for r in failed)
            or "none fails",
            "threshold": "none fails",
            "met": not failed,
        },
        {
            "name": f"SC-002 better than aligned S3 (mean nDCG@10, 10%-90%, Holm, {ALPHA})",
            "value": float(len(better)),
            "shown": f"{len(better)} of {count} environments",
            "threshold": f">= 1 of {count}",
            "met": len(better) >= 1,
        },
    ]


def per_budget(results: S4Results) -> list[list[str]]:
    """S4 minus the aligned control at each budget (test sets); Holm over budgets per
    environment."""
    base = results.base
    table = [["test", "k", CONTROL, CANDIDATE, "diff", "ci_low", "ci_high", "p", "p_holm"]]
    for env in base.environments:
        key = (env, "test")
        found: dict[str, tuple[float, ...]] = {}
        for k in base.budgets:
            new = s1.per_query(base, key, s1.method_name(CANDIDATE, env), k)
            old = s1.per_query(base, key, s1.method_name(CONTROL, env), k)
            queries = sorted(new)
            a, b = [new[q] for q in queries], [old[q] for q in queries]
            found[str(k)] = (mean(b), mean(a), *paired_bootstrap_ci(a, b),
                             paired_randomization_test(a, b))  # fmt: skip
        adjusted = holm({k: v[4] for k, v in found.items()})
        for budget, (old_mean, new_mean, low, high, p) in found.items():
            table.append([
                env, budget, _fmt(old_mean), _fmt(new_mean), _fmt(new_mean - old_mean, "+"),
                _fmt(low, "+"), _fmt(high, "+"), _fmt(p), _fmt(adjusted[budget]),
            ])  # fmt: skip
    return table


def weights(results: S4Results) -> list[list[str]]:
    """Each seed's prior weights at its kept checkpoint; the control's one weight applies at
    every budget."""
    budgets = results.base.budgets
    table = [["environment", "track", "seed", "kept step", *(f"a at k={k}" for k in budgets)]]
    for env in results.base.environments:
        for track in TRACKS:
            for seed, record in sorted(results.seeded[track][env].items()):
                found = record["validation"]["prior_weights"]
                # A budget selector's weights follow its budgets, the evaluated ones.
                by_k = (
                    dict.fromkeys(budgets, found[0])
                    if len(found) == 1
                    else dict(zip(budgets, found, strict=True))
                )
                table.append([
                    env, track, str(seed), str(record["validation"]["kept_step"]),
                    *(_fmt(by_k[k]) for k in budgets),
                ])  # fmt: skip
    return table


def prior_share(results: S4Results) -> list[list[str]]:
    """Validation-only diagnostics of research D5, mean over seeds."""
    table = [["environment", "track", "budget", "prior share", "overlap with variance",
              "overlap with the correction alone", "mean prior weight"]]  # fmt: skip
    for env in results.base.environments:
        for track in TRACKS:
            seeds = results.diagnostics[track][env]
            for k in sorted({k for v in seeds.values() for k in v}, key=int):
                values = [v[k] for v in seeds.values()]
                table.append([
                    env, track, k,
                    *(_fmt(mean(v[name] for v in values)) for name in
                      ("prior_share", "overlap_variance", "overlap_correction", "prior_weight")),
                ])  # fmt: skip
    return table


def ablation(results: S4Results) -> list[list[str]]:
    """Validation nDCG@10 of the kept states and of the same states without their prior, mean
    over seeds (context, not a criterion)."""
    table = [["environment", "track", "budget", "kept", "prior removed", "removed - kept"]]
    for env in results.base.environments:
        for track in TRACKS:
            seeds = results.ablation[track][env]
            for k in sorted({k for v in seeds.values() for k in v}, key=int):
                kept = mean(v[k]["kept"] for v in seeds.values())
                removed = mean(v[k]["no-prior"] for v in seeds.values())
                table.append([env, track, k, _fmt(kept), _fmt(removed),
                              _fmt(removed - kept, "+")])  # fmt: skip
    return table


def kept_steps(results: S4Results) -> list[list[str]]:
    """Per seed: the step kept by full-corpus validation nDCG@10, the step the rerank score would
    have kept, and the kept state's guard."""
    table = [["environment", "track", "seed", "kept step", "rerank-best step",
              "subset nDCG@10 at kept step", "soft mass at final temperature"]]  # fmt: skip
    for env in results.base.environments:
        for track in TRACKS:
            for seed, record in sorted(results.seeded[track][env].items()):
                entry = record["validation"]
                kept = entry["kept_step"]
                table.append([
                    env, track, str(seed), str(kept), str(int(entry["best_step"])),
                    _fmt(entry["candidate_scores"][str(kept)]["ndcg@10"]),
                    _fmt(entry["soft_mass"]),
                ])  # fmt: skip
    return table


def _rows(rows: list[dict[str, Any]], holm_column: bool) -> list[list[str]]:
    head = ["test", "query set", "metric", "candidate", "against", "summary", "reference",
            "candidate value", "diff", "ci_low", "ci_high", "p"]  # fmt: skip
    table = [head + (["p_holm"] if holm_column else []) + ["note"]]
    for r in rows:
        table.append([
            r["environment"], r["query_set"], r["metric"], r["candidate"], r["against"],
            r["summary"], _fmt(r["old"]), _fmt(r["new"]), _fmt(r["diff"], "+"),
            _fmt(r["ci_low"], "+"), _fmt(r["ci_high"], "+"), _fmt(r["p"]),
            *([_fmt(r["p_holm"])] if holm_column else []), note(r["environment"]),
        ])  # fmt: skip
    return table


def render(results: S4Results) -> tuple[str, dict[str, str]]:
    base = results.base
    computed = criteria(results)
    comparison = [
        row
        for metric in ("ndcg@10", *SECONDARY_METRICS)
        for row in comparisons(results, "test", metric)
    ]
    trec_rows = [r for key in sorted({k[1] for k in base.results} - {"test"})
                 for r in comparisons(results, key) if r["against"] != "L2S"]  # fmt: skip
    tables = {
        "s4": [["criterion", "value", "threshold", "met"],
               *[[c["name"], c["shown"], c["threshold"], _yes(c["met"])] for c in computed]],
        "noninferiority": [
            ["test", "against", "diff", "ci_low", "ci_high", "margin", "p (one-sided)",
             "p_holm", "non-inferior"],
            *[[r["environment"], r["against"], _fmt(r["diff"], "+"), _fmt(r["ci_low"], "+"),
               _fmt(r["ci_high"], "+"), _fmt(MARGIN), _fmt(r["p"]), _fmt(r["p_holm"]),
               _yes(r["met"])] for r in noninferiority(results)],
        ],
        "comparison": _rows(comparison, holm_column=True),
        "per_budget": per_budget(results),
        "weights": weights(results),
        "prior_share": prior_share(results),
        "ablation": ablation(results),
        "kept_steps": kept_steps(results),
        "progression": s2.progression(base, SELECTORS),
        "vs_full": s2.vs_full(base, ("variance", *SELECTORS)),
        "trec": _rows(trec_rows, holm_column=False),
    }  # fmt: skip
    cells = ", ".join(
        f"{env} `{base.grids[env]['selected']}` / `{results.control_grids[env]['selected']}`"
        for env in base.environments
    )
    lines = [
        "# S4: a prior weight per budget",
        "",
        "Protocol: in-domain. Scoring: exact inner product over the full corpus, hard top-k mask.",
        "The budget selector scores `a_k · log(q_d² · Var(x_d) + ε) + (W q + b)_d` with one weight",
        "`a_k >= 0` per 10%-90% budget and the linear map shared by every budget (at step 0 it is",
        "the `variance` method at every budget). Its control, aligned S3, is S3's selector with",
        "one weight, trained on the same 8 cells. Both keep the checkpoint with the best",
        "full-corpus validation nDCG@10 (step 0 included) on a fixed validation subset.",
        f"Selected cells (budget / aligned S3): {cells}; seeds",
        f"{', '.join(str(s) for s in base.seeds)}, averaged per query before every paired test.",
        "",
        f"- compare-selectors `{base.run}` (commit `{base.commit[:12]}`)",
        f"- Non-inferiority margin δ = {MARGIN} nDCG@10 (mean over budgets), fixed before any",
        "  S4 run.",
        f"- {REUSE_NOTE}",
        "",
        "## Criteria",
        "",
        *_markdown(tables["s4"]),
        "",
        "## Non-inferiority (SC-001)",
        "",
        f"One-sided sign-flip test of `mean(S4 − reference) <= −{MARGIN}`, Holm over the",
        f"environments for each reference, met at p_holm <= {NONINFERIORITY_ALPHA}.",
        "",
        *_markdown(tables["noninferiority"]),
        "",
        "## Against every reference on the test sets",
        "",
        "Two-sided tests, Holm over the environments for each pair, metric, and summary; the last",
        "pair is the aligned control against S3. MRR@10 and Recall@100 are secondary.",
        "",
        *_markdown(tables["comparison"]),
        "",
        "## By budget: S4 against aligned S3",
        "",
        "Holm over the budgets of each environment (secondary).",
        "",
        *_markdown(tables["per_budget"]),
        "",
        "## The learned prior weights",
        "",
        "At each seed's kept checkpoint; the control's one weight applies at every budget.",
        "",
        *_markdown(tables["weights"]),
        "",
        "## How much the prior still decides (validation queries)",
        "",
        "Prior share `σ(a_k π) / (σ(a_k π) + σ(W q + b))` per query, unchanged when `a`, `W`, `b`",
        "are scaled together; top-k overlaps with `variance` and with the correction alone.",
        "",
        *_markdown(tables["prior_share"]),
        "",
        "Validation nDCG@10 with every prior weight set to 0 (context, not a criterion):",
        "",
        *_markdown(tables["ablation"]),
        "",
        "## Kept checkpoints",
        "",
        *_markdown(tables["kept_steps"]),
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
        "## Notes",
        "",
        *[f"- **{e}**: {text}" for e, text in sorted(NOTES.items()) if e in base.environments],
    ]
    return "\n".join(lines) + "\n", {name: _csv(table) for name, table in tables.items()}


def write(root: Path, output: Path) -> Path:
    text, csvs = render(load(root))
    (output / "tables").mkdir(parents=True, exist_ok=True)
    (output / "README.md").write_text(text, encoding="utf-8")
    for name, content in csvs.items():
        (output / "tables" / f"{name}.csv").write_text(content, encoding="utf-8")
    return output / "README.md"


def decision(results: S4Results, adr: Path) -> GateDecision:
    base = results.base
    computed = criteria(results)
    context = {
        f"noninferior_vs_{label.replace(' ', '_').lower()}": ", ".join(
            f"{r['environment']} {_fmt(r['diff'], '+')} (Holm p {_fmt(r['p_holm'])})"
            for r in noninferiority(results) if r["against"] == label
        )
        for label in NOT_WORSE
    }  # fmt: skip
    context["against_aligned_s3"] = ", ".join(
        f"{r['environment']} {_fmt(r['diff'], '+')} (Holm p {_fmt(r['p_holm'])})"
        for r in comparisons(results)
        if r["summary"] == s1.SUMMARIES[0]
        and r["candidate"] == CANDIDATE
        and r["against"] == "aligned S3"
    )
    context["selected"] = ", ".join(
        f"{e} {base.grids[e]['selected']} / {results.control_grids[e]['selected']}"
        for e in base.environments
    )
    context["prior_weights"] = "; ".join(
        f"{row[0]} {row[1]} s{row[2]}: {' '.join(row[4:])}" for row in weights(results)[1:]
    )
    context["prior_share"] = "; ".join(
        f"{row[0]} {row[1]} k={row[2]} {row[3]}" for row in prior_share(results)[1:]
    )
    context["ablation"] = "; ".join(
        f"{row[0]} {row[1]} k={row[2]} {row[5]}" for row in ablation(results)[1:]
    )
    context["kept_steps"] = "; ".join(
        f"{row[0]} {row[1]} s{row[2]} {row[3]}" for row in kept_steps(results)[1:]
    )
    context["margin"] = f"{MARGIN} nDCG@10, fixed before any S4 run"
    context["test_set_reuse"] = REUSE_NOTE
    context.update({f"note_{e}": NOTES[e] for e in base.environments if e in NOTES})
    return GateDecision(
        gate="S4",
        outcome="pass" if all(c["met"] for c in computed) else "fail",
        criteria=[
            Criterion(name=c["name"], value=c["value"], threshold=c["threshold"], met=c["met"])
            for c in computed
        ],
        context=context,
        evidence=[
            base.run,
            *sorted(
                {
                    run
                    for e in base.environments
                    for t in TRACKS
                    for run in base.sources[e][t].values()
                }
            ),
        ],  # fmt: skip
        adr=adr.as_posix(),
        tag=TAG,
    )


def proposed_adr(results: S4Results, record: GateDecision) -> str:
    base = results.base
    lines = [
        "# ADR-0012: Experiment S4 — a prior weight per budget",
        "",
        "- **Status:** proposed",
        f"- **Date:** {base.finished}",
        "- **Issue:** F9 (`specs/010-budget-prior`), experiment S4",
        "",
        "## Context",
        "",
        "S3 (ADR-0011) learned one weight on the geometric prior and stayed below S2 at",
        "NFCorpus's small budgets, where the prior's value changes with the budget. S4 gives each",
        "of the nine budgets its own weight, and checkpoints both S4 and its aligned S3 control",
        "by full-corpus validation nDCG@10. This draft is generated by `hyperdime decide S4` from",
        "the result files of `reports/S4/README.md`.",
        "",
        "## Evidence",
        "",
        f"From compare-selectors `{base.run}` (commit `{base.commit[:12]}`); every value is in",
        "`reports/S4/README.md`.",
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
        f"The S4 criteria read **{record.outcome}**. The decision is the researcher's: accept",
        "this reading, or amend it here with the reasons.",
        "",
        "## Consequences",
        "",
        "- If S4 passes: the budget selector is the one later work builds on, confirmed by",
        "  rerunning this spec unchanged on the added environments (#60).",
        "- If S4 fails: the next candidates are the follow-ups of F9 research D9, in order:",
        "  straight-through hard masks, then masked-query negatives on MS MARCO.",
    ]
    return "\n".join(lines) + "\n"


def write_adr(results: S4Results, record: GateDecision, path: Path) -> Path:
    if path.exists():
        found = s1._STATUS.search(path.read_text(encoding="utf-8"))
        if found is None or found.group(1) != "proposed":
            raise ValueError(f"{path} is no longer proposed; it is not overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(proposed_adr(results, record), encoding="utf-8")
    return path


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
