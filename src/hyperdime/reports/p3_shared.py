"""Report of the shared-mask follow-up to P3, generated from one run; no number is typed here."""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from hyperdime.artifacts import latest_artifact
from hyperdime.contracts.manifests import read_manifest
from hyperdime.evaluation.representation_controls import row_spearman
from hyperdime.evaluation.runs import ratio_budgets
from hyperdime.reports.e0 import _markdown
from hyperdime.reports.p3_svg import Series, lines

CHECKPOINTS = tuple(ratio_budgets())  # the shared decile grid of every P3 stage (ADR-0017)
Z95 = 1.96
TINY_COHORT = 30
COMMON_CONSTRUCT_SIZES = (1, 2, 5, 10, 20)
TRANSFER_LABELS = {
    "held_mean_norm": "held-out positives, construction-only mask",
    "construct_mean_norm": "construction positives, same mask",
    "privileged_mean_norm": "held-out positives, all-positive mask (privileged)",
    "reference_mean_norm": "held-out positives, mask of unjudged documents",
}


def _num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _small(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2e}"


def _ci(curve: dict[str, Any], at: int) -> str:
    mean, se = curve["mean"][at], curve["se"][at]
    if mean is None:
        return "n/a"
    if se is None:
        return f"{mean:.4f} (n={curve['queries'][at]})"
    low, high = mean - Z95 * se, mean + Z95 * se
    return f"{mean:.4f} ({low:.4f} to {high:.4f})"


def _spread(curve: dict[str, Any], at: int) -> str:
    if curve["mean"][at] is None:
        return "n/a"
    return (f"{curve['mean'][at]:.4f} (p10 {_num(curve['p10'][at], 4)}, "
            f"p50 {_num(curve['p50'][at], 4)}, p90 {_num(curve['p90'][at], 4)})")  # fmt: skip


def _shared_table(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "positives", "k", "queries", "mean regret / mass (95% over queries)",
              "worst positive / mass", "p90 positive / mass", "spread within query",
              "mean raw regret", "uniform random mask / mass", "overlap with own mask",
              "worst positive's share of regret", "worst fifth's share"]]  # fmt: skip
    for env, s in summaries.items():
        for subset in s["shared"]["mean_norm"]:
            if not s["coverage"]["strata"][subset]:
                continue
            for k in CHECKPOINTS:
                at = k - 1
                row = {name: s["shared"][name][subset] for name in s["shared"]}
                table.append([
                    env, subset.removeprefix("positives "), str(k), str(row["mean_norm"]["n"]),
                    _ci(row["mean_norm"], at), _spread(row["max_norm"], at),
                    _num(row["p90_norm"]["mean"][at], 4), _num(row["std_norm"]["mean"][at], 4),
                    _num(row["mean_raw"]["mean"][at], 4),
                    _num(row["random_mean_norm"]["mean"][at], 4),
                    _num(row["overlap_mean"]["mean"][at]), _num(row["worst_share"]["mean"][at]),
                    _num(row["top20_share"]["mean"][at]),
                ])  # fmt: skip
    return table


def _fixed_table(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "documents in the group", "queries", "k", "mean regret / mass",
              "worst of the group / mass",
              "worst's share of regret (even would be 1/m)"]]  # fmt: skip
    for env, s in summaries.items():
        for m, row in s["fixed_count"].items():
            for k in CHECKPOINTS:
                table.append([
                    env, m, str(row["queries"]), str(k), _ci(row["mean_norm"], k - 1),
                    _ci(row["max_norm"], k - 1), _num(row["worst_share"]["mean"][k - 1]),
                ])  # fmt: skip
    return table


def _transfer_table(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "construct-evaluate documents", "queries", "partitions", "k",
              "held-out, construction mask", "construction side, same mask",
              "held-out minus construction", "held-out, all-positive mask (privileged)",
              "held-out, mask of unjudged documents", "held-out minus unjudged",
              "worst held-out", "overlap with held-out", "repeat range",
              "between-partition SD within a query"]]  # fmt: skip
    for env, s in summaries.items():
        for name, c in s["transfer"].items():
            if not c["queries"]:
                continue
            for k in CHECKPOINTS:
                at = k - 1
                band = c["repeat_band"]
                table.append([
                    env, name, str(c["queries"]), str(c["partitions_evaluated"]), str(k),
                    _ci(c["held_mean_norm"], at), _ci(c["construct_mean_norm"], at),
                    _ci(c["held_minus_construct"], at), _ci(c["privileged_mean_norm"], at),
                    _ci(c["reference_mean_norm"], at), _ci(c["held_minus_reference"], at),
                    _ci(c["held_max_norm"], at), _num(c["held_overlap"]["mean"][at]),
                    f"{_num(band['min'][at], 4)} to {_num(band['max'][at], 4)}",
                    _num(c["between_partition_sd_mean"][at], 4),
                ])  # fmt: skip
    return table


def _half_strata_table(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "positives", "k", "queries", "held-out", "construction side",
              "mask of unjudged documents"]]  # fmt: skip
    for env, s in summaries.items():
        for subset, row in s["transfer"]["half"].get("strata", {}).items():
            if not row["held_mean_norm"]["n"]:
                continue
            for k in CHECKPOINTS:
                at = k - 1
                table.append([
                    env, subset.removeprefix("positives "), str(k), str(row["held_mean_norm"]["n"]),
                    _ci(row["held_mean_norm"], at), _ci(row["construct_mean_norm"], at),
                    _ci(row["reference_mean_norm"], at),
                ])  # fmt: skip
    return table


def _overlap_table(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "k", "Spearman across queries, overlap vs regret",
              "mean Pearson within a query", "queries for it",
              "regret / mass at overlap under 0.25 (pairs)", "0.25 to 0.5 (pairs)",
              "0.5 to 0.75 (pairs)", "0.75 and over (pairs)"]]  # fmt: skip
    for env, s in summaries.items():
        o = s["overlap"]
        for k in CHECKPOINTS:
            at = k - 1
            bins = [
                f"{_num(o['regret_by_overlap_bin'][b][at], 4)} "
                f"({int(o['pairs_by_overlap_bin'][b][at])})"
                for b in range(4)
            ]
            table.append([
                env, str(k), _num(o["between_query_spearman"][at]),
                _num(o["within_query_pearson_mean"][at]),
                str(int(o["within_query_pearson_queries"][at])), *bins,
            ])  # fmt: skip
    return table


def cohort_positions(counts: np.ndarray, cell_minimum: int, cohort_minimum: int) -> np.ndarray:
    """Rows of a cell's stored arrays that belong to a common cohort. A cell stores only queries
    with at least ``cell_minimum`` positives, in query order; the cohort needs ``cohort_minimum``.
    """
    return np.flatnonzero(counts[counts >= cell_minimum] >= cohort_minimum)


def _query_counts(run: Path, inputs: dict[str, Any], env: str) -> np.ndarray | None:
    """Each selected query's positive count, from the checked P3 run that this run points to."""
    if inputs[env].get("canonical_p3_run") is None:
        return None
    source = run.parents[1].parent / "P3" / "all" / inputs[env]["canonical_p3_run"]
    path = source / env / "pairs.jsonl"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        found = Counter(json.loads(line)["query_id"] for line in handle)
    ids = json.loads((run / env / "selected-queries.json").read_text(encoding="utf-8"))
    return np.array([found[qid] for qid in ids], dtype=np.int64)


def _paired_mean(values: np.ndarray) -> str:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return "n/a"
    mean, se = float(finite.mean()), float(finite.std(ddof=1)) / finite.size**0.5
    return f"{mean:.4f} ({mean - Z95 * se:.4f} to {mean + Z95 * se:.4f})"


def _common_cohort(
    run: Path, inputs: dict[str, Any], summaries: dict[str, Any]
) -> tuple[list[list[str]], dict[str, list[list[str]]]]:
    """Cells compared on the same queries: each cohort is every query eligible for its largest
    cell, and each smaller cell is restricted to those queries from its stored per-query arrays."""
    table = [["environment", "cohort: positives at least", "queries", "construct-evaluate", "k",
              "held-out", "construction side", "held-out minus 1-1, same queries"]]  # fmt: skip
    curves: dict[str, list[list[str]]] = {}
    for env, s in summaries.items():
        counts = _query_counts(run, inputs, env)
        if counts is None:
            continue
        series: list[tuple[str, np.ndarray]] = []
        for largest in COMMON_CONSTRUCT_SIZES[1:]:
            cohort = 2 * largest
            queries = int((counts >= cohort).sum())
            if not queries:
                continue
            base = None
            for a in (size for size in COMMON_CONSTRUCT_SIZES if size <= largest):
                rows = cohort_positions(counts, 2 * a, cohort)
                stem = run / env / f"transfer{a}-{a}"
                held = np.asarray(np.load(f"{stem}-held_mean_norm.npy", mmap_mode="r")[rows],
                                  dtype=np.float64)  # fmt: skip
                built = np.asarray(
                    np.load(f"{stem}-construct_mean_norm.npy", mmap_mode="r")[rows],
                    dtype=np.float64,
                )
                base = held if base is None else base
                change = held - base
                for k in CHECKPOINTS:
                    at = k - 1
                    table.append([
                        env, str(cohort), str(queries), f"{a}-{a}", str(k),
                        _paired_mean(held[:, at]), _paired_mean(built[:, at]),
                        _paired_mean(change[:, at]) if a > 1 else "reference",
                    ])  # fmt: skip
                label = f"positives>={cohort} n={queries} {a}-{a}"
                series.append((f"{label} held-out mean", held.mean(axis=0)))
                series.append((f"{label} construction mean", built.mean(axis=0)))
                if a > 1:
                    series.append((f"{label} minus 1-1 mean", change.mean(axis=0)))
        if series:
            curve = [["k"] + [name for name, _ in series]]
            for at in range(len(s["k"])):
                curve.append([str(at + 1)] + [f"{values[at]:.6f}" for _, values in series])
            curves[f"transfer-common-cohort-curves-{env}"] = curve
    return table, curves


def _context_table(run: Path, summaries: dict[str, Any]) -> list[list[str]]:
    """Spearman across queries with the cosine among a query's positives, from P3's own record."""
    table = [["environment", "k", "queries", "mean regret of the shared mask",
              "worst regret of the shared mask", "one positive's mask on another"]]  # fmt: skip
    for env in summaries:
        source = run / env / "positive-cosine.npy"
        if not source.exists():
            continue
        cosine = np.load(source)
        for k in CHECKPOINTS:
            at = k - 1
            found = [
                row_spearman(cosine[None], np.load(run / env / name, mmap_mode="r")[:, at][None])[0]
                for name in ("shared-mean_norm.npy", "shared-max_norm.npy",
                             "transfer1-1-held_mean_norm.npy")
            ]  # fmt: skip
            table.append([env, str(k), str(cosine.size), *[f"{value:.3f}" for value in found]])
    return table


def _weighting_table(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "k", "query-balanced mean regret / mass",
              "pair-weighted mean regret / mass", "query-balanced raw regret",
              "pair-weighted raw regret"]]  # fmt: skip
    for env, s in summaries.items():
        pair, shared = s["pair_weighted"], s["shared"]
        for k in CHECKPOINTS:
            at = k - 1
            table.append([
                env, str(k), _num(shared["mean_norm"]["multi"]["mean"][at], 4),
                _num(pair["mean_norm"][at], 4), _num(shared["mean_raw"]["multi"]["mean"][at], 4),
                _num(pair["mean_raw"][at], 4),
            ])  # fmt: skip
    return table


def _deciles(curve: list[float | None], digits: int = 4) -> str:
    """One curve's values on the decile grid, every point reported alike."""
    return ", ".join(
        f"{k}: {'n/a' if curve[k - 1] is None else format(curve[k - 1], f'.{digits}f')}"
        for k in CHECKPOINTS
    )


def _findings(summaries: dict[str, Any]) -> list[str]:
    """What the curves show on the decile grid, with numbers read from them; every k is in the
    tables."""
    lines_: list[str] = []
    for env, s in summaries.items():
        shared = s["shared"]
        mean = shared["mean_norm"]["multi"]["mean"]
        peak = max(range(len(mean)), key=lambda i: mean[i] if mean[i] is not None else -1.0)
        lines_.append(
            f"- **{env}, one shared mask.** Mean regret / mass peaks at {mean[peak]:.4f} at "
            f"k = {peak + 1}. On the decile grid: {_deciles(mean)}. A uniform random mask: "
            f"{_deciles(shared['random_mean_norm']['multi']['mean'])}. Mean overlap: "
            f"{_deciles(shared['overlap_mean']['multi']['mean'], 3)}."
        )
        for m, r in s["fixed_count"].items():
            if r["queries"]:
                lines_.append(
                    f"- **{env}, worst of {m} positives ({r['queries']} queries).** Worst regret "
                    f"/ mass: {_deciles(r['max_norm']['mean'])}. Its share of the group's "
                    f"regret, where an even split would be 1/m: "
                    f"{_deciles(r['worst_share']['mean'], 3)}. This measures the single worst "
                    "positive, and it does not describe the rest of the regret distribution."
                )
        cell = s["transfer"]["1-1"]
        lines_.append(
            f"- **{env}, transfer.** One positive's mask on another: "
            f"{_deciles(cell['held_mean_norm']['mean'])}. On the same queries, the privileged "
            f"all-positive mask: {_deciles(cell['privileged_mean_norm']['mean'])}. A mask of one "
            f"unjudged document: {_deciles(cell['reference_mean_norm']['mean'])}."
        )
        tiny = [
            f"{name} ({c['queries']} queries)"
            for name, c in s["transfer"].items()
            if 0 < c["queries"] < TINY_COHORT
        ] + [
            f"fixed {m} ({r['queries']} queries)"
            for m, r in s["fixed_count"].items()
            if 0 < r["queries"] < TINY_COHORT
        ]
        if tiny:
            lines_.append(f"- **{env}, small cohorts.** Read with care: {', '.join(tiny)}.")
    return lines_


def _coverage_table(inputs: dict[str, Any], summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "F2 cap on train and validation queries", "training queries",
              "with a positive", "eligible multi-positive", "selected", "pairs", "largest group",
              "2-5", "6-20", "21+", "grades", "capped"]]  # fmt: skip
    for env, s in summaries.items():
        c = c_ = s["coverage"]
        strata = c_["strata"]
        table.append([
            env, str(inputs[env]["train_query_cap"] or "none"), str(c["training_queries"]),
            str(c["queries_with_positive"]), str(c["eligible_multi_positive"]), str(c["queries"]),
            str(c["pairs"]), str(c["largest_group"]), str(strata.get("positives 2-5", 0)),
            str(strata.get("positives 6-20", 0)), str(strata.get("positives 21+", 0)),
            ", ".join(str(g) for g in inputs[env]["grades"]), "yes" if c["capped"] else "no",
        ])  # fmt: skip
    return table


def _check_table(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "lowest raw regret", "largest regret at k = 1024",
              "largest mean-identity error", "documents without mass",
              "orderings differing from P3 (compared)", "largest difference from P3's mean regret",
              "largest difference from P3's worst regret"]]  # fmt: skip
    for env, s in summaries.items():
        c = s["checks"]
        canonical = c.get("canonical") or {}
        table.append([
            env, f"{c['lowest_raw_regret']:.2e}", f"{c['largest_full_k_regret']:.2e}",
            f"{c['largest_mean_identity_error']:.2e}", str(c["zero_mass_documents"]),
            f"{c['orderings_differing_from_canonical']} ({c['documents_compared_with_canonical']})",
            _small(canonical.get("largest_mean_regret_difference")),
            _small(canonical.get("largest_worst_regret_difference")),
        ])  # fmt: skip
    return table


def _source_table(inputs: dict[str, Any], summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "dataset", "embeddings", "documents sha256", "queries sha256",
              "train judgments sha256", "model revision", "P3 run checked against"]]  # fmt: skip
    for env in summaries:
        i = inputs[env]
        table.append([
            env, i["dataset"], i["embeddings"], i["documents_sha256"][:16],
            i["queries_sha256"][:16], i["qrels_train_sha256"][:16], i["model_revision"][:12],
            str(i["canonical_p3_run"]),
        ])  # fmt: skip
    return table


def _cost_table(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "device", "total, seconds", "load and verify", "per-query pass",
              "summaries", "read", "transfer", "synchronized compute", "write", "peak device MiB",
              "peak RSS MiB (process so far)", "artifact MiB"]]  # fmt: skip
    for env, s in summaries.items():
        c, phase = s["cost"], s["cost"]["phase_seconds"]
        table.append([
            env, c["device_memory"]["name"], _num(c["total_seconds"], 1),
            _num(c["load_seconds"], 1), _num(c["pass_seconds"], 1), _num(c["summary_seconds"], 1),
            *[_num(phase.get(name, 0.0), 1) for name in ("read", "transfer", "compute", "write")],
            _num(c["device_memory"].get("peak_allocated_mib"), 0), _num(c["peak_rss_mib"], 0),
            _num(c["artifact_bytes"] / 2**20, 1),
        ])  # fmt: skip
    return table


def _curves(s: dict[str, Any]) -> list[list[str]]:
    """One row per ``k``: the full curves, with standard errors and eligible queries."""
    series: list[tuple[str, list[Any]]] = []
    for name, row in s["shared"].items():
        curve = row["multi"]
        series += [(f"shared {name} mean", curve["mean"]), (f"shared {name} se", curve["se"]),
                   (f"shared {name} p90", curve["p90"]),
                   (f"shared {name} queries", curve["queries"])]  # fmt: skip
    for name, values in s["pair_weighted"].items():
        series.append((f"pair-weighted {name} mean", values))
    for m, row in s["fixed_count"].items():
        for name in ("mean_norm", "max_norm", "worst_share"):
            series += [(f"fixed {m} {name} mean", row[name]["mean"]),
                       (f"fixed {m} {name} se", row[name]["se"])]  # fmt: skip
    for cell, row in s["transfer"].items():
        for name in ("held_mean_norm", "construct_mean_norm", "privileged_mean_norm",
                     "reference_mean_norm", "held_minus_construct", "held_minus_reference",
                     "held_max_norm", "held_overlap"):  # fmt: skip
            series += [(f"transfer {cell} {name} mean", row[name]["mean"]),
                       (f"transfer {cell} {name} se", row[name]["se"])]  # fmt: skip
        series += [(f"transfer {cell} repeats min", row["repeat_band"]["min"]),
                   (f"transfer {cell} repeats max", row["repeat_band"]["max"]),
                   (f"transfer {cell} queries", row["held_mean_norm"]["queries"])]  # fmt: skip
    table = [["k"] + [name for name, _ in series]]
    for at, k in enumerate(s["k"]):
        table.append([str(k)] + [_num(values[at], 6) for _, values in series])
    return table


def _figures(summaries: dict[str, Any]) -> dict[str, str]:
    found = {}
    for env, s in summaries.items():
        k = s["k"]
        row = s["shared"]
        found[f"shared-{env}"] = lines(
            f"What one shared mask costs its positives ({env})",
            "Regret / sum|c|, multi-positive queries, query-balanced. A compromise in similarity, "
            "not a ranking measure. All curves reach 0 at k = 1024.",
            "Retained dimensions k", "Regret / mass",
            [Series("Mean positive", k, row["mean_norm"]["multi"]["mean"]),
             Series("Worst positive", k, row["max_norm"]["multi"]["mean"]),
             Series("Uniform random mask", k, row["random_mean_norm"]["multi"]["mean"],
                    neutral=True)],
        )  # fmt: skip
        fixed = [
            Series(f"Worst of {m}", k, r["max_norm"]["mean"])
            for m, r in s["fixed_count"].items()
            if r["queries"]
        ]
        if fixed:
            found[f"fixed-{env}"] = lines(
                f"The worst positive at a fixed group size ({env})",
                "Mask of a drawn group scored on that group, so the count is equal on every "
                "query that enters a curve.",
                "Retained dimensions k", "Regret / mass", fixed,
            )  # fmt: skip
        cell = s["transfer"]["1-1"]
        if cell["queries"]:
            found[f"transfer-{env}"] = lines(
                f"Does one positive's mask transfer to another? ({env})",
                "One construction document, one held-out document of the same training query. "
                "Band: repeated partitions.",
                "Retained dimensions k", "Regret / mass",
                [Series("Held-out positive", k, cell["held_mean_norm"]["mean"],
                        cell["repeat_band"]["min"], cell["repeat_band"]["max"]),
                 Series("Mask of an unjudged document", k, cell["reference_mean_norm"]["mean"]),
                 Series("All-positive mask (privileged)", k,
                        cell["privileged_mean_norm"]["mean"])],
            )  # fmt: skip
    return found


def _csv(table: list[list[str]]) -> str:
    out = io.StringIO()
    csv.writer(out, lineterminator="\n").writerows(table)
    return out.getvalue()


def _coverage_note(summaries: dict[str, Any]) -> list[str]:
    return [
        f"- **{env}.** {s['coverage']['queries']} multi-positive queries, "
        f"{s['coverage']['pairs']} relevant pairs; "
        + ", ".join(f"{r['queries']} queries at {m} documents" for m, r in s["fixed_count"].items())
        + "."  # fmt: skip
        for env, s in summaries.items()
    ]


def write(root: Path, target: Path) -> Path:
    """Write README, tables, and figures for the latest run; the same run gives the same bytes."""
    run = latest_artifact(root, "evaluation/P3-shared-mask", "all")
    inputs = json.loads((run / "inputs.json").read_text(encoding="utf-8"))
    summaries = {
        env: json.loads((run / env / "summary.json").read_text(encoding="utf-8"))
        for env in inputs["environments"]
    }
    manifest = read_manifest(run)
    tables = {
        "shared": _shared_table(summaries),
        "fixed-count": _fixed_table(summaries),
        "transfer": _transfer_table(summaries),
        "transfer-half-strata": _half_strata_table(summaries),
        "overlap": _overlap_table(summaries),
        "document-similarity": _context_table(run, summaries),
        "weighting": _weighting_table(summaries),
        "coverage": _coverage_table(inputs, summaries),
        "checks": _check_table(summaries),
        "sources": _source_table(inputs, summaries),
        "cost": _cost_table(summaries),
    }
    tables["transfer-common-cohort"], cohort_curves = _common_cohort(run, inputs, summaries)
    tables.update(cohort_curves)
    for env, s in summaries.items():
        tables[f"curves-{env}"] = _curves(s)
    figures = _figures(summaries)
    (target / "tables").mkdir(parents=True, exist_ok=True)
    (target / "figures").mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        (target / "tables" / f"{name}.csv").write_text(_csv(table), encoding="utf-8")
    for name, svg in figures.items():
        (target / "figures" / f"{name}.svg").write_text(svg, encoding="utf-8")

    def show(stem: str) -> list[str]:
        return [f"![{n}](figures/{n}.svg)" for n in figures if n.startswith(f"{stem}-")]

    text = [
        "# P3 follow-up: what one shared mask costs a query's relevant documents",
        "",
        "Exploratory and descriptive, on published F2 training queries and training judgments,",
        "with frozen vectors at D = 1024. Stored documents are unchanged and masked queries are",
        "not re-normalized. This is not selector training or a ranking evaluation. Regret",
        "describes a compromise in similarity: high regret does not show retrieval damage or",
        "rule out a better ranking mask, and low overlap with low regret is a valid outcome.",
        "Held-out documents are documents of a training query, not validation or test queries,",
        "so nothing here is evidence about unseen queries.",
        "",
        f"- Run `{run.name}` (commit `{manifest.git_commit[:12]}`, dirty "
        f"`{manifest.git_dirty}`, device `{manifest.device}`)",
        f"- Pilot bound on queries: {inputs['max_queries']}; seed {inputs['seed']}",
        "",
        "## What the curves show",
        "",
        *_findings(summaries),
        "",
        "A nonzero mean regret under the mean-optimal mask shows that no single mask reaches every",
        "positive's optimum at that `k`. It does not show that the observed worst-positive regret",
        "is irreducible, since a minimax mask could trade mean score for a better worst case.",
        "",
        "These compare similarity at equal `k` and describe training pairs. They do not show a",
        "mask harms retrieval, that no better ranking mask exists, or that an unseen query would",
        "behave the same. Positives also lie closer to one another than unjudged documents, so a",
        "smaller regret than the unjudged-document mask is not attributed to relevance itself.",
        "",
        "## Definitions",
        "",
        "`c_ij = q_j * d_ij`. A positive's optimum at `k` is the sum of its own `k` largest",
        "`c_ij`. The shared mask is the top `k` of the weighted mean of a group's",
        "contributions, with the repository's relevance weights (`2^grade - 1`, normalized;",
        "every grade here is 1). It exactly maximizes the weighted mean score at that `k`, and",
        "no single positive's score. Regret is a positive's optimum minus its score under the",
        "mask at the same `k`; dividing by `sum|c_ij|` gives regret / mass, undefined for a",
        "document with no mass. Masks can differ and tie in score, so regret is compared, and",
        "overlap reported apart.",
        "",
        "A query counts once in query-balanced summaries. Pair-weighted curves are labeled and",
        "separate. Intervals are 95% over queries, pointwise and exploratory: neighboring `k`",
        "are strongly dependent, so they are not independent confirmations. Repeat ranges and",
        "between-partition spread show variation from partitions, a different thing from",
        "sampling queries. No threshold for a small loss is applied.",
        "",
        "## Coverage",
        "",
        *_coverage_note(summaries),
        "",
        *_markdown(tables["coverage"]),
        "",
        "## The cost of one shared mask",
        "",
        "Every `k` is in `tables/curves-<environment>.csv`; checkpoint rows sample those",
        "curves. The uniform random mask's regret is an exact expectation, a reference with no",
        "structure.",
        "",
        *show("shared"),
        "",
        *_markdown(tables["shared"]),
        "",
        "## The worst positive at a matched count",
        "",
        "A group's worst depends on the group's size, so natural-group worst values cannot be",
        "compared across strata. Here a mask is built on a drawn group of fixed size and scored",
        "on it. Check the query counts: a small cohort describes itself, and queries with more",
        "positives enter only the larger sizes. The mask is mean-optimal and not minimax. Another",
        "mask could give up mean score for a better worst positive, so these values are not the",
        "least worst-case regret one mask can reach. That was not computed here.",
        "",
        *show("fixed"),
        "",
        *_markdown(tables["fixed-count"]),
        "",
        "## Transfer among positives",
        "",
        "A mask is built only from construction documents, and their grades alone set its",
        "weights. Held-out documents are scored against their own optima. Cells `a-a` use `a`",
        "documents on each side, for queries with at least `2a` positives; `half` splits each",
        "query's positives. Two positives give both directed transfers, and a split is",
        "exhaustive when at most `partitions` exist; otherwise seeded draws. The all-positive",
        "mask saw the held-out documents, and is a privileged reference only. A mask of",
        "unjudged corpus documents, paired with the same query, shows what sharing the query",
        "alone transfers; those documents are not negatives. Construction-side regret and",
        "held-out regret use one mask, so their difference is what leaving a document out",
        "costs. Any cell's scope is only the queries that enter it.",
        "",
        *show("transfer"),
        "",
        *_markdown(tables["transfer"]),
        "",
        "Each cell above covers its own eligible queries, so differences between cells mix",
        "construction size with which queries enter. The next table restricts cells to the same",
        "queries, from the stored per-query arrays: each cohort is the queries eligible for its",
        "largest cell. It describes these cohorts, whose queries have many positives, and",
        "partitions are drawn separately per cell. The full curves are in",
        "`tables/transfer-common-cohort-curves-<environment>.csv`.",
        "",
        *_markdown(tables["transfer-common-cohort"]),
        "",
        "The half split, by the query's positive count:",
        "",
        *_markdown(tables["transfer-half-strata"]),
        "",
        "## Overlap and regret",
        "",
        "Regret and overlap are computed from the same orderings, so some relationship is",
        "mechanical. The bins are pair-weighted.",
        "",
        *_markdown(tables["overlap"]),
        "",
        "## Document similarity, as context",
        "",
        "Spearman across queries with the cosine among a query's positives, as recorded by the",
        "canonical P3 run. Context only: it does not attribute regret to similarity or relevance.",
        "",
        *_markdown(tables["document-similarity"]),
        "",
        "## Query-balanced and pair-weighted",
        "",
        "A query counts once in the first columns, and a pair counts once in the others, which",
        "gives large groups more say.",
        "",
        *_markdown(tables["weighting"]),
        "",
        "## Verification and provenance",
        "",
        "Identities a correct run holds to rounding, and agreement with the canonical P3 run at",
        "its checkpoints and orderings.",
        "",
        *_markdown(tables["checks"]),
        "",
        *_markdown(tables["sources"]),
        "",
        "## Cost",
        "",
        "One run on the device named, and it varies with load. Read, transfer, synchronized",
        "compute, and write cover only the per-query pass, a part of the total. Peak RSS counts",
        "pages of memory-mapped files.",
        "",
        *_markdown(tables["cost"]),
        "",
    ]
    (target / "README.md").write_text("\n".join(text), encoding="utf-8")
    return target / "README.md"
