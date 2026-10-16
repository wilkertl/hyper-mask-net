"""The E1 report and gate G2, generated from result files only (F4 research D4–D9, research R8).

Nothing numeric is typed here: every value comes from the latest `transfer-matrix` run and the
seeded selectors it names. Protocol: in-domain and transfer. Scoring: exact.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperdime.artifacts import latest_artifact
from hyperdime.contracts.manifests import read_manifest
from hyperdime.contracts.schemas import EvaluationResult, TransferCell, load_records
from hyperdime.evaluation.statistics import holm, paired_bootstrap_ci, paired_randomization_test
from hyperdime.evaluation.transfer import ALPHA, seed_average
from hyperdime.reports.e0 import _csv, _fmt, _markdown, _yes

HEADLINE_K = 307
DIM = 1024
MIN_SEEDS = 3
BASELINES = ("full", "prefix", "random", "global-importance", "global-selector")
MEASURES = ("overlap", "js", "kl_ab", "kl_ba", "spearman_importance", "spearman_frequency")
NOTES = {
    "msmarco": (
        "E0 does not reproduce Learning-to-Select on MS MARCO (ADR-0006). E1 scores MS MARCO dev "
        "against the full 8.8M-passage corpus; the paper evaluates the `test` split of an "
        "unpublished `msmarco_subset_50k` folder, with an ideal DCG taken from the retrieved list "
        "(F4 research D9)."
    ),
    "nfcorpus": (
        "NFCorpus's selector does not fit its oracle distribution as a calibrated distribution "
        "(ADR-0006); read its shift with its ranking-fit result (F4 research D2)."
    ),
}
GLOBAL_NOTE = (
    "`global-selector` is F3's single selector trained on the training labels of all "
    "environments together (never on a test label), so it is not an in-domain baseline."
)


def note(*environments: str) -> str:
    """The note reference of a row that involves an environment with a note (FR-016)."""
    noted = sorted({env for env in environments if env in NOTES})
    return "see note: " + ", ".join(noted) if noted else "—"


@dataclass
class E1Results:
    run: str
    commit: str
    finished: str
    environments: list[str]
    seeds: list[int]
    results: dict[tuple[str, str], dict[str, EvaluationResult]]  # (test env, set) -> method
    budgets: dict[tuple[str, str], dict[tuple[str, int], EvaluationResult]]  # -> (method, k)
    cells: list[TransferCell]
    fit: dict[str, Any]
    behavior: dict[str, Any]
    battery: dict[str, Any]
    seeded: dict[str, dict[int, dict[str, Any]]]  # env -> seed -> selection.json
    seeded_runs: list[str]


def load(root: Path) -> E1Results:
    directory = latest_artifact(root, "evaluation/E1", "all")
    manifest = read_manifest(directory)
    sources = _json(directory / "inputs.json")
    environments = sorted(e for e in sources if e != "global")
    results: dict[tuple[str, str], dict[str, EvaluationResult]] = {}
    budgets: dict[tuple[str, str], dict[tuple[str, int], EvaluationResult]] = {}
    for path in sorted(directory.glob("results-*.jsonl")):
        records = load_records(EvaluationResult, path)
        test = records[0].environment
        query_set = path.stem.removeprefix(f"results-{test}-")
        results[(test, query_set)] = {
            r.selector: r for r in records if r.k == HEADLINE_K or r.selector == "full"
        }
        budgets[(test, query_set)] = {(r.selector, r.k): r for r in records}
    cells = load_records(TransferCell, directory / "cells.jsonl")
    seeds = sorted(cells[0].seeds)
    if len(seeds) < MIN_SEEDS:
        raise ValueError(f"E1 needs at least {MIN_SEEDS} seeds per environment; found {seeds}")
    seeded = {
        env: {
            seed: _json(
                root
                / "selectors"
                / "seeded"
                / env
                / f"seed-{seed}"
                / sources[env][f"seed-{seed}"]
                / "selection.json"
            )  # fmt: skip
            for seed in seeds
        }
        for env in environments
    }
    return E1Results(
        directory.name,
        manifest.git_commit,
        f"{manifest.finished_at:%Y-%m-%d}" if manifest.finished_at else "",
        environments,
        seeds,
        results,
        budgets,
        cells,
        _json(directory / "fit.json"),
        _json(directory / "behavior.json"),
        _json(directory / "battery.json"),
        seeded,
        [sources[env][f"seed-{seed}"] for env in environments for seed in seeds],
    )


def criteria(e1: E1Results) -> list[dict[str, Any]]:
    """Research R8's two G2 criteria, computed from the cells and the behavior result."""
    pairs = [
        c for c in e1.cells if c.query_set == "test" and c.train_environment != c.test_environment
    ]
    drops = sum(
        1 for c in pairs if c.degradation > 0 and c.p_holm is not None and c.p_holm <= ALPHA
    )
    behavior = e1.behavior
    return [
        {
            "name": "1. cross-domain nDCG@10 significantly below in-domain (Holm, alpha 0.05)",
            "value": float(drops),
            "shown": f"{drops} of {len(pairs)} ordered pairs",
            "threshold": f">= {(len(pairs) + 1) // 2} of {len(pairs)}",
            "met": bool(pairs) and 2 * drops >= len(pairs),
        },
        {
            "name": "2. cross-environment overlap@k below within-environment (exact permutation)",
            "value": float(behavior["p_value"]),
            "shown": (
                f"T = {_fmt(behavior['statistic'], '+')}, p = {_fmt(behavior['p_value'])} "
                f"over {behavior['assignments']} assignments"
            ),
            "threshold": f"T > 0 and p <= {ALPHA}",
            "met": behavior["statistic"] > 0 and behavior["p_value"] <= ALPHA,
        },
    ]


def in_domain(e1: E1Results, test: str, query_set: str = "test") -> dict[str, float]:
    """Seed-averaged per-query nDCG@10 of the test environment's own selectors."""
    found = e1.results[(test, query_set)]
    averaged = seed_average([found[f"selector:{test}:s{s}"].per_query for s in e1.seeds])
    return {q: v["ndcg@10"] for q, v in averaged.items()}


def comparisons(e1: E1Results, against: Mapping[str, Sequence[str]]) -> list[dict[str, Any]]:
    """In-domain selector against other methods on each test set, Holm over all rows."""
    rows: list[dict[str, Any]] = []
    for test in e1.environments:
        own = in_domain(e1, test)
        for method in against[test]:
            other = e1.results[(test, "test")][method].per_query
            qids = sorted(own)
            xs, ys = [own[q] for q in qids], [other[q]["ndcg@10"] for q in qids]
            low, high = paired_bootstrap_ci(xs, ys)
            reference = e1.results[(test, "test")]
            rows.append({
                "test": test, "method": method,
                "full": reference["full"].metrics["ndcg@10"],
                "prefix": reference["prefix"].metrics["ndcg@10"],
                "in_domain": sum(xs) / len(xs), "other": sum(ys) / len(ys),
                "diff": sum(xs) / len(xs) - sum(ys) / len(ys), "ci_low": low, "ci_high": high,
                "p": paired_randomization_test(xs, ys),
            })  # fmt: skip
    adjusted = holm({str(i): row["p"] for i, row in enumerate(rows)})
    for i, row in enumerate(rows):
        row["p_holm"] = adjusted[str(i)]
    return rows


def payoff(e1: E1Results) -> list[dict[str, Any]]:
    return comparisons(e1, {test: ["global-selector"] for test in e1.environments})


def fixed_subset_rows(e1: E1Results) -> list[dict[str, Any]]:
    return comparisons(e1, {t: [f"fixed:{a}" for a in e1.environments] for t in e1.environments})


def progression(e1: E1Results) -> list[list[str]]:
    """Mean nDCG@10 of every method at every budget; seeds of a selector are averaged, and the
    last column (k = 1024) is the full query for every method."""
    ks = sorted({k for found in e1.budgets.values() for method, k in found if method != "full"})
    methods = [m for m in BASELINES if m != "full"]
    methods += [f"selector:{env}" for env in e1.environments]
    methods += [f"fixed:{env}" for env in e1.environments]
    header = [f"{round(100 * k / DIM)}% (k={k})" for k in [*ks, DIM]]
    rows = [["test", "query set", "method", "protocol", *header, "note"]]
    for (test, query_set), found in sorted(e1.budgets.items()):
        full = found[("full", DIM)].metrics["ndcg@10"]
        rows.append([test, query_set, "full", "in-domain", *[_fmt(full)] * len(header), "—"])
        for method in methods:
            seeds = [method]
            if method.startswith("selector:"):
                seeds = [f"{method}:s{s}" for s in e1.seeds]
            values = [
                sum(found[(name, k)].metrics["ndcg@10"] for name in seeds) / len(seeds) for k in ks
            ]
            rows.append([
                test, query_set, method, found[(seeds[0], ks[0])].protocol,
                *(_fmt(v) for v in values), _fmt(full),
                note(test, method.split(":")[1] if ":" in method else test),
            ])  # fmt: skip
    return rows


def departures(e1: E1Results) -> list[str]:
    """Environments where F3's validation pick was not the wide-grid family F4 uses."""
    lines = []
    for env in e1.environments:
        source = e1.seeded[env][e1.seeds[0]]["source"]
        if source["selected"] != "wide-grid":
            scores = source["family_scores"]
            lines.append(
                f"`{env}`: F3 selected `{source['selected']}` (validation nDCG@10 "
                f"{_fmt(scores[source['selected']])}); F4 uses `wide-grid` "
                f"({_fmt(scores['wide-grid'])}), as the spec's clarification Q1 decided."
            )
    return lines


def render(e1: E1Results) -> tuple[str, dict[str, str]]:
    tables: dict[str, list[list[str]]] = {}
    g2 = criteria(e1)
    tables["g2"] = [
        ["criterion", "value", "threshold", "met"],
        *[[c["name"], c["shown"], c["threshold"], _yes(c["met"])] for c in g2],
    ]
    cells = {(c.train_environment, c.test_environment, c.query_set): c for c in e1.cells}
    for name, gated in (("transfer", True), ("trec", False)):
        ap = [] if gated else ["selector AP", "in-domain AP"]
        rows = [[
            "train", "test", "query set", "protocol", "selector nDCG@10", "in-domain nDCG@10",
            "full nDCG@10", "prefix nDCG@10", "degradation", "ci_low", "ci_high", "p", "p_holm",
            "overlap@k", *ap, "note",
        ]]  # fmt: skip
        for c in e1.cells:
            if (c.query_set == "test") != gated:
                continue
            methods = e1.results[(c.test_environment, c.query_set)]
            own = cells[(c.test_environment, c.test_environment, c.query_set)]
            ap_values = [] if gated else [_fmt(c.metrics["ap"]), _fmt(own.metrics["ap"])]
            rows.append([
                c.train_environment, c.test_environment, c.query_set,
                "in-domain" if c.train_environment == c.test_environment else "transfer",
                _fmt(c.metrics["ndcg@10"]), _fmt(c.in_domain_ndcg10),
                _fmt(methods["full"].metrics["ndcg@10"]),
                _fmt(methods["prefix"].metrics["ndcg@10"]),
                _fmt(c.degradation, "+"), _opt(c.ci_low, "+"), _opt(c.ci_high, "+"),
                _opt(c.p_value), _opt(c.p_holm), _fmt(c.overlap_at_k), *ap_values,
                note(c.train_environment, c.test_environment),
            ])  # fmt: skip
        tables[name] = rows
    method_rows = [["test", "query set", "method", "k", "nDCG@10", "AP"]]  # headline budget
    for (test, query_set), methods in sorted(e1.results.items()):
        ordered = [m for m in BASELINES if m in methods]
        for method in ordered + sorted(set(methods) - set(ordered)):
            r = methods[method]
            method_rows.append([test, query_set, method, str(r.k), _fmt(r.metrics["ndcg@10"]),
                                _fmt(r.metrics["ap"])])  # fmt: skip
    tables["methods"] = method_rows
    tables["progression"] = progression(e1)
    seed_rows = [["environment", "seed", "cell", "validation nDCG@10", "test nDCG@10", "note"]]
    for env in e1.environments:
        for seed in e1.seeds:
            record = e1.seeded[env][seed]
            result = e1.results[(env, "test")][f"selector:{env}:s{seed}"]
            seed_rows.append([env, str(seed), record["source"]["cell"],
                              _fmt(record["validation_ndcg@10"]),
                              _fmt(result.metrics["ndcg@10"]), note(env)])  # fmt: skip
    tables["seeds"] = seed_rows
    behavior = e1.behavior
    tables["behavior"] = [
        ["a", "b", "within", *MEASURES],
        *[[p["a"], p["b"], _yes(p["within"]), *(_fmt(p[m]) for m in MEASURES)]
          for p in behavior["pairs"]],
    ]  # fmt: skip
    noise = [["measure", "within environments (seed noise floor)", "across environments"]]
    for measure in MEASURES:
        within = [p[measure] for p in behavior["pairs"] if p["within"]]
        cross = [p[measure] for p in behavior["pairs"] if not p["within"]]
        noise.append([measure, _fmt(sum(within) / len(within)), _fmt(sum(cross) / len(cross))])
    tables["noise_floor"] = noise
    env_pairs = [["environments", "cross overlap@k", "within overlap@k (each)"]]
    for pair, values in sorted(behavior["environment_pairs"].items()):
        each = ", ".join(
            f"{k.removeprefix('within_')} {_fmt(v)}"
            for k, v in sorted(values.items())
            if k.startswith("within_")
        )
        env_pairs.append([pair.replace("|", " / "), _fmt(values["cross"]), each])
    tables["environment_pairs"] = env_pairs
    tables["fit"] = [
        ["environment", "mean gain over global importance", "queries", "p", "p_holm", "fits",
         "note"],
        *[[env, _fmt(r["mean_gain"], "+"), str(r["num_queries"]), _fmt(r["p_value"]),
           _fmt(r["p_holm"]), _yes(r["fits"]), note(env)] for env, r in sorted(e1.fit.items())],
    ]  # fmt: skip
    for name, rows_ in (("payoff", payoff(e1)), ("fixed_subset", fixed_subset_rows(e1))):
        tables[name] = [
            ["test", "method", "full", "prefix", "in-domain", "method nDCG@10", "diff", "ci_low",
             "ci_high", "p", "p_holm", "note"],
            *[[r["test"], r["method"], _fmt(r["full"]), _fmt(r["prefix"]),
               _fmt(r["in_domain"]), _fmt(r["other"]),
               _fmt(r["diff"], "+"), _fmt(r["ci_low"], "+"), _fmt(r["ci_high"], "+"),
               _fmt(r["p"]), _fmt(r["p_holm"]),
               note(r["test"], r["method"].removeprefix("fixed:"))] for r in rows_],
        ]  # fmt: skip
    passed = all(c["met"] for c in g2)
    notes = [f"- **{env}**: {NOTES[env]}" for env in e1.environments if env in NOTES]
    lines = [
        "# E1: selector shift and gate G2",
        "",
        "Generated by `hyperdime report E1` from the `transfer-matrix` run and the seeded",
        "selectors named below; no number is typed by hand. Protocols: in-domain and transfer.",
        f"Scoring: exact inner product over the full corpus. Budget: k = {HEADLINE_K} (30%).",
        f"Selectors: F3's `wide-grid` cell per environment, seeds {', '.join(map(str, e1.seeds))};",
        "seed scores are averaged per query before every paired test (F4 research D4).",
        "",
        f"- transfer-matrix `{e1.run}` (commit `{e1.commit[:12]}`)",
        *[
            f"- `{env}` seeded selectors: "
            + ", ".join(f"seed {s} cell `{e1.seeded[env][s]['source']['cell']}`" for s in e1.seeds)
            for env in e1.environments
        ],
        "",
        "## Gate G2 (research R8)",
        "",
        *_markdown(tables["g2"]),
        "",
        f"**G2 criteria: {'both met' if passed else 'not both met'}** "
        "(the decision is the researcher's, in the G2 ADR)",
        "",
        "## Notes",
        "",
        *(notes or ["- none"]),
        *[f"- {line}" for line in departures(e1)],
        "",
        "## Transfer matrix: test queries (criterion 1)",
        "",
        "Degradation = in-domain minus cross-domain nDCG@10, paired randomization test over the",
        "test environment's queries, Holm over the six ordered pairs, 95% bootstrap interval;",
        "overlap@k is the mean over seed pairs of the two environments' selectors.",
        "",
        *_markdown(tables["transfer"]),
    ]
    lines += [
        "",
        "## nDCG@10 by budget (every 10% of the 1024 dimensions)",
        "",
        "Every method at every budget on every query set, exact scoring; `selector:<env>` is the",
        "mean over that environment's seeds; at k = 1024 every mask keeps the full query. Reported",
        f"for context: gate G2 reads k = {HEADLINE_K} only.",
        "",
        *_markdown(tables["progression"]),
    ]
    if len(tables["trec"]) > 1:
        lines += [
            "",
            "## Transfer to evaluation query sets (reported, not counted in criterion 1)",
            "",
            *_markdown(tables["trec"]),
        ]
    lines += [
        "",
        "## Behavior on the common battery (criterion 2)",
        "",
        f"Top-{behavior['k']} masks on a battery of {e1.battery['n']} unlabeled test queries from",
        f"each environment (seed {e1.battery['seed']}, SHA-256 `{e1.battery['sha256'][:12]}`);",
        "within-environment seed pairs are the noise floor (F4 research D5, D7).",
        "",
        *_markdown(tables["noise_floor"]),
        "",
        *_markdown(tables["environment_pairs"]),
        "",
        "## Selector fit as a ranking (validation queries, F4 research D2)",
        "",
        *_markdown(tables["fit"]),
        *(["", f"NFCorpus: {NOTES['nfcorpus']}"] if "nfcorpus" in e1.environments else []),
        "",
        "## Payoff of in-domain selectors over the global selector (context)",
        "",
        GLOBAL_NOTE,
        "",
        *_markdown(tables["payoff"]),
        "",
        "## In-domain selectors against transferred fixed subsets (context, ADR-0006)",
        "",
        *_markdown(tables["fixed_subset"]),
        "",
        "## Seeds",
        "",
        *_markdown(tables["seeds"]),
        "",
        "## Every method on every query set",
        "",
        GLOBAL_NOTE,
        "",
        *_markdown(tables["methods"]),
        "",
        "All selector pairs are in `tables/behavior.csv`.",
    ]
    return "\n".join(lines) + "\n", {name: _csv(rows) for name, rows in tables.items()}


def write(root: Path, output: Path) -> Path:
    text, csvs = render(load(root))
    (output / "tables").mkdir(parents=True, exist_ok=True)
    (output / "README.md").write_text(text, encoding="utf-8")
    for name, content in csvs.items():
        (output / "tables" / f"{name}.csv").write_text(content, encoding="utf-8")
    return output / "README.md"


def _opt(value: float | None, sign: str = "") -> str:
    return "—" if value is None else _fmt(value, sign)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
