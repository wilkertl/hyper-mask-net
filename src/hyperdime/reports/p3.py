"""The P3 report: per-dimension contributions of known pairs, generated from result files only.

Reads the latest ``analyze-representations`` run (``inputs.json``, ``manifest.json``, and each
environment's ``summary.json``) and nothing else. No number is typed here. Regenerating from the
same run gives identical bytes, cost lines included, because costs are read from the run too.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hyperdime.artifacts import latest_artifact
from hyperdime.contracts.manifests import read_manifest
from hyperdime.contracts.validation import ContractError
from hyperdime.reports import p3_controls
from hyperdime.reports.e0 import _csv, _markdown
from hyperdime.reports.p3_svg import Series as _Series
from hyperdime.reports.p3_svg import _ticks as _ticks
from hyperdime.reports.p3_svg import bars as _bars
from hyperdime.reports.p3_svg import lines as _lines

DIM = 1024
F2_PROVENANCE = "published F2 artifacts"
STABLE_SPEARMAN = p3_controls.STABLE_SPEARMAN
FREQUENCY_BUDGETS = 3


def load(root: Path) -> tuple[str, dict[str, Any], str, bool | None, dict[str, dict[str, Any]]]:
    """The run's name, its inputs, its commit and dirty flag, and each environment's summary, with
    its all-cardinality curves under ``"curves"``."""
    directory = latest_artifact(root, "evaluation/P3", "all")
    inputs = json.loads((directory / "inputs.json").read_text(encoding="utf-8"))
    manifest = read_manifest(directory)
    summaries = {
        env: json.loads((directory / env / "summary.json").read_text(encoding="utf-8"))
        for env in inputs["environments"]
    }
    for env, summary in summaries.items():
        path = directory / env / "curves.json"
        if not path.exists():
            raise ContractError(
                f"{directory} predates the all-cardinality curves; rerun analyze-representations "
                "(its own report is kept at reports/P3-spike/initial-run)"
            )
        summary["curves"] = json.loads(path.read_text(encoding="utf-8"))
    return directory.name, inputs, manifest.git_commit, manifest.git_dirty, summaries


def _num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _spread(item: dict[str, Any], digits: int = 3) -> str:
    """``mean (p10-p90)`` of a stored distribution."""
    if not item["n"]:
        return "n/a"
    return (
        f"{_num(item['mean'], digits)} ({_num(item['p10'], digits)} to {_num(item['p90'], digits)})"
    )


def _coverage_table(inputs: dict[str, Any], summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "training queries", "analyzed", "relevant pairs", "mean positives",
              "max positives", "single-positive queries", "multi-positive queries",
              "judged nonrelevant pairs", "mean unjudged documents per query",
              "grades"]]  # fmt: skip
    for env, s in summaries.items():
        c = s["coverage"]
        grades = ", ".join(f"{g}: {n}" for g, n in sorted(c["grades"].items()))
        table.append([
            env, str(c["training_queries"]), str(c["selected_queries"]), str(c["relevant_pairs"]),
            _num(c["positives_per_query"]["mean"], 2), str(c["max_positives_per_query"]),
            str(c["single_positive_queries"]), str(c["multi_positive_queries"]),
            str(c["judged_nonrelevant_pairs"]), _num(c["mean_unjudged_per_query"], 1), grades,
        ])  # fmt: skip
    return table


def _strata_table(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "positives per query", "queries", "pairs"]]
    for env, s in summaries.items():
        table += [
            [env, row["positives"], str(row["queries"]), str(row["pairs"])]
            for row in s["coverage"]["strata"]
        ]
    return table


def _inputs_table(inputs: dict[str, Any], summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "inputs", "dataset run", "dataset fingerprint",
              "train judgments sha256",
              "embeddings run", "documents sha256", "queries sha256", "split seed",
              "validation fraction", "model revision", "stored dtype", "pooling"]]  # fmt: skip
    for env in summaries:
        i = inputs[env]
        table.append([
            env, i["provenance"], i["dataset"], i["dataset_fingerprint"][:12],
            i["qrels_train_sha256"][:12], i["embeddings"], i["documents_sha256"][:12],
            i["queries_sha256"][:12], str(i["split_seed"]), str(i["validation_fraction"]),
            i["model_revision"][:12], i["dtype"], i["pooling"],
        ])  # fmt: skip
    return table


SCALARS = (
    ("score", "full-vector score"),
    ("query_norm", "query norm"),
    ("document_norm", "document norm"),
    ("positive_mass", "positive mass P"),
    ("negative_mass", "negative mass N"),
    ("negative_share", "N / (P + N)"),
    ("n_positive", "dimensions with c > 0"),
    ("n_negative", "dimensions with c < 0"),
    ("n_zero", "dimensions with c = 0"),
    ("sign_agreement", "sign agreement, by count"),
    ("effective_positive_dims", "effective positive dimensions"),
    ("effective_negative_dims", "effective negative dimensions"),
    ("top_1_share", "share of P in the largest c"),
    ("top_10_share", "share of P in the top 10"),
    ("top_100_share", "share of P in the top 100"),
    ("dims_for_50", "dimensions for 50% of P"),
    ("dims_for_80", "dimensions for 80% of P"),
    ("dims_for_90", "dimensions for 90% of P"),
    ("dims_for_95", "dimensions for 95% of P"),
    ("dims_for_99", "dimensions for 99% of P"),
    ("k_return", "smallest k whose maximum reaches the full score"),
    ("log_variance_query_share", "var log|c|: query share"),
    ("log_variance_document_share", "var log|c|: document share"),
    ("log_variance_covariance_share", "var log|c|: covariance share"),
    ("top_query_percentile", "|q| percentile of the top 50 c"),
    ("top_document_percentile", "|d| percentile of the top 50 c"),
)


def _decomposition_table(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["statistic", "environment", "all queries, mean (p10 to p90)",
              "single positive, mean (p10 to p90)", "multiple positives, mean (p10 to p90)",
              "pair-weighted mean"]]  # fmt: skip
    for name, label in SCALARS:
        for env, s in summaries.items():
            row = s["pairs"]["scalar"][name]
            table.append([label, env, _spread(row["all"]), _spread(row["single"]),
                          _spread(row["multi"]), _num(row["pair_weighted_mean"])])  # fmt: skip
    return table


def _budget_table(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "k", "max pair score", "gain over full / (P + N)",
              "|max - full| / (P + N)", "positive mass removed",
              "negative mass removed"]]  # fmt: skip
    for env, s in summaries.items():
        per = s["pairs"]["per_budget"]
        for b, k in enumerate(s["budgets"]):
            table.append([
                env, str(k), _spread(per["max_score"]["all"][b]),
                _spread(per["gain_over_full"]["all"][b]),
                _spread(per["preservation_error"]["all"][b]),
                _spread(per["removed_positive_mass"]["all"][b]),
                _spread(per["removed_negative_mass"]["all"][b]),
            ])  # fmt: skip
    return table


def _agreement_table(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "k", "within-query Jaccard", "worst pair in the query",
              "across-query Jaccard", "chance", "dims in all positives",
              "dims in a majority", "dims in at most half", "dims in any"]]  # fmt: skip
    for env, s in summaries.items():
        agree = s["agreement"]
        across = s["curves"]["across_query"].get("positive_raw")
        for b, k in enumerate(s["budgets"]):
            table.append([
                env, str(k), _spread(agree["mean_jaccard"]["multi"][b]),
                _spread(agree["min_jaccard"]["multi"][b]),
                p3_controls.spread_at(across, k) if across else "n/a",
                _num(s["chance_jaccard"][b]),
                _spread(agree["core_dims"]["multi"][b], 1),
                _spread(agree["majority_dims"]["multi"][b], 1),
                _spread(agree["minority_dims"]["multi"][b], 1),
                _spread(agree["union_dims"]["multi"][b], 1),
            ])  # fmt: skip
    return table


def _conflict_table(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "queries", "n", "share of dims with a sign conflict",
              "dims always helpful", "dims always harmful", "mean Spearman of c orderings",
              "worst Spearman pair", "mass cancelled by averaging",
              "variance between positives / second moment",
              "dims whose mean opposes a positive"]]  # fmt: skip
    for env, s in summaries.items():
        a = s["agreement"]
        for subset in _multi_subsets(s):
            table.append([
                env, subset, str(a["sign_conflict_share"][subset]["n"]),
                _spread(a["sign_conflict_share"][subset]),
                _spread(a["always_helpful_dims"][subset], 1),
                _spread(a["always_harmful_dims"][subset], 1),
                _spread(a["mean_spearman"][subset]), _spread(a["min_spearman"][subset]),
                _spread(a["aggregate_cancelled_share"][subset]),
                _spread(a["aggregate_within_positive_variance_share"][subset]),
                _spread(a["aggregate_flipped_dims"][subset], 1),
            ])  # fmt: skip
    return table


def _multi_subsets(summary: dict[str, Any]) -> list[str]:
    return [k for k in summary["agreement"]["sign_conflict_share"] if k != "single"]


def _aggregate_table(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "k", "positive's mask inside the mean's mask",
              "least-served positive", "lost pair score / (P + N)", "worst lost pair score",
              "minority dims the mean's mask drops",
              "conflicted dims in the mean's mask"]]  # fmt: skip
    for env, s in summaries.items():
        a = s["agreement"]
        for b, k in enumerate(s["budgets"]):
            table.append([
                env, str(k), _spread(a["aggregate_mean_overlap"]["multi"][b]),
                _spread(a["aggregate_min_overlap"]["multi"][b]),
                _spread(a["aggregate_mean_regret"]["multi"][b]),
                _spread(a["aggregate_worst_regret"]["multi"][b]),
                _spread(a["aggregate_minority_dropped"]["multi"][b]),
                _spread(a["aggregate_conflicted_in_mask"]["multi"][b]),
            ])  # fmt: skip
    return table


def _recurrence_table(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "k", "chance frequency", "max frequency", "effective dims",
              "dims never selected", "mask inside the k most frequent dims",
              "Spearman with pair-weighted frequency (pair-weighted)",
              "Spearman with mean|q| x mean|d|"]]  # fmt: skip
    for env, s in summaries.items():
        for row in s["corpus"]["per_budget"]:
            table.append([
                env, str(row["k"]), _num(row["chance_frequency"]), _num(row["max_frequency"]),
                _num(row["effective_dims"], 1), str(row["dims_never_selected"]),
                _spread(row["core_overlap"]), _num(row["pair_weighted_spearman"]),
                _num(row["marginal_magnitude_spearman"]),
            ])  # fmt: skip
    return table


def _stability_table(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "k", "draws", "split-half Spearman, mean (min to max)",
              "top-k overlap of the two halves, mean (min to max)",
              "same, dimension identity shuffled per query",
              f"stable (min >= {STABLE_SPEARMAN})"]]  # fmt: skip
    for env, s in summaries.items():
        stability = s["corpus"]["stability"]
        for row in stability["per_budget"]:
            spearman = row["spearman"]
            table.append([
                env, str(row["k"]), str(stability["draws"]), _range(spearman),
                _range(row["top_k_overlap"]), _num(row["shuffled_dimension_spearman"]),
                "n/a" if spearman["min"] is None else _yes(spearman["min"] >= STABLE_SPEARMAN),
            ])  # fmt: skip
    return table


def _range(item: dict[str, Any]) -> str:
    return (
        "n/a"
        if item["mean"] is None
        else f"{_num(item['mean'])} ({_num(item['min'])} to {_num(item['max'])})"
    )


def _yes(flag: bool) -> str:
    return "yes" if flag else "no"


FINDINGS = (
    ("negative_share", "N / (P + N)"),
    ("sign_agreement", "sign agreement"),
    ("effective_positive_dims", "effective positive dimensions"),
    ("dims_for_90", "dimensions for 90% of P"),
    ("k_return", "smallest k reaching the full score"),
    ("sign_conflict_share", "share of dims with a sign conflict"),
    ("aggregate_cancelled_share_multi", "mass cancelled by averaging"),
)


def _finding_table(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["finding", "environment", "query-balanced value",
              "difference between query halves, mean (min to max)"]]  # fmt: skip
    for name, label in FINDINGS:
        for env, s in summaries.items():
            row = s["finding_stability"][name]
            table.append([label, env, _num(row["overall"]), _range(row)])
    for env, s in summaries.items():
        for k in s["budgets"]:
            row = s["finding_stability"][f"mean_jaccard_k{k}"]
            table.append([f"within-query Jaccard, k = {k}", env, _num(row["overall"]), _range(row)])
    return table


def _group_table(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "k", "positives per query", "queries",
              "Spearman of the stratum's frequency with the corpus's"]]  # fmt: skip
    for env, s in summaries.items():
        for groups in s["corpus"]["groups"]:
            table += [
                [env, str(groups["k"]), row["positives"], str(row["queries"]),
                 _num(row["spearman_with_corpus"])]
                for row in groups["strata"]
            ]  # fmt: skip
    return table


def _component_table(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "k", "explained by the first component",
              "explained by the first ten"]]  # fmt: skip
    for env, s in summaries.items():
        for groups in s["corpus"]["groups"]:
            explained = groups["explained_by_components"]
            table.append([env, str(groups["k"]), _num(explained[0]) if explained else "n/a",
                          _num(explained[-1]) if explained else "n/a"])  # fmt: skip
    return table


def _grade_table(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "grade", "queries", "pairs", "score", "positive mass",
              "N / (P + N)", "effective positive dims", "dims for 90% of P"]]  # fmt: skip
    for env, s in summaries.items():
        for row in s["grades"]:
            table.append([
                env, str(row["grade"]), str(row["queries"]), str(row["pairs"]),
                _spread(row["score"]), _spread(row["positive_mass"]),
                _spread(row["negative_share"]), _spread(row["effective_positive_dims"], 1),
                _spread(row["dims_for_90"], 1),
            ])  # fmt: skip
    return table


def _nonrelevant_table(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "queries with a judged nonrelevant document", "coverage", "pairs",
              "statistic", "mean (p10 to p90), per pair"]]  # fmt: skip
    for env, s in summaries.items():
        n = s["nonrelevant"]
        head = [
            env,
            str(n["queries_with_judged_nonrelevant"]),
            _num(n["coverage"]),
            str(n["pairs"]),
        ]
        if not n["statistics"]:
            table.append([*head, "none available", "n/a"])
        for name, item in sorted(n["statistics"].items()):
            table.append([*head, name, _spread(item)])
    return table


def _cost_table(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "device", "batch pairs", "pairs", "seconds (total, offline)",
              "pair analysis", "corpus summaries",
              "pairs per second", "peak RSS (MiB)", "artifact MiB",
              "bytes per pair (arrays)"]]  # fmt: skip
    for env, s in summaries.items():
        cost, pairs = s["cost"], s["coverage"]["relevant_pairs"]
        sizes = cost["artifact_bytes"]
        arrays = sizes["pair-contributions.npy"] + sizes["pair-order.npy"]
        table.append([
            env, cost["device_memory"]["name"], str(cost["batch_pairs"]), str(pairs),
            _num(cost["total_seconds"], 1), _num(cost["pair_seconds"], 1),
            _num(cost["corpus_seconds"], 1), _num(cost["pairs_per_second"], 0),
            _num(cost["peak_rss_mib"], 0), _num(sum(sizes.values()) / 2**20, 1),
            str(arrays // max(pairs, 1)),
        ])  # fmt: skip
    return table


def _sentences(summaries: dict[str, Any]) -> dict[str, list[str]]:
    """Findings at checkpoints that do not depend on the controls, each with its numbers from the
    summaries. Checkpoints are named when a statement needs one; none is singled out."""
    observed: list[str] = []
    stable: list[str] = []
    conceals: list[str] = []
    for env, s in summaries.items():
        scalar = s["pairs"]["scalar"]
        budgets = s["budgets"]
        low, high = 0, len(budgets) - 1
        observed.append(
            f"**{env}, contributions.** Coordinates agree in sign on "
            f"{_num(scalar['sign_agreement']['all']['mean'])} of dimensions (independent, "
            f"symmetric signs would give 0.5); negative mass is "
            f"{_num(scalar['negative_share']['all']['mean'])} of absolute mass. Positive mass "
            f"spreads over about "
            f"{_num(scalar['effective_positive_dims']['all']['mean'], 0)} effective dimensions, "
            f"and 90% of it takes {_num(scalar['dims_for_90']['all']['mean'], 0)}."
        )
        observed.append(
            f"**{env}, magnitude source.** `log|c| = log|q| + log|d|` splits its variance into "
            f"{_num(scalar['log_variance_query_share']['all']['mean'])} query, "
            f"{_num(scalar['log_variance_document_share']['all']['mean'])} document, and "
            f"{_num(scalar['log_variance_covariance_share']['all']['mean'])} covariance. The top "
            f"50 contributions sit at the {_num(scalar['top_query_percentile']['all']['mean'])} "
            f"percentile of |q| and {_num(scalar['top_document_percentile']['all']['mean'])} of "
            f"|d| (0.5 would be a typical coordinate)."
        )
        agree = s["agreement"]
        if agree["mean_jaccard"]["multi"][0]["n"]:
            conceals.append(
                f"**{env}.** Averaging cancels "
                f"{_num(agree['aggregate_cancelled_share']['multi']['mean'])} of the absolute "
                f"contribution mass and removes the variance between positives, which is "
                f"{_num(agree['aggregate_within_positive_variance_share']['multi']['mean'])} of "
                f"the second moment. The mean opposes at least one positive on "
                f"{_num(agree['aggregate_flipped_dims']['multi']['mean'], 0)} dimensions. From "
                f"k = {budgets[low]} to k = {budgets[high]} its mask holds "
                f"{_num(agree['aggregate_mean_overlap']['multi'][low]['mean'])} to "
                f"{_num(agree['aggregate_mean_overlap']['multi'][high]['mean'])} of a positive's "
                f"own mask, and "
                f"{_num(agree['aggregate_min_overlap']['multi'][low]['mean'])} to "
                f"{_num(agree['aggregate_min_overlap']['multi'][high]['mean'])} for the "
                f"least-served positive. Those overlaps rise with k partly by construction."
            )
        else:
            conceals.append(f"**{env}.** No multi-positive query was analyzed; nothing to conceal.")
        stability = s["corpus"]["stability"]
        below = [
            row["k"]
            for row in stability["per_budget"]
            if row["spearman"]["min"] is not None and row["spearman"]["min"] < STABLE_SPEARMAN
        ]
        stable.append(
            f"**{env}.** Among the checkpoints, the relevant-document frequency profile falls "
            f"below this report's rule (lowest of {stability['draws']} split-half Spearmans at "
            f"least {STABLE_SPEARMAN}) at "
            + (f"k = {', '.join(str(k) for k in below)}" if below else "none of them")
            + f". Mean contribution per dimension: "
            f"{_range(stability['mean_contribution_spearman'])}. High stability here does not "
            f"separate relevant documents from coordinates; the excess profile below does."
        )
    return {"observed": observed, "stable": stable, "conceals": conceals}


def render(root: Path) -> tuple[str, dict[str, str], dict[str, str]]:
    run, inputs, commit, dirty, summaries = load(root)
    differences = p3_controls.load_differences(root)
    settings = inputs["settings"]
    tables = {
        "inputs": _inputs_table(inputs, summaries),
        "coverage": _coverage_table(inputs, summaries),
        "strata": _strata_table(summaries),
        "decomposition": _decomposition_table(summaries),
        "budgets": _budget_table(summaries),
        "agreement": _agreement_table(summaries),
        "conflicts": _conflict_table(summaries),
        "aggregation": _aggregate_table(summaries),
        "recurrence": _recurrence_table(summaries),
        "groups": _group_table(summaries),
        "components": _component_table(summaries),
        "stability": _stability_table(summaries),
        "findings": _finding_table(summaries),
        "grades": _grade_table(summaries),
        "nonrelevant": _nonrelevant_table(summaries),
        "cost": _cost_table(summaries),
    }
    figures = _figures(summaries)
    figures.update(p3_controls.figures(summaries))
    tables.update(p3_controls.tables(inputs, summaries, differences))
    words = _sentences(summaries)
    control_words = p3_controls.sentences(summaries)
    identity = [
        f"- {env}: largest `|Σ c_j - dot(q, d)|` {s['identities']['max_decomposition_error']:.2e} "
        f"over {s['identities']['pairs']} pairs; largest "
        f"`|weighted_mean(q * d) - q * weighted_mean(d)|` "
        f"{s['identities']['max_aggregation_identity_error']:.2e}."
        for env, s in summaries.items()
    ]
    lines = [
        "# P3: what dimension-level contributions reveal about known relevant pairs",
        "",
        "Exploratory and descriptive. Protocol: in-domain, training queries and training judgments",
        "only. This is not a benchmark: no selector, retrieval method, or ranking is evaluated.",
        "Unjudged documents appear only as a labeled reference, never as negatives, and the corpus",
        "is never ranked. Findings describe these training pairs and do not claim to hold for",
        "unseen queries.",
        "",
        f"- analyze-representations `{run}` (commit `{commit[:12]}`"
        + (", dirty working tree" if dirty else "")
        + ")",
        f"- Settings: {_query_bound(settings['max_queries'])} per environment, every k from 1 to "
        f"1024, summary checkpoints {settings['budgets']}, "
        f"{settings['reference_samples']} reference samples per query, at most "
        f"{settings['max_query_pairs']} document pairs per query, "
        f"{settings['stability_draws']} split-half draws, "
        f"{settings['across_pairs']} across-query pairs, seed {settings['seed']}.",
        *_status(inputs, summaries),
        "",
        "## What is measured, and what is not",
        "",
        "For a query `q` and a relevant document `d`, `c_j = q_j * d_j` and `Σ c_j = dot(q, d)`.",
        "Keeping the `k` largest `c_j` gives the largest unnormalized inner product any mask of",
        "exactly `k` dimensions can give that pair. Three things are different:",
        "",
        "1. **Maximizing similarity to one known relevant document**: measured.",
        "2. **Preserving the original similarity**: measured.",
        "3. **Improving a ranking**: not measured, and not implied by 1 or 2. A mask that raises "
        "one pair's score may raise competing documents' scores more.",
        "",
        "## Representations available",
        "",
        "The stored vectors are the only representation here: the L2-normalized last-token pooled",
        "output of frozen Qwen3-Embedding-0.6B, 1024 wide, with the instruction prefix on queries",
        "and none on documents, and stored as float16. Token embeddings and intermediate encoder",
        "states are not persisted by the embedding pipeline, so they are out of scope, and nothing",
        "here assumes their coordinates correspond to these. Measured norms are in the",
        "decomposition table below. Embeddings are opened read-only; masks are never applied",
        "to a stored matrix, and masked queries are not re-normalized.",
        "",
        *_markdown(tables["inputs"]),
        "",
        *p3_controls.provenance_lines(tables, differences),
        "",
        "## Judgment coverage",
        "",
        *_markdown(tables["coverage"]),
        "",
        *_markdown(tables["strata"]),
        "",
        "Judged nonrelevant (grade 0) and unjudged documents are different things. Reference",
        "documents below are unjudged for their query, not confirmed negatives; some may be",
        "relevant, which would pull a difference from relevant documents toward zero.",
        "",
        "## 1. What was directly observed",
        "",
        "Summaries are query-balanced (mean within a query, then across queries) unless a column",
        "says pair-weighted; ranges are the 10th to 90th percentiles of query values.",
        "",
        *[f"- {line}" for line in words["observed"]],
        "",
        "Identities, checked on every pair:",
        "",
        *identity,
        "",
        *_figure_lines(figures, "cumulative"),
        "",
        *_markdown(tables["decomposition"]),
        "",
        *p3_controls.observed_lines(figures, tables, control_words),
        "",
        "### Relevant-document masks at checkpoints",
        "",
        "Removing a dimension removes its contribution: the top-k ordering drops negative",
        "contributions first and positive mass last, so `max - full` is negative mass removed",
        "less positive mass removed. These rows are samples of the curves above, not substitutes.",
        "",
        *_markdown(tables["budgets"]),
        "",
        *_markdown(tables["agreement"]),
        "",
        *_figure_lines(figures, "conflict"),
        "",
        *_markdown(tables["conflicts"]),
        "",
        "### Relevance grades",
        "",
        *_grade_note(summaries),
        "",
        *_markdown(tables["grades"]),
        "",
        "### Recurrence across the training queries",
        "",
        *_figure_lines(figures, "frequency"),
        "",
        *_figure_lines(figures, "distribution"),
        "",
        *_markdown(tables["recurrence"]),
        "",
        "Groups: queries are stratified by number of positives only; no clustering was run. Each",
        "row below describes the query-by-dimension selection matrix at one checkpoint.",
        "",
        *_markdown(tables["components"]),
        "",
        *_markdown(tables["groups"]),
        "",
        "Dimension identities are not interpreted here: nothing in these data gives a coordinate a",
        "meaning.",
        "",
        "### A limited reference: judged nonrelevant documents",
        "",
        *_nonrelevant_note(summaries),
        "",
        *_markdown(tables["nonrelevant"]),
        "",
        "## 2. What is stable within the analyzed training data",
        "",
        "Stable means stable across disjoint halves of these training queries, a query always",
        "keeping all of its positives. It says nothing about unseen queries.",
        "",
        *[f"- {line}" for line in words["stable"]],
        *[f"- {line}" for line in control_words["stable"]],
        "",
        *_figure_lines(figures, "stability"),
        "",
        *_markdown(tables["stability"]),
        "",
        *_markdown(tables["findings"]),
        "",
        "## 3. What aggregation conceals",
        "",
        "`weighted_mean(q * d) = q * weighted_mean(d)` holds exactly, with the repository's oracle",
        "weights `2**grade - 1`, so averaging contributions and averaging documents are the same",
        "operation. This compares masks and pair scores; it does not benchmark Learning-to-Select.",
        "",
        *[f"- {line}" for line in words["conceals"]],
        "",
        *p3_controls.conceal_lines(figures, tables, control_words),
        "",
        *_figure_lines(figures, "cancelled"),
        "",
        *_markdown(tables["aggregation"]),
        "",
        "## 4. Questions that remain unanswered",
        "",
        "- How much of a difference from reference documents comes from relevance, and how much",
        "  from relevant documents simply lying closer to the query and to one another. The score",
        "  and document-cosine rows above show those gaps; this design does not remove them.",
        "- Whether the same differences appear against documents as close to the query as the",
        "  positives are. Matching on score within a bounded sample needs no ranking of the",
        "  corpus; that control has not been run here.",
        "- Whether a higher pair score under a mask changes any ranking. Nothing here compares a",
        "  document's score with a competitor's.",
        "- Whether a pattern is predictable from a query embedding for unseen queries. All of it",
        "  is computed with the labels of the queries it describes.",
        "- What judged nonrelevant documents would show, where none exists in these judgments.",
        "- What the token-level and intermediate representations would show. They are not stored.",
        "",
        "## 5. Follow-up hypotheses these observations motivate",
        "",
        "Hypotheses to test, not recommendations. No method follows from a descriptive pattern.",
        "",
        "- *Closeness versus relevance.* If the difference from reference documents mostly",
        "  reflects closeness to the query, it should shrink against reference documents matched",
        "  on score.",
        "  Testing it needs a design for choosing them that does not call them negatives.",
        "- *Excess profile.* If the excess of relevant over reference selection frequency is",
        "  query-independent structure, it should repeat across disjoint queries of one corpus.",
        "  The split-half curves bound this in-sample only.",
        "- *Conflict and ranking.* If sign conflicts mark documents reached through different",
        "  coordinates, then masks that serve one positive may hurt another. Testing it needs a",
        "  ranking measurement, which this experiment excludes.",
        "",
        "## Offline cost",
        "",
        "All of it is offline; this analysis has no online path. Each row is one run on the device",
        "it names, and costs vary by machine and load.",
        "",
        *_markdown(tables["cost"]),
        "",
        "Wall time by stage. `read`, `transfer`, `synchronized compute`, `write`, and `plan` are",
        "only the batched kernels, on the pair orderings and the control pass. The total also",
        "holds loading and hashing, the CPU per-pair statistics and corpus summaries, the all-k",
        "summaries, and writing the JSON, so those columns do not add up to it.",
        "",
        *_markdown(tables["wall-time"]),
        "",
        *_markdown(tables["control-cost"]),
        "",
        "Peak RSS is the process peak so far, so later environments include earlier ones, and it",
        "counts pages of memory-mapped files: the read-only embeddings and the staged arrays. The",
        "system can drop those pages, so RSS is not all allocation. Hashing streams a file and",
        "adds no RSS. Device memory holds batches only, never the corpus.",
        "",
        "## Limitations",
        "",
        "- Training data only, so every statement is in-sample by design.",
        "- Stored vectors are float16, and the decomposition uses them as stored.",
        "- Percentile ranges describe query values, not uncertainty of the means.",
        "- `Spearman` between profiles over 1024 dimensions counts dimensions as observations; "
        "they are not independent, so no p-value is given.",
        "- Sign-conflict coverage can only grow as positives are added; the cancelled share need",
        "  not, and can fall. Both depend on the count, so compare them within a stratum of",
        "  positive counts, not across environments with different counts.",
        "- `chance` assumes independent uniform subsets. Coordinate magnitudes are not uniform, so",
        "  it is a reference point and not a null model of these data.",
        "- Scaling by `mean|q_j| * mean|d_j|` removes one per-dimension size and nothing else: it",
        "  keeps within-query and sign structure, and is not a model of the coordinates.",
        "- Reference documents are a uniform corpus sample excluding the query's judged documents.",
        "  Only training judgments are read, so a reference document's status is otherwise open.",
        "",
        "## Reproduce",
        "",
        "```bash",
        *p3_controls.reproduce_lines(bool(_provisional(inputs, summaries)), inputs, settings),
        "```",
    ]
    return (
        "\n".join(lines) + "\n",
        {name: _csv(table) for name, table in tables.items()},
        figures,
    )


def _status(inputs: dict[str, Any], summaries: dict[str, Any]) -> list[str]:
    found = []
    foreign = _provisional(inputs, summaries)
    if foreign:
        found.append(
            f"- **Validation only, canonical results pending ({', '.join(foreign)}):** these "
            "inputs are not the published F2 artifacts. Do not cite these numbers."
        )
    bounded = [
        env
        for env, s in summaries.items()
        if s["coverage"]["selected_queries"] < s["coverage"]["training_queries"]
    ]
    if bounded:
        found.append(f"- **Bounded pilot ({', '.join(bounded)}):** a seeded subset of queries.")
    absent = sorted(set(("scifact", "nfcorpus", "msmarco")) - set(summaries))
    if absent:
        found.append(f"- **Not analyzed yet:** {', '.join(absent)}.")
    return found


def _query_bound(limit: int | None) -> str:
    return "all training queries" if limit is None else f"at most {limit} training queries"


def _provisional(inputs: dict[str, Any], summaries: dict[str, Any]) -> list[str]:
    return [env for env in summaries if inputs[env]["provenance"] != F2_PROVENANCE]


def _grade_note(summaries: dict[str, Any]) -> list[str]:
    one_grade = [env for env, s in summaries.items() if len(s["grades"]) < 2]
    if not one_grade:
        return ["Each query counts once within a grade."]
    return [
        f"Only one relevance grade among the analyzed judgments of {', '.join(one_grade)}, so",
        "variability associated with grades cannot be assessed there.",
    ]


def _nonrelevant_note(summaries: dict[str, Any]) -> list[str]:
    absent = [env for env, s in summaries.items() if not s["nonrelevant"]["pairs"]]
    if not absent:
        return ["Per pair, and kept apart from the positives; these are not ranked or compared."]
    return [
        f"No judged nonrelevant training document in {', '.join(absent)}: every judgment analyzed",
        "there is positive, so that reference is unavailable and positive-only patterns stay",
        "unconfirmed as relevance-specific.",
    ]


def write(root: Path, output: Path | None = None) -> Path:
    """Write the report; a run on other inputs never lands in the canonical ``reports/P3``."""
    _, inputs, _, _, summaries = load(root)
    foreign = bool(_provisional(inputs, summaries))
    target = output or Path("reports") / ("P3-spike" if foreign else "P3")
    if foreign and target.name == "P3":
        raise ContractError(f"a run on non-F2 inputs must not be written to {target}")
    text, csvs, figures = render(root)
    (target / "tables").mkdir(parents=True, exist_ok=True)
    (target / "figures").mkdir(parents=True, exist_ok=True)
    (target / "README.md").write_text(text, encoding="utf-8")
    for name, content in csvs.items():
        (target / "tables" / f"{name}.csv").write_text(content, encoding="utf-8")
    for name, content in figures.items():
        (target / "figures" / f"{name}.svg").write_text(content, encoding="utf-8")
    return target / "README.md"


# --- Figures: static SVG with no plotting dependency, every value from the summaries ------------


def _figures(summaries: dict[str, Any]) -> dict[str, str]:
    envs = list(summaries)
    figures: dict[str, str] = {}
    figures["cumulative"] = _lines(
        "How contributions accumulate, largest first",
        "Median pair score over the query's positive mass, 10th to 90th percentile band; "
        "the curve ends at the full score.",
        "Retained dimensions k (log scale)",
        "Score / positive mass",
        [
            _Series(env, s["curve_k"], s["curve"]["p50"], s["curve"]["p10"], s["curve"]["p90"])
            for env, s in summaries.items()
        ],
        log_x=True,
    )
    for env, s in summaries.items():
        table = s["corpus"]["dimension_contributions"]
        figures[f"distribution-{env}"] = _lines(
            f"Contribution of each dimension across queries ({env})",
            "Per query, the mean c_j over positives; median and 10th to 90th percentile, "
            "dimensions ordered by median.",
            "Dimension rank by median contribution",
            "Query-mean c_j",
            [_Series("Median", table["rank"], table["p50"], table["p10"], table["p90"])],
        )
        per_budget = s["corpus"]["per_budget"]
        step = max(1, (len(per_budget) - 1) // (FREQUENCY_BUDGETS - 1))
        chosen = per_budget[::step][:FREQUENCY_BUDGETS]
        figures[f"frequency-{env}"] = _lines(
            f"How often is each dimension selected? ({env})",
            "Query-balanced frequency in pairwise top-k masks, dimensions sorted per budget; "
            "uniform selection would be flat at k/1024.",
            "Dimensions by frequency (percent of 1024)",
            "Selection frequency",
            [
                _Series(
                    f"k = {row['k']}",
                    [
                        100 * i / (len(row["sorted_profile"]) - 1)
                        for i in range(len(row["sorted_profile"]))
                    ],
                    row["sorted_profile"],
                )
                for row in chosen
            ],  # fmt: skip
        )
    strata = [row["positives"] for row in summaries[envs[0]]["coverage"]["strata"]][1:]
    figures["conflict"] = _bars(
        "How many dimensions change sign across a query's positives?",
        "Mean share of the 1024 dimensions with both a positive and a negative contribution.",
        "Positives per query",
        "Share of dimensions",
        strata,
        {env: _by_stratum(s, "sign_conflict_share", strata) for env, s in summaries.items()},
    )
    figures["cancelled"] = _bars(
        "How much contribution mass does averaging cancel?",
        "Mean share of absolute contribution mass lost when a query's positives are averaged.",
        "Positives per query",
        "Share of mass cancelled",
        strata,
        {env: _by_stratum(s, "aggregate_cancelled_share", strata) for env, s in summaries.items()},
    )
    return figures


def _by_stratum(summary: dict[str, Any], name: str, strata: Sequence[str]) -> list[float | None]:
    return [summary["agreement"][name][f"positives {label}"]["mean"] for label in strata]


def _figure_lines(figures: dict[str, str], stem: str) -> list[str]:
    names = [n for n in sorted(figures) if n == stem or n.startswith(f"{stem}-")]
    return [f"![{name}](figures/{name}.svg)" for name in names]
