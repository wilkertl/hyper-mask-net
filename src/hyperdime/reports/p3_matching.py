"""Report of the score-matching coverage pilot, generated from one run; no number is typed here."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from hyperdime.contracts.manifests import read_manifest
from hyperdime.reports.e0 import _markdown


def _num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _spread(item: dict[str, Any]) -> str:
    if not item.get("n"):
        return "n/a"
    return f"{_num(item['mean'])} ({_num(item['p10'])} to {_num(item['p90'])}), n={item['n']}"


def _coverage(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "selected queries", "pairs", "unweighted complete rate",
              "unweighted pair rate", "complete rate by repeat", "complete in every repeat",
              "complete in no repeat", "weighted complete rate (SE)",
              "weighted pair rate"]]  # fmt: skip
    for env, s in summaries.items():
        sample, weighted = s["sample"], s["weighted"]
        rate = weighted["complete_rate"]
        table.append([
            env, str(sample["queries"]), str(sample["pairs"]),
            _num(sample["complete_rate_unweighted"]), _num(sample["pair_rate_unweighted"]),
            " / ".join(_num(v) for v in sample["complete_rate_by_repeat_unweighted"]),
            str(sample["complete_in_all_repeats"]), str(sample["complete_in_no_repeat"]),
            f"{_num(rate['estimate'])} ({_num(rate['standard_error'])})",
            _num(weighted["pair_rate"]),
        ])  # fmt: skip
    return table


def _strata(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "positives per query", "eligible queries", "selected",
              "sampling weight", "complete rate", "complete by repeat", "pair rate"]]  # fmt: skip
    for env, s in summaries.items():
        for row in s["strata"]:
            table.append([
                env, row["positives"], str(row["eligible_queries"]), str(row["selected_queries"]),
                _num(row["sampling_weight"]), _num(row["complete_rate"]),
                " / ".join(_num(v) for v in row["complete_rate_by_repeat"]) or "n/a",
                _num(row["pair_rate"]),
            ])  # fmt: skip
    return table


def _reasons(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "query-repeat outcome", "count"]]
    for env, s in summaries.items():
        for reason, count in s["reasons"].items():
            table.append([env, reason, str(count)])
    return table


def _algorithm(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "exact checks disagreeing", "complete query-repeats, exact",
              "complete, greedy nearest-unused", "greedy false failures"]]  # fmt: skip
    for env, s in summaries.items():
        greedy = s["nearest_unused"]
        table.append([
            env, str(s["exact_check_disagreements"]), str(greedy["exact_complete_query_repeats"]),
            str(greedy["greedy_complete_query_repeats"]),
            str(greedy["query_repeats_greedy_falsely_failed"]),
        ])  # fmt: skip
    return table


def _scores(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "matched positive score", "matched reference score",
              "unmatched positive score", "reference minus positive", "absolute residual",
              "largest absolute residual", "sample mean score per query"]]  # fmt: skip
    for env, s in summaries.items():
        c = s["scores"]
        table.append([
            env, _spread(c["matched_positive"]), _spread(c["matched_reference"]),
            _spread(c["unmatched_positive"]), _spread(c["residual_reference_minus_positive"]),
            _spread(c["absolute_residual"]), _num(c["largest_absolute_residual"], 6),
            _spread(c["sample_mean_score_per_query"]),
        ])  # fmt: skip
    return table


def _deciles(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "positive score decile", "score range", "pairs", "pair rate"]]
    for env, s in summaries.items():
        for index, row in enumerate(s["scores"]["pair_rate_by_positive_score_decile"], 1):
            table.append([
                env, str(index), f"{_num(row['low'])} to {_num(row['high'])}", str(row["pairs"]),
                _num(row["pair_rate"]),
            ])  # fmt: skip
    return table


def _selection(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "group", "queries", "weighted share", "positives per query",
              "cosine among positives"]]  # fmt: skip
    for env, s in summaries.items():
        for row in s["selection"]:
            table.append([
                env, row["group"], str(row["queries"]), _num(row["weighted_share"]),
                _spread(row["positives"]), _spread(row["positive_cosine"]),
            ])  # fmt: skip
    return table


def _cosine(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "queries complete at least once", "among positives",
              "among matched references", "among random references, same queries",
              "positives minus matched", "positives minus random",
              "matched by repeat"]]  # fmt: skip
    for env, s in summaries.items():
        c = s["cosine"]
        table.append([
            env, str(c["queries_complete_at_least_once"]), _spread(c["positive"]),
            _spread(c["matched_reference"]), _spread(c["random_reference"]),
            _spread(c["positive_minus_matched"]), _spread(c["positive_minus_random"]),
            " / ".join(_num(v) for v in c["matched_by_repeat"]),
        ])  # fmt: skip
    return table


def _cost(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "device", "seconds", "seconds per query-repeat", "read", "transfer",
              "compute", "match", "write", "peak device MiB", "peak RSS MiB (process so far)",
              "artifact KiB",
              "matching alone, all eligible queries, 3 repeats, seconds"]]  # fmt: skip
    for env, s in summaries.items():
        c, phase = s["cost"], s["cost"]["phase_seconds"]
        eligible = sum(row["eligible_queries"] for row in s["strata"])
        table.append([
            env, c["device_memory"]["name"], _num(c["total_seconds"], 1),
            _num(c["seconds_per_query_repeat"], 4),
            *[_num(phase.get(name, 0.0), 1) for name in ("read", "transfer", "compute", "match",
                                                         "write")],
            _num(c["device_memory"].get("peak_allocated_mib"), 0), _num(c["peak_rss_mib"], 0),
            _num(sum(c["artifact_bytes"].values()) / 1024, 0),
            _num(c["seconds_per_query_repeat"] * eligible * s["settings"]["repeats"], 0),
        ])  # fmt: skip
    return table


def _coverage_lines(summaries: dict[str, Any]) -> list[str]:
    lines = []
    for env, s in summaries.items():
        rate = s["weighted"]["complete_rate"]["estimate"]
        if rate is None:
            lines.append(f"- **{env}.** No weighted estimate: a stratum has no defined value.")
        elif rate < 0.5:
            lines.append(
                f"- **{env}: limited coverage.** About {rate:.3f} of eligible multi-positive "
                "queries match completely. Matched queries are then a selected subset, and a "
                "comparison on them would not describe the others."
            )
        else:
            lines.append(
                f"- **{env}.** About {rate:.3f} of eligible multi-positive queries match "
                "completely; the selection table shows which are left out."
            )
    return lines


def _csv(table: list[list[str]]) -> str:
    out = io.StringIO()
    csv.writer(out, lineterminator="\n").writerows(table)
    return out.getvalue()


def write_report(run: Path, target: Path) -> None:
    """Write README and tables for one pilot run; the same run gives the same bytes."""
    inputs = json.loads((run / "inputs.json").read_text(encoding="utf-8"))
    settings = inputs["settings"]
    summaries = {
        env: json.loads((run / env / "summary.json").read_text(encoding="utf-8"))
        for env in settings["envs"]
    }
    manifest = read_manifest(run)
    tables = {
        "coverage": _coverage(summaries),
        "strata": _strata(summaries),
        "reasons": _reasons(summaries),
        "algorithm": _algorithm(summaries),
        "scores": _scores(summaries),
        "score-deciles": _deciles(summaries),
        "selection": _selection(summaries),
        "cosine": _cosine(summaries),
        "cost": _cost(summaries),
    }
    (target / "tables").mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        (target / "tables" / f"{name}.csv").write_text(_csv(table), encoding="utf-8")
    lines = [
        "# P3: coverage pilot of a score-matched reference control",
        "",
        "Exploratory and descriptive, training judgments only. This run measures whether each",
        "relevant document can be paired with an unjudged corpus document of a similar score. It",
        "computes no mask, no ordering, and no all-k curve, and it ranks no corpus: scores exist",
        "only for a bounded seeded sample. Reference documents are unjudged, not negatives, and",
        "high-scoring ones may be relevant. Published P3 results are not changed by this pilot.",
        "",
        f"- Run `{run.name}` (commit `{manifest.git_commit[:12]}`, device `{manifest.device}`)",
        f"- At most {settings['max_queries']} multi-positive queries per environment, "
        f"{settings['repeats']} seeded samples, caliper +/- {settings['caliper']}, sample of "
        f"min({settings['sample_size']}, half the corpus) documents, fixed before seeing results.",
        "",
        "## Coverage",
        "",
        "A query is complete in a repeat when all its positives match distinct documents. A",
        "weighted rate is a stratified estimate over all eligible multi-positive queries, with",
        "weights in the strata table; the unweighted rates describe the selection only and are",
        "not population coverage. The standard error covers selecting queries and not sampling",
        "documents; the by-repeat columns show the second.",
        "",
        *_coverage_lines(summaries),
        "",
        *_markdown(tables["coverage"]),
        "",
        *_markdown(tables["strata"]),
        "",
        "## Why groups are incomplete",
        "",
        "The matching is exact, so an incomplete group is never an algorithm's failure. The",
        "two exact methods are compared on every group. Greedy nearest-unused is shown only to",
        "count the complete groups it would have missed.",
        "",
        *_markdown(tables["reasons"]),
        "",
        *_markdown(tables["algorithm"]),
        "",
        "## Scores and residuals",
        "",
        *_markdown(tables["scores"]),
        "",
        "Pair rate by the positive's score, over pairs of the selection:",
        "",
        *_markdown(tables["score-deciles"]),
        "",
        "## Selection",
        "",
        "Queries that match are not a random part of the eligible ones. Anything later computed",
        "on matched queries describes this group only.",
        "",
        *_markdown(tables["selection"]),
        "",
        "## Document-document cosine on complete groups",
        "",
        "Score is matched, and cosine among a group's documents is not. Random references are the",
        "same number of eligible documents from the same sample. The positives' cosine is for all",
        "queries complete at least once; differences are per query, and they are descriptive.",
        "",
        *_markdown(tables["cosine"]),
        "",
        "## Cost",
        "",
        "One run on the device named. The last column scales this run's seconds per query-repeat",
        "to every eligible multi-positive query; it is an extrapolation, covers matching only, and",
        "includes no all-k curves. An earlier estimate, made before any measurement, gave this",
        "pilot under 5 minutes and under 10 MB and a full curve run of 1 to 3 GB; treat the full",
        "figure as unverified. Peak RSS counts pages of memory-mapped files.",
        "",
        *_markdown(tables["cost"]),
        "",
    ]
    (target / "README.md").write_text("\n".join(lines), encoding="utf-8")
