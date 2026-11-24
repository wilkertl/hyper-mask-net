"""The P3 report's curves at every cardinality and its controls, from recorded outputs only.

Reads what ``representation_controls`` wrote. Statements report what the numbers are and what each
comparison holds fixed, whichever way they come out; none says a pattern is caused by relevance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hyperdime.artifacts import latest_artifact
from hyperdime.contracts.validation import ContractError
from hyperdime.reports.e0 import _markdown
from hyperdime.reports.p3_svg import Series, bars, lines

DIM = 1024
# A rule for the word "stable", fixed in ADR-0014; it describes these data and promises nothing
# about unseen queries.
STABLE_SPEARMAN = 0.8
Z95 = 1.96
SIZE_LABELS = {"sign_conflict_share": "sign conflict", "cancelled_share": "cancelled mass"}


def num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def spread(item: dict[str, Any], digits: int = 3) -> str:
    if not item["n"]:
        return "n/a"
    return f"{num(item['mean'], digits)} ({num(item['p10'], digits)} to {num(item['p90'], digits)})"


def spread_at(curve: dict[str, Any], k: int) -> str:
    """One checkpoint of a stored curve, as ``mean (p10 to p90)``."""
    if not curve["n"]:
        return "n/a"
    at = k - 1
    return f"{num(curve['mean'][at])} ({num(curve['p10'][at])} to {num(curve['p90'][at])})"


def load_differences(root: Path) -> dict[str, Any] | None:
    """The latest recorded comparison with spike inputs, if one was made on this root."""
    try:
        directory = latest_artifact(root, "evaluation/P3-inputs", "all")
    except ContractError:
        return None
    found: dict[str, Any] = json.loads((directory / "differences.json").read_text(encoding="utf-8"))
    return {"run": directory.name, "environments": found}


def _verification_table(inputs: dict[str, Any], summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "inputs", "payload hashes", "D", "documents", "train queries",
              "F2 cap on train and validation queries", "document norms (sampled)",
              "train query norms", "documents sampled for norms",
              "embeddings commit", "instruction", "encoding settings"]]  # fmt: skip
    for env in summaries:
        check = inputs[env]["verification"]
        low, high = check["document_norm_range"]
        q_low, q_high = check["train_query_norm_range"]
        table.append([
            env, inputs[env]["provenance"], "match" if check["payload_hashes_match"] else "differ",
            str(check["dim"]), str(check["documents"]),
            str(check["queries_per_split"].get("train", 0)),
            str(check.get("train_query_cap") or "none"), f"{low:.4f} to {high:.4f}",
            f"{q_low:.4f} to {q_high:.4f}", str(check["norm_sampled_documents"]),
            check["embedding_commit"][:12], check["instruction"] or "none",
            json.dumps(check["encoding_settings"], sort_keys=True),
        ])  # fmt: skip
    return table


def _difference_table(differences: dict[str, Any] | None) -> list[list[str]]:
    table = [["environment", "spike inputs", "identical", "differ", "train queries shared",
              "only F2", "only spike", "train judgments shared", "documents shared",
              "lowest document cosine", "lowest query cosine"]]  # fmt: skip
    if differences is None:
        return table
    for env, row in differences["environments"].items():
        if not row["spike_inputs"]:
            table.append([env, "none", *["n/a"] * 9])
            continue
        same = row["same"]
        table.append([
            env, "present", ", ".join(k for k, v in same.items() if v) or "none",
            ", ".join(k for k, v in same.items() if not v) or "none",
            str(row["train_queries"]["shared"]), str(row["train_queries"]["only_f2"]),
            str(row["train_queries"]["only_spike"]), str(row["train_judgments"]["shared"]),
            str(row["documents"]["shared"]), num(row["document_vectors"]["min_cosine"], 6),
            num(row["query_vectors"]["min_cosine"], 6),
        ])  # fmt: skip
    return table


def _curve_table(s: dict[str, Any]) -> list[list[str]]:
    """One row per ``k``: the full curve in a form anything can read. ``k = 0`` is labeled."""
    c = s["curves"]
    pair, agree = c["pair_curves"], c["agreement"]
    header = ["k", "note", "chance Jaccard", "floor Jaccard"]
    series: list[tuple[str, list[float | None]]] = []
    for name in ("similarity", "retained_positive_share", "removed_negative_share"):
        row = pair[name]["all"]
        for group in ("positive", "reference", "difference"):
            series.append((f"{name} {group} mean", row[group]["mean"]))
        series.append((f"{name} difference standard error", row["difference"]["se"]))
        series.append((f"{name} eligible queries", row["difference"]["queries"]))
        series.append((f"{name} positive p10", row["positive"]["p10"]))
        series.append((f"{name} positive p90", row["positive"]["p90"]))
        series.append((f"{name} repeats min", c["reference_repeats"][name]["min"]))
        series.append((f"{name} repeats max", c["reference_repeats"][name]["max"]))
    for name in ("jaccard", "scaled_jaccard"):
        row = agree[name]["multi"]
        for group in ("positive", "reference", "difference"):
            series.append((f"{name} {group} mean", row[group]["mean"]))
        series.append((f"{name} difference standard error", row["difference"]["se"]))
        series.append((f"{name} eligible queries", row["difference"]["queries"]))
        series.append((f"{name} difference p10", row["difference"]["p10"]))
        series.append((f"{name} difference p90", row["difference"]["p90"]))
        series.append((f"{name} reference repeats min", c["reference_repeats"][name]["min"]))
        series.append((f"{name} reference repeats max", c["reference_repeats"][name]["max"]))
        series.append((f"{name} difference halves min", c["delta_stability"][name]["min"]))
        series.append((f"{name} difference halves max", c["delta_stability"][name]["max"]))
    for name, curve in c["across_query"].items():
        series.append((f"different queries {name} mean", curve["mean"]))
    table = [header + [name for name, _ in series]]
    empty = c["empty_mask_reference"]
    zero = {
        f"{name} positive mean": empty[name]
        for name in ("similarity", "retained_positive_share", "removed_negative_share")
    }
    table.append(["0", "empty-mask reference", "", ""] +
                 [num(zero[name], 6) if name in zero else "" for name, _ in series])  # fmt: skip
    for at, k in enumerate(c["k"]):
        table.append(
            [str(k), "", num(c["chance_jaccard"][at], 6), num(c["floor_jaccard"][at], 6)]
            + [num(values[at], 6) for _, values in series]
        )
    return table


def _frequency_table(s: dict[str, Any]) -> list[list[str]]:
    frequency = s["curves"]["frequency"]
    names = ("max_positive", "max_reference", "positive_vs_magnitude", "reference_vs_magnitude",
             "positive_vs_reference", "excess_vs_magnitude", "total_variation")  # fmt: skip
    halves = ("split_half_positive", "split_half_reference", "split_half_excess")
    series: list[tuple[str, list[float | None]]] = []
    for variant in ("raw", "scaled"):
        row = frequency[variant]
        series += [(f"{variant} {name}", row[name]) for name in names]
        for name in halves:
            series.append((f"{variant} {name} mean", row[name]["mean"]))
            series.append((f"{variant} {name} min", row[name]["min"]))
    table = [["k"] + [name for name, _ in series]]
    for at, k in enumerate(s["curves"]["k"]):
        table.append([str(k)] + [num(values[at], 6) for _, values in series])
    return table


def _ladder_table(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "k", "chance", "floor", "different queries, reference documents",
              "same query, reference documents", "same query, relevant documents",
              "relevant minus reference (query p10 to p90)", "reference repeats (min to max)",
              "difference over query halves (min to max)"]]  # fmt: skip
    for env, s in summaries.items():
        c = s["curves"]
        multi = c["agreement"]["jaccard"]["multi"]
        across = c["across_query"].get("reference_raw")
        repeats = c["reference_repeats"]["jaccard"]
        halves = c["delta_stability"]["jaccard"]
        for k in s["budgets"]:
            at = k - 1
            table.append([
                env, str(k), num(c["chance_jaccard"][at]), num(c["floor_jaccard"][at]),
                spread_at(across, k) if across else "n/a",
                spread_at(multi["reference"], k), spread_at(multi["positive"], k),
                spread_at(multi["difference"], k),
                f"{num(repeats['min'][at])} to {num(repeats['max'][at])}",
                f"{num(halves['min'][at])} to {num(halves['max'][at])}",
            ])  # fmt: skip
    return table


def _scaled_table(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "k", "different queries, reference", "same query, reference",
              "same query, relevant", "relevant minus reference"]]  # fmt: skip
    for env, s in summaries.items():
        c = s["curves"]
        multi = c["agreement"]["scaled_jaccard"]["multi"]
        across = c["across_query"].get("reference_scaled")
        for k in s["budgets"]:
            table.append([
                env, str(k), spread_at(across, k) if across else "n/a",
                spread_at(multi["reference"], k), spread_at(multi["positive"], k),
                spread_at(multi["difference"], k),
            ])  # fmt: skip
    return table


def _strata_table(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "positives per query", "k", "queries", "same query, relevant",
              "same query, matched reference", "relevant minus reference",
              "mean difference, 95% interval over queries"]]  # fmt: skip
    for env, s in summaries.items():
        agree = s["curves"]["agreement"]["jaccard"]
        for subset, row in _strata(agree):
            for k in s["budgets"]:
                table.append([
                    env, subset.removeprefix("positives "), str(k), str(row["positive"]["n"]),
                    spread_at(row["positive"], k), spread_at(row["reference"], k),
                    spread_at(row["difference"], k), _interval(row["difference"], k - 1),
                ])  # fmt: skip
    return table


def _strata_curve_table(s: dict[str, Any]) -> list[list[str]]:
    """One row per ``k``: relevant minus reference agreement in each positive-count stratum."""
    series: list[tuple[str, list[float | None]]] = []
    for name in ("jaccard", "scaled_jaccard"):
        for subset, row in _strata(s["curves"]["agreement"][name]):
            label = f"{name} {subset}"
            series.append((f"{label} relevant mean", row["positive"]["mean"]))
            series.append((f"{label} reference mean", row["reference"]["mean"]))
            series.append((f"{label} difference mean", row["difference"]["mean"]))
            series.append((f"{label} difference standard error", row["difference"]["se"]))
            series.append((f"{label} eligible queries", row["difference"]["queries"]))
    table = [["k"] + [name for name, _ in series]]
    for at, k in enumerate(s["curves"]["k"]):
        table.append([str(k)] + [num(values[at], 6) for _, values in series])
    return table


def _interval(row: dict[str, Any], at: int) -> str:
    """A mean difference with its 95% interval over queries; empty when it is undefined."""
    mean, se = row["mean"][at], row["se"][at]
    if mean is None or se is None:
        return ""
    return f"{mean:.3f} ({mean - Z95 * se:.3f} to {mean + Z95 * se:.3f})"


def _uncertainty_sentence(env: str, row: dict[str, Any]) -> str | None:
    """How many cardinalities have an interval that leaves zero; ``k = D`` never does."""
    known = [
        (at + 1, mean, se)
        for at, (mean, se) in enumerate(zip(row["mean"], row["se"], strict=True))
        if mean is not None and se is not None
    ]
    clear = [k for k, mean, se in known if abs(mean) > Z95 * se]
    if not known:
        return None
    where = f", first {clear[0]}, last {clear[-1]}" if clear else ""
    return (
        f"**{env}, uncertainty.** Over {row['n']} multi-positive queries, the 95% interval of the "
        f"mean within-query difference leaves zero at {len(clear)} of {len(known)} k{where}. It "
        "reflects query sampling only, as a normal approximation, and says nothing about queries "
        "outside these training sets. These intervals are pointwise and exploratory: neighboring "
        "k are strongly dependent, so the count is not that many independent confirmations, and "
        "a difference from zero is not evidence of relevance structure by itself."
    )


def _strata(rows: dict[str, Any]) -> list[tuple[str, Any]]:
    """Positive-count strata in numeric order, which JSON's sorted keys do not keep."""
    found = [(name, row) for name, row in rows.items() if name.startswith("positives ")]
    return sorted(
        found, key=lambda item: int(item[0].removeprefix("positives ").split("-")[0].rstrip("+"))
    )


def _pair_table(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "k", "raw similarity, relevant", "raw similarity, reference",
              "positive mass retained, relevant", "positive mass retained, reference",
              "negative mass removed, relevant", "negative mass removed, reference"]]  # fmt: skip
    for env, s in summaries.items():
        pair = s["curves"]["pair_curves"]
        for k in s["budgets"]:
            table.append([
                env, str(k),
                spread_at(pair["similarity"]["all"]["positive"], k),
                spread_at(pair["similarity"]["all"]["reference"], k),
                spread_at(pair["retained_positive_share"]["all"]["positive"], k),
                spread_at(pair["retained_positive_share"]["all"]["reference"], k),
                spread_at(pair["removed_negative_share"]["all"]["positive"], k),
                spread_at(pair["removed_negative_share"]["all"]["reference"], k),
            ])  # fmt: skip
    return table


def _magnitude_table(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "coordinates", "k", "relevant vs magnitude", "reference vs magnitude",
              "relevant vs reference", "selection placed differently", "excess vs magnitude",
              "split-half relevant, mean (min)", "split-half reference, mean (min)",
              "split-half excess, mean (min)"]]  # fmt: skip
    for env, s in summaries.items():
        frequency = s["curves"]["frequency"]
        for variant in ("raw", "scaled"):
            row = frequency[variant]
            for k in s["budgets"]:
                at = k - 1
                table.append([
                    env, variant, str(k), num(row["positive_vs_magnitude"][at]),
                    num(row["reference_vs_magnitude"][at]), num(row["positive_vs_reference"][at]),
                    num(row["total_variation"][at]), num(row["excess_vs_magnitude"][at]),
                    _half(row["split_half_positive"], at), _half(row["split_half_reference"], at),
                    _half(row["split_half_excess"], at),
                ])  # fmt: skip
    return table


def _half(item: dict[str, Any], at: int) -> str:
    return f"{num(item['mean'][at])} ({num(item['min'][at])})"


def _frequency_strata_table(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "positives per query", "queries", "k", "stratum vs corpus",
              "stratum vs its own matched reference"]]  # fmt: skip
    for env, s in summaries.items():
        for row in s["curves"]["frequency"]["strata"]:
            if not row["queries"]:
                continue
            for k in s["budgets"]:
                table.append([
                    env, row["positives"], str(row["queries"]), str(k),
                    num(row["positive_vs_corpus"][k - 1]),
                    num(row["positive_vs_own_reference"][k - 1]),
                ])  # fmt: skip
    return table


def _scalar_table(summaries: dict[str, Any]) -> list[list[str]]:
    labels = {"score": "full score", "document_cosine": "cosine between documents",
              "sign_conflict_share": "share of dims with a sign conflict",
              "cancelled_share": "mass cancelled by an unweighted mean"}  # fmt: skip
    table = [["statistic", "environment", "queries", "n", "relevant", "matched reference",
              "relevant minus reference"]]  # fmt: skip
    for name, label in labels.items():
        for env, s in summaries.items():
            for subset, row in s["curves"]["scalars"][name].items():
                table.append([
                    label, env, subset, str(row["difference"]["n"]), spread(row["positive"]),
                    spread(row["reference"]), spread(row["difference"]),
                ])  # fmt: skip
    return table


def _size_table(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "documents in the group", "queries with that many positives",
              "measure", "relevant", "matched reference", "relevant minus reference"]]  # fmt: skip
    for env, s in summaries.items():
        for row in s["curves"]["group_size"]:
            for name, label in SIZE_LABELS.items():
                item = row[name]
                table.append([
                    env, str(row["size"]), str(row["queries"]), label, spread(item["positive"]),
                    spread(item["reference"]), spread(item["difference"]),
                ])  # fmt: skip
    return table


def _consistency_table(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "|similarity(1024) - dot(q, d)|", "|retained P(1024) - 1|",
              "|removed N(1024)|", "|Jaccard(1024) - 1|"]]  # fmt: skip
    for env, s in summaries.items():
        error = s["curves"]["consistency_at_full"]
        table.append([
            env, f"{error['similarity']:.2e}", f"{error['retained_positive_share']:.2e}",
            f"{error['removed_negative_share']:.2e}", f"{error['jaccard']:.2e}",
        ])  # fmt: skip
    return table


def _coverage_table(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "queries", "multi-positive", "reference samples per query",
              "reference documents drawn", "groups short of documents",
              "queries using sampled pairs", "pairs per query at most",
              "corpus documents behind the scale"]]  # fmt: skip
    for env, s in summaries.items():
        c, settings = s["curves"]["coverage"], s["curves"]["settings"]
        table.append([
            env, str(c["queries"]), str(c["multi_positive_queries"]),
            str(settings["reference_samples"]), str(c["reference_documents_drawn"]),
            str(c["reference_groups_short_of_documents"]), str(c["queries_with_sampled_pairs"]),
            str(settings["max_query_pairs"]), str(settings["scale_documents"]),
        ])  # fmt: skip
    return table


def _cost_table(summaries: dict[str, Any]) -> list[list[str]]:
    table = [["environment", "curves and controls, seconds", "their summaries, seconds",
              "read", "transfer", "synchronized compute", "write", "peak device memory, MiB",
              "of which all-k and reference arrays, MiB"]]  # fmt: skip
    for env, s in summaries.items():
        cost = s["curves"]["cost"]
        sizes = s["cost"]["artifact_bytes"]
        arrays = sum(v for k, v in sizes.items() if k.startswith(("curve-", "reference-",
                                                                  "frequency-curves")))  # fmt: skip
        phase = s["cost"]["phase_seconds"]
        table.append([
            env, num(cost["pass_seconds"], 1), num(cost["summary_seconds"], 1),
            num(phase.get("read"), 1), num(phase.get("transfer"), 1),
            num(phase.get("compute"), 1), num(phase.get("write"), 1),
            num(s["cost"]["device_memory"].get("peak_allocated_mib"), 0),
            num(arrays / 2**20, 1),
        ])  # fmt: skip
    return table


def _wall_table(summaries: dict[str, Any]) -> list[list[str]]:
    """Where a run's wall time went; only the last two columns come from the kernel profile."""
    table = [["environment", "total, seconds", "load and verify inputs",
              "orderings and per-pair statistics", "corpus summaries", "control pass",
              "all-k summaries", "other (JSON, sizes)", "of those, profiled kernels",
              "unprofiled"]]  # fmt: skip
    for env, s in summaries.items():
        cost, curves = s["cost"], s["curves"]["cost"]
        total = cost["total_seconds"]
        stages = (cost["load_seconds"], cost["pair_seconds"], cost["corpus_seconds"],
                  curves["pass_seconds"], curves["summary_seconds"])  # fmt: skip
        profiled = sum(cost["phase_seconds"].values())
        table.append([
            env, num(total, 1), *[num(value, 1) for value in stages], num(total - sum(stages), 1),
            num(profiled, 1), num(total - profiled, 1),
        ])  # fmt: skip
    return table


def tables(
    inputs: dict[str, Any], summaries: dict[str, Any], differences: dict[str, Any] | None
) -> dict[str, list[list[str]]]:
    found = {
        "verification": _verification_table(inputs, summaries),
        "input-differences": _difference_table(differences),
        "agreement-ladder": _ladder_table(summaries),
        "agreement-scaled": _scaled_table(summaries),
        "agreement-strata": _strata_table(summaries),
        "pair-curves-checkpoints": _pair_table(summaries),
        "magnitude": _magnitude_table(summaries),
        "frequency-strata": _frequency_strata_table(summaries),
        "group-covariates": _scalar_table(summaries),
        "group-size": _size_table(summaries),
        "consistency": _consistency_table(summaries),
        "reference-coverage": _coverage_table(summaries),
        "control-cost": _cost_table(summaries),
        "wall-time": _wall_table(summaries),
    }
    for env, s in summaries.items():
        found[f"curves-{env}"] = _curve_table(s)
        found[f"frequency-curves-{env}"] = _frequency_table(s)
        found[f"strata-curves-{env}"] = _strata_curve_table(s)
    return found


def _fragment(values: list[float | None]) -> tuple[int, float] | None:
    """The ``k`` and value of the largest defined entry."""
    defined = [(v, at) for at, v in enumerate(values) if v is not None]
    if not defined:
        return None
    value, at = max(defined, key=lambda item: (item[0], -item[1]))
    return at + 1, value


def _meets(values: list[float | None], limit: float) -> str:
    """Where a curve reaches ``limit``, without implying that the ``k`` in between all do."""
    defined = [(at + 1, v) for at, v in enumerate(values) if v is not None]
    found = [k for k, v in defined if v >= limit]
    if not found:
        return f"at none of the {len(defined)} defined k"
    return f"at {len(found)} of the {len(defined)} defined k (first {found[0]}, last {found[-1]})"


def sentences(summaries: dict[str, Any]) -> dict[str, list[str]]:
    """What the curves show, by environment, with numbers read from them."""
    curves: list[str] = []
    shared: list[str] = []
    magnitude: list[str] = []
    counts: list[str] = []
    stable: list[str] = []
    for env, s in summaries.items():
        c = s["curves"]
        similarity = c["pair_curves"]["similarity"]["all"]["positive"]["mean"]
        reference = c["pair_curves"]["similarity"]["all"]["reference"]["mean"]
        peak = _fragment(similarity)
        ref_peak = _fragment(reference)
        if peak and ref_peak:
            curves.append(
                f"**{env}, raw similarity.** Relevant pairs peak at {num(peak[1])} with "
                f"k = {peak[0]} and end at {num(similarity[-1])}; matched reference pairs peak "
                f"at {num(ref_peak[1])} with k = {ref_peak[0]} and end at {num(reference[-1])}. "
                "Both are the largest inner product any mask of that size can give the pair."
            )
        multi = c["agreement"]["jaccard"]["multi"]
        sentence = _uncertainty_sentence(env, multi["difference"])
        if sentence:
            shared.append(sentence)
        largest = _fragment(multi["difference"]["mean"])
        across = c["across_query"].get("reference_raw")
        if largest and multi["positive"]["n"] and across:
            k = largest[0]
            at = k - 1
            repeats = c["reference_repeats"]["jaccard"]
            halves = c["delta_stability"]["jaccard"]
            shared.append(
                f"**{env}, sharing q.** Over {multi['positive']['n']} multi-positive queries, the "
                f"gap between relevant and reference masks is largest at k = {k}: relevant "
                f"{num(multi['positive']['mean'][at])}, reference with the same query "
                f"{num(multi['reference']['mean'][at])} (repeats {num(repeats['min'][at])} to "
                f"{num(repeats['max'][at])}), reference across queries {num(across['mean'][at])}, "
                f"chance {num(c['chance_jaccard'][at])}. Query halves put the gap between "
                f"{num(halves['min'][at])} and {num(halves['max'][at])}. Agreement among reference "
                "documents is what sharing q gives without any judged relationship."
            )
            cos = c["scalars"]["document_cosine"]["multi"]
            score = c["scalars"]["score"]["multi"]
            shared.append(
                f"**{env}, what else differs.** The two groups are not alike apart from their "
                f"judgment: full score {num(score['positive']['mean'])} against "
                f"{num(score['reference']['mean'])}, and cosine between a group's documents "
                f"{num(cos['positive']['mean'])} against {num(cos['reference']['mean'])}. The gap "
                "above may reflect either, not relevance itself."
            )
        raw, scaled = c["frequency"]["raw"], c["frequency"]["scaled"]
        marks = s["budgets"]
        rows = " ".join(
            f"k = {k}: {num(raw['positive_vs_magnitude'][k - 1])} / "
            f"{num(raw['reference_vs_magnitude'][k - 1])} / "
            f"{num(raw['positive_vs_reference'][k - 1])};"
            for k in (marks[0], marks[len(marks) // 2], marks[-1])
        )
        scaled_rows = " ".join(
            f"k = {k}: {num(scaled['positive_vs_magnitude'][k - 1])} / "
            f"{num(scaled['reference_vs_magnitude'][k - 1])} / "
            f"{num(scaled['split_half_excess']['mean'][k - 1])};"
            for k in (marks[0], marks[len(marks) // 2], marks[-1])
        )
        magnitude.append(
            f"**{env}, coordinate magnitude.** Selection frequency against `mean|q| x mean|d|` for "
            f"relevant / reference documents, and relevant against reference: {rows} After scaling "
            "each contribution by that size, relevant / reference against it, and split-half "
            f"Spearman of the excess over reference: {scaled_rows} The full curves are in "
            f"`tables/frequency-curves-{env}.csv`."
        )
        for row in c["group_size"]:
            conflict, cancelled = row["sign_conflict_share"], row["cancelled_share"]
            if conflict["difference"]["n"]:
                counts.append(
                    f"**{env}, groups of {row['size']}** ({conflict['difference']['n']} queries). "
                    f"Sign conflict {num(conflict['positive']['mean'])} relevant, "
                    f"{num(conflict['reference']['mean'])} reference. Cancelled mass "
                    f"{num(cancelled['positive']['mean'])} relevant, "
                    f"{num(cancelled['reference']['mean'])} reference."
                )
        excess = raw["split_half_excess"]
        positive = raw["split_half_positive"]
        stable.append(
            f"**{env}, every k.** The lowest split-half Spearman reaches {STABLE_SPEARMAN} for the "
            f"relevant-document frequency profile {_meets(positive['min'], STABLE_SPEARMAN)}, and "
            f"for its excess over reference documents {_meets(excess['min'], STABLE_SPEARMAN)}. A "
            "frequency profile can repeat because coordinates repeat; the excess is what is left "
            "after the profile of reference documents is subtracted."
        )
    return {"curves": curves, "shared": shared, "magnitude": magnitude, "counts": counts,
            "stable": stable}  # fmt: skip


def provenance_lines(tables_: dict[str, list[list[str]]], differences: Any) -> list[str]:
    return [
        "### Input verification",
        "",
        "Checked before analysis, and a failure stops the run: payload hashes, D = 1024,",
        "embeddings made from this dataset, split counts against the dataset manifest, disjoint",
        "splits, and unit norms within 0.01 on every training query and a seeded document sample.",
        "",
        *_markdown(tables_["verification"]),
        "",
        "### Differences from the earlier spike-derived inputs",
        "",
        *(
            _markdown(tables_["input-differences"])
            if differences
            else [
                "Not recorded for this run. On the published artifacts, run",
                "`python scripts/checks/p3_input_differences.py artifacts artifacts/p3-spike`.",
            ]
        ),
    ]


def observed_lines(
    figures_: dict[str, str], tables_: dict[str, list[list[str]]], words: dict[str, list[str]]
) -> list[str]:
    return [
        "### Every cardinality",
        "",
        "Each pair has one descending contribution ordering, computed once and reused for every k",
        "from 1 to 1024. Three different quantities come from it: **raw similarity** (the",
        "unnormalized score of the top k), **positive mass retained**, and **negative mass",
        "removed**. `k = 0`, the empty mask, scores 0 by definition and appears only as a labeled",
        "reference row. Every value, with its standard error and eligible query count, is in",
        "`tables/curves-<environment>.csv` (positive-count strata in",
        "`tables/strata-curves-<environment>.csv`); checkpoint rows here are samples of those.",
        "",
        *[f"- {line}" for line in words["curves"]],
        "",
        *_figures_for(figures_, "similarity"),
        "",
        *_figures_for(figures_, "retained"),
        "",
        *_figures_for(figures_, "removed"),
        "",
        *_markdown(tables_["pair-curves-checkpoints"]),
        "",
        "At `k = 1024` the curves must return to the unmasked pair, which checks the arithmetic:",
        "",
        *_markdown(tables_["consistency"]),
        "",
        "### What cardinality forces",
        "",
        "At `k = 1024` every mask is the same set, so Jaccard 1 and a zero gap there are not",
        "evidence of shared relevance structure. The same effect acts earlier: above `k = 512` two",
        "masks must share at least `2k - 1024` dimensions (the floor), chance rises with k, and",
        "any gap is squeezed toward zero by construction. Selection frequency saturates the same",
        "way, and at `k = 1024` its rank correlations are undefined because every dimension is",
        "always selected, so a fall in correlation near the top of the range is not by itself a",
        "loss of structure. There, what a mask leaves out is the most negative contributions,",
        "which can be large in magnitude, so correlation with magnitude can change sign.",
        "Chance-corrected curves are in the run's `curves.json`.",
        "",
        "### Control 1: sharing the same query vector",
        "",
        "For each query, the masks of its relevant documents are compared with masks of",
        "corpus-reference documents paired with that same query: the same number of documents,",
        "drawn uniformly without looking at content, excluding the query's judged documents, in",
        "repeated seeded samples. Reference documents are unjudged, not negatives. Reading up the",
        "ladder: chance has no structure; different queries share only coordinates; the same query",
        "adds `q`; relevant documents add whatever else relevant documents have in common.",
        "",
        *[f"- {line}" for line in words["shared"]],
        "",
        *_figures_for(figures_, "agreement"),
        "",
        *_markdown(tables_["agreement-ladder"]),
        "",
        *_markdown(tables_["reference-coverage"]),
        "",
        "### Control 2: coordinate magnitude",
        "",
        "Two views. First, whether relevant and reference documents select the same dimensions at",
        "all, with `selection placed differently` the share of selection the two profiles place",
        "differently. Second, the same analysis after dividing each contribution by",
        "`mean|q_j| * mean|d_j|`, estimated from unlabeled vectors. Scaling removes that one",
        "per-dimension size, not sign structure or within-query structure.",
        "",
        *[f"- {line}" for line in words["magnitude"]],
        "",
        *_figures_for(figures_, "frequency-control"),
        "",
        *_markdown(tables_["magnitude"]),
        "",
        *_figures_for(figures_, "agreement-scaled"),
        "",
        *_markdown(tables_["agreement-scaled"]),
        "",
        "### Control 3: the number of relevant documents",
        "",
        "Mean pairwise Jaccard has no built-in dependence on group size, so it is compared within",
        "strata. Sign-conflict coverage can only grow as documents join a group; the cancelled",
        "share need not, and can fall. Both depend on group size, so they are compared at a fixed",
        "group size, on relevant and matched reference documents. Only",
        "queries with at least that many positives enter a row, and that selection remains.",
        "",
        *_figures_for(figures_, "agreement-strata"),
        "",
        *_markdown(tables_["agreement-strata"]),
        "",
        *_markdown(tables_["frequency-strata"]),
        "",
    ]


def conceal_lines(
    figures_: dict[str, str], tables_: dict[str, list[list[str]]], words: dict[str, list[str]]
) -> list[str]:
    return [
        "### Cancellation and conflict with the document count held fixed",
        "",
        "A sign conflict needs `d_j` to change sign wherever `q_j` is nonzero, so it comes from",
        "the documents. Cancellation is then weighted by `|q_j|`. Means here are unweighted,",
        "because a reference document has no grade.",
        "",
        *[f"- {line}" for line in words["counts"]],
        "",
        *_figures_for(figures_, "group-size"),
        "",
        *_markdown(tables_["group-size"]),
        "",
        *_markdown(tables_["group-covariates"]),
    ]


def _figures_for(figures_: dict[str, str], stem: str) -> list[str]:
    """Figures named ``stem`` or ``stem-<environment>``, excluding longer stems of the same root."""
    names = sorted(figures_)
    found = []
    for name in names:
        if name == stem or (
            name.startswith(f"{stem}-") and name.removeprefix(f"{stem}-") in _environments(names)
        ):
            found.append(f"![{name}](figures/{name}.svg)")
    return found


def _environments(names: list[str]) -> set[str]:
    return {n.removeprefix("similarity-") for n in names if n.startswith("similarity-")}


def reproduce_lines(foreign: bool, inputs: dict[str, Any], settings: dict[str, Any]) -> list[str]:
    envs = " ".join(inputs["environments"])
    bound = f" --max-queries {settings['max_queries']}" if settings["max_queries"] else ""
    if foreign:
        return [
            "python scripts/checks/p3_spike_inputs.py artifacts/spike artifacts/p3-spike",
            f"hyperdime analyze-representations --output artifacts/p3-spike --envs {envs}{bound} "
            "--allow-spike-inputs",
            "hyperdime report P3 --artifacts artifacts/p3-spike  # writes reports/P3-spike",
        ]
    return [
        "python scripts/checks/p3_input_differences.py artifacts artifacts/p3-spike",
        f"hyperdime analyze-representations --envs {envs}{bound}",
        "hyperdime report P3",
    ]


def figures(summaries: dict[str, Any]) -> dict[str, str]:
    found: dict[str, str] = {}
    envs = list(summaries)
    for env, s in summaries.items():
        c = s["curves"]
        k = c["k"]
        for stem, name, title, label, subtitle in (
            ("similarity", "similarity", f"Raw similarity at every k ({env})", "Score",
             "Mean largest inner product of a mask of size k; ends at the full score at k = 1024."),
            ("retained", "retained_positive_share", f"Positive mass retained ({env})",
             "Share of positive mass", "Mean share of a pair's positive mass inside its top k."),
            ("removed", "removed_negative_share", f"Negative mass removed ({env})",
             "Share of negative mass",
             "Mean share of a pair's negative mass outside its top k; reaches 0 at k = 1024."),
        ):  # fmt: skip
            row = c["pair_curves"][name]["all"]
            found[f"{stem}-{env}"] = lines(
                title, subtitle, "Retained dimensions k", label,
                [Series("Relevant documents", k, row["positive"]["mean"]),
                 Series("Reference documents, same query", k, row["reference"]["mean"])],
            )  # fmt: skip
        multi = c["agreement"]["jaccard"]["multi"]
        ladder = [
            Series("Same query, relevant", k, multi["positive"]["mean"]),
            Series("Same query, reference", k, multi["reference"]["mean"]),
        ]
        across = c["across_query"].get("reference_raw")
        if across:
            ladder.append(Series("Different queries, reference", k, across["mean"]))
        ladder.append(Series("Chance", k, c["chance_jaccard"], neutral=True))
        found[f"agreement-{env}"] = lines(
            f"Do documents of a query pick the same dimensions? ({env})",
            "Mean Jaccard of pairwise top-k masks, multi-positive queries, query-balanced. All "
            "curves reach 1 at k = 1024 by construction.",
            "Retained dimensions k",
            "Mean Jaccard",
            ladder,
        )
        halves = c["delta_stability"]["jaccard"]
        found[f"agreement-difference-{env}"] = lines(
            f"Relevant minus reference agreement ({env})",
            "Mean within-query difference in Jaccard, with its range over disjoint query halves. "
            "Forced toward 0 as k approaches 1024.",
            "Retained dimensions k",
            "Jaccard difference",
            [Series("Relevant minus reference", k, multi["difference"]["mean"], halves["min"],
                    halves["max"])],
        )  # fmt: skip
        scaled = c["agreement"]["scaled_jaccard"]["multi"]
        found[f"agreement-scaled-{env}"] = lines(
            f"Agreement after scaling out coordinate size ({env})",
            "Mean Jaccard of top-k masks of c_j / (mean|q_j| * mean|d_j|). All curves reach 1 at "
            "k = 1024 by construction.",
            "Retained dimensions k",
            "Mean Jaccard",
            [Series("Same query, relevant", k, scaled["positive"]["mean"]),
             Series("Same query, reference", k, scaled["reference"]["mean"]),
             Series("Chance", k, c["chance_jaccard"], neutral=True)],
        )  # fmt: skip
        by_stratum = [
            Series(subset.removeprefix("positives ") + " positives", k, row["difference"]["mean"])
            for subset, row in _strata(c["agreement"]["jaccard"])
            if row["difference"]["n"]
        ]
        if by_stratum:
            found[f"agreement-strata-{env}"] = lines(
                f"Relevant minus reference agreement by positives per query ({env})",
                "Each stratum against reference groups of the same size as its own positives.",
                "Retained dimensions k",
                "Jaccard difference",
                by_stratum,
            )
        raw = c["frequency"]["raw"]
        found[f"frequency-control-{env}"] = lines(
            f"Is recurrence about relevant documents or coordinates? ({env})",
            "Spearman over 1024 dimensions at each k; undefined at k = 1024, where every dimension "
            "is always selected.",
            "Retained dimensions k",
            "Spearman",
            [Series("Relevant vs magnitude", k, raw["positive_vs_magnitude"]),
             Series("Relevant vs reference", k, raw["positive_vs_reference"]),
             Series("Excess over reference, split-half", k, raw["split_half_excess"]["mean"],
                    raw["split_half_excess"]["min"], None)],
        )  # fmt: skip
    found["stability"] = lines(
        "Does the relevant-document frequency profile repeat across query halves?",
        "Split-half Spearman of query-balanced frequency at every k, mean. Stability within these "
        "training data only.",
        "Retained dimensions k",
        "Spearman",
        [Series(env, s["curves"]["k"],
                s["curves"]["frequency"]["raw"]["split_half_positive"]["mean"])
         for env, s in summaries.items()],
    )  # fmt: skip
    sizes = [
        (env, row)
        for env in envs
        for row in summaries[env]["curves"]["group_size"]
        if row["sign_conflict_share"]["difference"]["n"]
    ]
    categories = [
        f"{env}, {row['size']} (n={row['sign_conflict_share']['difference']['n']})"
        for env, row in sizes
    ]
    for stem, name, title, label in (
        ("group-size-conflict", "sign_conflict_share",
         "Sign conflict at a fixed number of documents", "Share of dimensions"),
        ("group-size-cancelled", "cancelled_share",
         "Mass cancelled at a fixed number of documents", "Share of mass"),
    ):  # fmt: skip
        found[stem] = bars(
            title,
            "Same number of documents in each bar, per environment; relevant and matched reference "
            "documents of the same queries.",
            "Environment, documents in the group (n = queries)",
            label,
            categories,
            {
                "Relevant": [row[name]["positive"]["mean"] for _, row in sizes],
                "Reference": [row[name]["reference"]["mean"] for _, row in sizes],
            },
        )
    return found
