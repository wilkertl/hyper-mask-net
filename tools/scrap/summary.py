"""File and block summaries, extraction pressure, and refactoring guidance (SCRAP semantics)."""

from __future__ import annotations

import itertools
import math
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

from scrap import policy
from scrap.analyze import Example

Features = Callable[[Example], frozenset[str]]

CHANNELS: dict[str, Features] = {
    "setup": lambda e: e.setup_features,
    "assertion": lambda e: e.assert_features,
    "fixture": lambda e: e.fixture_features,
    "literal": lambda e: e.literal_features,
    "arrange": lambda e: e.arrange_features,
}


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def harmful_features(example: Example) -> frozenset[str]:
    return (
        example.setup_features
        | example.assert_features
        | example.fixture_features
        | example.arrange_features
    )


def similar_to_any(example: Example, examples: Sequence[Example], key: Features) -> bool:
    features = key(example)
    return bool(features) and any(
        jaccard(features, key(other)) >= policy.DUPLICATION_THRESHOLD
        for other in examples
        if other is not example
    )


def similar_example_count(examples: Sequence[Example], key: Features) -> int:
    return sum(1 for e in examples if similar_to_any(e, examples, key))


def coverage_matrix_candidate(example: Example, features: frozenset[str]) -> bool:
    """Small, flat, single-assertion examples: likely a coverage table, not harmful repetition."""
    return (
        example.scrap <= policy.MATRIX_MAX_SCRAP
        and example.line_count <= policy.MATRIX_MAX_LINES
        and example.assertions <= policy.MATRIX_MAX_ASSERTIONS
        and example.branches <= policy.MATRIX_MAX_BRANCHES
        and example.setup_depth <= policy.MATRIX_MAX_SETUP_DEPTH
        and example.with_redefs <= policy.MATRIX_MAX_WITH_REDEFS
        and example.temp_resources <= policy.MATRIX_MAX_TEMP_RESOURCES
        and example.helper_hidden_lines <= policy.MATRIX_MAX_HELPER_HIDDEN_LINES
        and (
            example.table_driven
            or (
                len(example.subject_symbols) <= policy.MATRIX_MAX_SUBJECT_SYMBOLS and bool(features)
            )
        )
    )


def coverage_matrix_count(examples: Sequence[Example]) -> int:
    def candidate(e: Example) -> bool:
        return coverage_matrix_candidate(e, e.assert_features | e.arrange_features)

    def subjects(e: Example) -> frozenset[str]:
        return e.subject_symbols

    return sum(
        1
        for e in examples
        if candidate(e)
        and (
            similar_to_any(e, examples, CHANNELS["setup"])
            or similar_to_any(e, examples, CHANNELS["arrange"])
            or (
                similar_to_any(e, examples, CHANNELS["assertion"])
                and similar_to_any(e, examples, subjects)
            )
        )
    )


def duplication_cost(shared_forms: int, instances: int, variable_points: int) -> float:
    """``D_before``: zero for small or highly variable clusters, else grows with repetition."""
    if shared_forms <= policy.MIN_SHARED_FORMS or variable_points > policy.MAX_VARIABLE_POINTS:
        return 0.0
    repeated: float = max(0, instances - 1) ** 1.5
    return (shared_forms - policy.MIN_SHARED_FORMS) * repeated / (variable_points + 1)


def duplicate_clusters(examples: Sequence[Example]) -> list[list[Example]]:
    """Connected components of examples whose harmful features are at least 50% similar."""
    features = [harmful_features(e) for e in examples]
    adjacency: dict[int, set[int]] = {i: set() for i in range(len(examples))}
    for i, j in itertools.combinations(range(len(examples)), 2):
        if jaccard(features[i], features[j]) >= policy.DUPLICATION_THRESHOLD:
            adjacency[i].add(j)
            adjacency[j].add(i)
    seen: set[int] = set()
    clusters: list[list[Example]] = []
    for start in range(len(examples)):
        if start in seen:
            continue
        component, stack = set(), [start]
        while stack:
            index = stack.pop()
            if index not in component:
                component.add(index)
                stack.extend(adjacency[index] - component)
        seen |= component
        if len(component) > 1:
            clusters.append([examples[i] for i in sorted(component)])
    return clusters


def extraction(cluster: Sequence[Example]) -> dict[str, Any]:
    sets = [harmful_features(e) for e in cluster]
    shared = frozenset.intersection(*sets)
    every = frozenset.union(*sets)
    shared_forms, variable_points = len(shared), len(every) - len(shared)
    before = duplication_cost(shared_forms, len(cluster), variable_points)
    helper_cost = shared_forms + variable_points
    paths = Counter(" / ".join(e.describe_path) for e in cluster)
    return {
        "test_names": [e.name for e in cluster],
        "describe_path": paths.most_common(1)[0][0],
        "line_start": min(e.line for e in cluster),
        "line_end": max(e.end_line for e in cluster),
        "instances": len(cluster),
        "shared_forms": shared_forms,
        "variable_points": variable_points,
        "duplication_before": before,
        "duplication_after": 0.0,
        "helper_cost": helper_cost,
        "net_benefit": max(0.0, before - helper_cost),
    }


def summarize_extractions(examples: Sequence[Example]) -> dict[str, Any]:
    recommendations = [
        extraction(cluster)
        for cluster in duplicate_clusters(examples)
        if not all(coverage_matrix_candidate(e, harmful_features(e)) for e in cluster)
    ]
    recommendations = sorted(
        (r for r in recommendations if r["net_benefit"] > 0),
        key=lambda r: (r["line_start"], -r["net_benefit"]),
    )
    return {
        "recommended_extraction_count": len(recommendations),
        "extraction_pressure_score": sum(r["net_benefit"] for r in recommendations),
        "recommended_extractions": recommendations,
    }


def summarize_examples(examples: Sequence[Example]) -> dict[str, Any]:
    total = len(examples)
    counts = {name: similar_example_count(examples, key) for name, key in CHANNELS.items()}
    subject_repetition = similar_example_count(examples, lambda e: e.subject_symbols)
    matrix = coverage_matrix_count(examples)
    extractions = summarize_extractions(examples)
    return {
        "example_count": total,
        "avg_scrap": sum(e.scrap for e in examples) / total if total else 0.0,
        "max_scrap": max((e.scrap for e in examples), default=0.0),
        "branching_examples": sum(1 for e in examples if e.branches > 0 and not e.table_driven),
        "low_assertion_examples": sum(1 for e in examples if e.assertions <= 1),
        "zero_assertion_examples": sum(1 for e in examples if e.assertions == 0),
        "with_redefs_examples": sum(1 for e in examples if e.with_redefs > 0),
        "helper_hidden_example_count": sum(1 for e in examples if e.helper_hidden_lines > 0),
        "table_driven_examples": sum(1 for e in examples if e.table_driven),
        **{f"{name}_duplication_score": count for name, count in counts.items()},
        "subject_repetition_score": subject_repetition,
        "duplication_score": sum(counts.values()),
        "harmful_duplication_score": sum(c for n, c in counts.items() if n != "literal"),
        "coverage_matrix_candidates": matrix,
        "case_matrix_repetition": matrix,
        **extractions,
        "effective_duplication_score": extractions["extraction_pressure_score"],
    }


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def stable(summary: dict[str, Any]) -> bool:
    count = summary["example_count"]
    zero = ratio(summary["zero_assertion_examples"], count)
    low = ratio(summary["low_assertion_examples"], count)
    small = (
        count <= 2
        and summary["max_scrap"] <= 10
        and summary["effective_duplication_score"] <= 1
        and summary["helper_hidden_example_count"] == 0
        and zero <= 0.0
    )
    general = (
        count > 0
        and summary["max_scrap"] <= 12
        and summary["effective_duplication_score"] <= 3
        and zero <= 0.0
        and low <= 0.35
    )
    return small or general


def pressure_score(summary: dict[str, Any]) -> float:
    count = summary["example_count"]
    factor = next(f for limit, f in policy.SIZE_FACTORS if limit is None or count <= limit)
    base = (
        policy.WEIGHT_AVG_SCRAP * summary["avg_scrap"]
        + policy.WEIGHT_MAX_SCRAP * summary["max_scrap"]
        + policy.WEIGHT_EFFECTIVE_DUPLICATION * summary["effective_duplication_score"]
        + policy.WEIGHT_LOW_ASSERTION_RATIO * ratio(summary["low_assertion_examples"], count)
        + policy.WEIGHT_BRANCHING_RATIO * ratio(summary["branching_examples"], count)
        + policy.WEIGHT_WITH_REDEFS_RATIO * ratio(summary["with_redefs_examples"], count)
        + policy.WEIGHT_HELPER_HIDDEN_RATIO * ratio(summary["helper_hidden_example_count"], count)
    )
    credit = policy.MATRIX_CREDIT * summary["case_matrix_repetition"]
    return max(0.0, float(factor * base - credit))


def pressure_level(summary: dict[str, Any]) -> str:
    if stable(summary):
        return "STABLE"
    score = pressure_score(summary)
    if score >= policy.LEVEL_CRITICAL:
        return "CRITICAL"
    if score >= policy.LEVEL_HIGH:
        return "HIGH"
    return "MEDIUM" if score >= policy.LEVEL_MEDIUM else "LOW"


def remediation_mode(summary: dict[str, Any], blocks: Sequence[dict[str, Any]]) -> str:
    if stable(summary):
        return "STABLE"
    high_blocks = sum(1 for b in blocks if b["level"] in ("HIGH", "CRITICAL"))
    split_pressure = (
        summary["avg_scrap"] >= policy.SPLIT_AVG_SCRAP
        or summary["effective_duplication_score"] >= policy.SPLIT_EFFECTIVE_DUPLICATION
        or summary["subject_repetition_score"] >= policy.SPLIT_SUBJECT_REPETITION
        or summary["helper_hidden_example_count"] > 0
    )
    if (
        summary["example_count"] >= policy.SPLIT_EXAMPLE_COUNT
        and (
            high_blocks >= policy.SPLIT_HIGH_PRESSURE_BLOCKS
            or summary["max_scrap"] >= policy.SPLIT_MAX_SCRAP
        )
        and split_pressure
    ):
        return "SPLIT"
    return "LOCAL"


def ratios(summary: dict[str, Any]) -> dict[str, float]:
    count = summary["example_count"]
    return {
        "low_assertion": ratio(summary["low_assertion_examples"], count),
        "zero_assertion": ratio(summary["zero_assertion_examples"], count),
        "branching": ratio(summary["branching_examples"], count),
        "mocking": ratio(summary["with_redefs_examples"], count),
    }


def actionability(summary: dict[str, Any], mode: str) -> tuple[str, str]:
    r = ratios(summary)
    duplication = summary["effective_duplication_score"]
    if mode == "STABLE":
        return (
            "LEAVE_ALONE",
            "Leave this file alone unless explicitly requested; the structure is stable enough.",
        )
    matrix = summary["case_matrix_repetition"]
    if (
        summary["coverage_matrix_candidates"] > 0
        and matrix
        >= max(
            policy.MATRIX_MIN_CASE_REPETITION,
            math.floor(duplication / policy.MATRIX_EFFECTIVE_DUPLICATION_DIVISOR),
        )
        and summary["max_scrap"] <= policy.MATRIX_ACTION_MAX_SCRAP
        and r["branching"] <= policy.MATRIX_MAX_BRANCHING_RATIO
        and r["mocking"] < policy.MATRIX_MAX_MOCKING_RATIO
    ):
        return (
            "AUTO_TABLE_DRIVE",
            "Safe to parametrize automatically; the repetition looks like a coverage matrix.",
        )
    if (
        mode == "LOCAL"
        and (
            duplication > 0
            or r["zero_assertion"] > 0
            or r["low_assertion"] > policy.LOCAL_LOW_ASSERTION_RATIO
            or summary["max_scrap"] > policy.LOCAL_MAX_SCRAP
        )
        and r["branching"] <= policy.LOCAL_MAX_BRANCHING_RATIO
        and r["mocking"] < policy.LOCAL_MAX_MOCKING_RATIO
    ):
        return (
            "AUTO_REFACTOR",
            "Safe to refactor locally: strengthen assertions, shrink oversized tests, and extract "
            "duplicated scaffolding.",
        )
    if mode == "SPLIT":
        return (
            "MANUAL_SPLIT",
            "Do not refactor locally first; split the file by responsibility before cleanup.",
        )
    return (
        "REVIEW_FIRST",
        "Do not refactor immediately; inspect the file before acting on the recommendations.",
    )


def actions(summary: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    """Ranked recommendations (HIGH, MEDIUM, LOW), at most four."""
    if mode == "STABLE":
        return [_action(3, "No refactor recommended; the file is stable enough to leave alone.")]
    r = ratios(summary)
    rules = [
        (
            mode == "SPLIT",
            3,
            "Split this test file by responsibility before local cleanup; the pressure is spread "
            "across several hotspots.",
        ),
        (
            summary["coverage_matrix_candidates"] > 0,
            3,
            "Convert repeated low-complexity tests into one @pytest.mark.parametrize test; this is "
            "coverage-matrix repetition, not harmful duplication.",
        ),
        (
            r["zero_assertion"] > 0 or r["low_assertion"] > policy.LOCAL_LOW_ASSERTION_RATIO,
            3,
            "Strengthen assertions in weak tests before structural cleanup.",
        ),
        (
            mode == "LOCAL" and summary["max_scrap"] > policy.LOCAL_MAX_SCRAP,
            3,
            "Split oversized tests into narrower tests.",
        ),
        (
            summary["effective_duplication_score"] > 0,
            2,
            "Extract shared setup or repeated assertion scaffolding only where harmful duplication "
            "dominates (see the extraction candidates).",
        ),
        (
            r["mocking"] > policy.LOCAL_MAX_MOCKING_RATIO,
            2,
            "Reduce mocking and move coverage toward higher-level behavior.",
        ),
        (
            r["branching"] > policy.LOCAL_MAX_BRANCHING_RATIO,
            2,
            "Remove logic from tests, or keep variation in parametrize tables rather than control "
            "flow.",
        ),
        (
            summary["helper_hidden_example_count"] > 0,
            1,
            "Be skeptical of helpers that only hide setup; their lines still count as complexity.",
        ),
        (
            mode == "LOCAL" and summary["avg_scrap"] > policy.LOCAL_AVG_SCRAP,
            1,
            "Consider splitting this file or class by responsibility.",
        ),
    ]
    found = [_action(confidence, text) for enabled, confidence, text in rules if enabled]
    return sorted(found, key=lambda a: (-a["confidence"], a["text"]))[: policy.MAX_ACTIONS]


def blocks(examples: Sequence[Example]) -> list[dict[str, Any]]:
    """One summary per test class (SCRAP's describe blocks); module-level tests have none."""
    grouped: dict[tuple[str, ...], list[Example]] = {}
    for example in examples:
        if example.describe_path:
            grouped.setdefault(example.describe_path, []).append(example)
    result = []
    for path, members in sorted(grouped.items()):
        summary = summarize_examples(members)
        worst = max(members, key=lambda e: e.scrap)
        result.append(
            {
                "path": list(path),
                "summary": summary,
                "score": pressure_score(summary),
                "level": pressure_level(summary),
                "worst_example": worst.name,
            }
        )
    return result


def guidance(examples: Sequence[Example]) -> dict[str, Any]:
    summary = summarize_examples(examples)
    file_blocks = blocks(examples)
    mode = remediation_mode(summary, file_blocks)
    action_mode, action_message = actionability(summary, mode)
    top_blocks = sorted(file_blocks, key=lambda b: -b["score"])[: policy.TOP_BLOCK_COUNT]
    top_examples = sorted(examples, key=lambda e: -e.scrap)[: policy.TOP_EXAMPLE_COUNT]
    return {
        "summary": summary,
        "blocks": file_blocks,
        "file_score": pressure_score(summary),
        "file_level": pressure_level(summary),
        "remediation_mode": mode,
        "ai_actionability": action_mode,
        "ai_actionability_message": action_message,
        "actions": actions(summary, mode),
        "top_blocks": [b["path"] for b in top_blocks],
        "top_examples": [e.name for e in top_examples],
    }


def _action(confidence: int, text: str) -> dict[str, Any]:
    return {
        "confidence": confidence,
        "label": {3: "HIGH", 2: "MEDIUM", 1: "LOW"}[confidence],
        "text": text,
    }
