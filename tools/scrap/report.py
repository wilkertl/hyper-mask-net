"""Report documents (JSON), the text report, baselines, and baseline comparison."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from scrap import policy
from scrap.analyze import Example, FileAnalysis
from scrap.summary import guidance, pressure_score

EXAMPLE_FIELDS = (
    "name",
    "line",
    "end_line",
    "line_count",
    "raw_line_count",
    "assertions",
    "branches",
    "setup_depth",
    "with_redefs",
    "helper_calls",
    "helper_hidden_lines",
    "temp_resources",
    "table_driven",
    "api_contract",
    "complexity",
    "complexity_score",
    "scrap",
)


def file_report(analysis: FileAnalysis, source: bytes) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": analysis.path,
        "content_hash": hashlib.sha256(source).hexdigest(),
        "errors": list(analysis.errors),
        "examples": [example_record(e) for e in analysis.examples],
    }
    if analysis.examples:
        report.update(guidance(analysis.examples))
    return report


def example_record(example: Example) -> dict[str, Any]:
    record = {name: getattr(example, name) for name in EXAMPLE_FIELDS}
    record["describe_path"] = list(example.describe_path)
    record["smells"] = list(example.smells)
    return record


def document(reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {"tool": "scrap-py", "version": policy.BASELINE_VERSION, "reports": list(reports)}


def compare(baseline: dict[str, Any], reports: Sequence[dict[str, Any]]) -> None:
    """Attach a ``comparison`` with deltas and a verdict to each report with a baseline."""
    previous = {r["path"]: r for r in baseline.get("reports", [])}
    for report in reports:
        before = previous.get(report["path"], {}).get("summary")
        after = report.get("summary")
        if before is None or after is None:
            continue
        comparison = {
            "baseline_hash": previous[report["path"]]["content_hash"],
            "current_hash": report["content_hash"],
            "file_score_delta": pressure_score(after) - pressure_score(before),
            "avg_scrap_delta": after["avg_scrap"] - before["avg_scrap"],
            "max_scrap_delta": after["max_scrap"] - before["max_scrap"],
            "harmful_duplication_delta": after["harmful_duplication_score"]
            - before["harmful_duplication_score"],
            "extraction_pressure_delta": after["effective_duplication_score"]
            - before["effective_duplication_score"],
            "case_matrix_delta": after["case_matrix_repetition"] - before["case_matrix_repetition"],
            "helper_hidden_delta": after["helper_hidden_example_count"]
            - before["helper_hidden_example_count"],
        }
        comparison["verdict"] = verdict(comparison)
        report["comparison"] = comparison


def verdict(c: dict[str, Any]) -> str:
    if (
        c["file_score_delta"] <= -5
        and c["extraction_pressure_delta"] <= 0
        and c["max_scrap_delta"] <= 0
    ):
        return "improved"
    if (
        c["helper_hidden_delta"] > 0
        and c["extraction_pressure_delta"] >= 0
        and c["case_matrix_delta"] <= 0
    ):
        return "worse"
    if c["extraction_pressure_delta"] > 0 or c["max_scrap_delta"] > 0 or c["file_score_delta"] >= 5:
        return "worse"
    return "unchanged" if c["file_score_delta"] == 0 else "mixed"


def render_text(reports: Sequence[dict[str, Any]], verbose: bool = False) -> str:
    lines: list[str] = []
    for report in reports:
        lines += _render_file(report, verbose)
    files = len(reports)
    unstable = [r for r in reports if r.get("remediation_mode") not in (None, "STABLE")]
    errors = sum(len(r["errors"]) for r in reports)
    lines.append(
        f"SCRAP: {files} test files, {len(unstable)} not stable, {errors} structural errors."
    )
    return "\n".join(lines) + "\n"


def _render_file(report: dict[str, Any], verbose: bool) -> list[str]:
    lines = [f"== {report['path']}"]
    lines += [f"  ERROR {error}" for error in report["errors"]]
    if "summary" not in report:
        return [*lines, "  no tests", ""]
    summary = report["summary"]
    lines.append(
        f"  {report['remediation_mode']} / {report['ai_actionability']}: pressure "
        f"{report['file_score']:.1f} ({report['file_level']}); {summary['example_count']} tests, "
        f"avg SCRAP {summary['avg_scrap']:.1f}, max {summary['max_scrap']:.1f}"
    )
    if "comparison" in report:
        c = report["comparison"]
        lines.append(f"  vs baseline: {c['verdict']} (pressure {c['file_score_delta']:+.1f})")
        if c["verdict"] == "worse":
            lines.append("  The change made the file structurally worse; simplify or revert it.")
    if report["remediation_mode"] == "STABLE" and not verbose:
        return [*lines, ""]
    lines.append(f"  {report['ai_actionability_message']}")
    lines += [f"  [{a['label']}] {a['text']}" for a in report["actions"]]
    examples = {e["name"]: e for e in report["examples"]}
    for name in report["top_examples"]:
        e = examples[name]
        smells = ", ".join(e["smells"]) or "no smells"
        lines.append(
            f"  worst: {name} (lines {e['line']}-{e['end_line']}) SCRAP {e['scrap']:.1f}: {smells}"
        )
    for extraction in summary["recommended_extractions"]:
        lines.append(
            f"  extract: {', '.join(extraction['test_names'])} (lines {extraction['line_start']}-"
            f"{extraction['line_end']}), net benefit {extraction['net_benefit']:.1f}"
        )
    if verbose:
        lines += _verbose(report)
    return [*lines, ""]


def _verbose(report: dict[str, Any]) -> list[str]:
    lines = ["  summary:"]
    lines += [
        f"    {k}: {v}"
        for k, v in sorted(report["summary"].items())
        if k != "recommended_extractions"
    ]
    for block in report["blocks"]:
        lines.append(
            f"  class {' / '.join(block['path'])}: {block['level']} ({block['score']:.1f}), "
            f"worst {block['worst_example']}"
        )
    for e in report["examples"]:
        metrics = ", ".join(f"{k}={e[k]}" for k in EXAMPLE_FIELDS[1:])
        lines.append(f"  test {e['name']}: {metrics}")
    return lines


def baseline_path(root: Path) -> Path:
    return root / "artifacts" / "scrap" / "baseline.json"
