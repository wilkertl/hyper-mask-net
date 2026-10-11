"""Command line: ``python -m scrap [paths] [--json] [--verbose] [--write-baseline] [--compare F]``.

Exit status is 1 when a file has structural errors (a parse error, a test shadowed by a later
test of the same name, or a test nested inside another test) and 0 otherwise: the scores and
recommendations are decision support and never fail the run.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from scrap.analyze import analyze_file, discover
from scrap.report import baseline_path, compare, document, file_report, render_text


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scrap", description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=[Path("tests")])
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument("--verbose", action="store_true", help="every metric, every file")
    parser.add_argument(
        "--write-baseline",
        nargs="?",
        type=Path,
        const=baseline_path(Path.cwd()),
        help="write the JSON report as a baseline (default: artifacts/scrap/baseline.json)",
    )
    parser.add_argument("--compare", type=Path, help="compare with a baseline written earlier")
    args = parser.parse_args(argv)

    files = discover(args.paths)
    reports = []
    for path in files:
        display = _display(path)
        reports.append(file_report(analyze_file(path, display), path.read_bytes()))
    if args.compare is not None:
        compare(json.loads(args.compare.read_text(encoding="utf-8")), reports)
    doc = document(reports)
    if args.write_baseline is not None:
        args.write_baseline.parent.mkdir(parents=True, exist_ok=True)
        args.write_baseline.write_text(_dumps(doc), encoding="utf-8")
    sys.stdout.write(_dumps(doc) if args.json else render_text(reports, args.verbose))
    return 1 if any(r["errors"] for r in reports) else 0


def _display(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _dumps(doc: object) -> str:
    return json.dumps(doc, indent=2, sort_keys=True) + "\n"
