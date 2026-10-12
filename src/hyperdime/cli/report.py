"""``hyperdime report E0``: generate the E0 report and the G1 criteria from result files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from hyperdime.reports import e0

COMMAND = "report"


def register(commands: Any) -> None:
    parser = commands.add_parser(COMMAND, help="generate a report from result files")
    parser.add_argument("experiment", choices=["E0"])
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--envs", nargs="+", default=["scifact", "nfcorpus", "msmarco"])
    parser.add_argument("--report-dir", type=Path, default=Path("reports/E0"))
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    try:
        path = e0.write(args.artifacts, args.envs, args.report_dir)
    except Exception as error:  # noqa: BLE001 - reported as exit 1
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(path)
    return 0
