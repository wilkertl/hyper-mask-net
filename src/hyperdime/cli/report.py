"""``hyperdime report E0|E1|S1|P1|S2``: generate a report and its criteria from result files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from hyperdime.reports import e0, e1, p1, s1, s2

COMMAND = "report"


def register(commands: Any) -> None:
    parser = commands.add_parser(COMMAND, help="generate a report from result files")
    parser.add_argument("experiment", choices=["E0", "E1", "S1", "P1", "S2"])
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--envs", nargs="+", default=["scifact", "nfcorpus", "msmarco"])
    parser.add_argument("--report-dir", type=Path, help="default: reports/<experiment>")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    try:
        report_dir = args.report_dir or Path("reports") / args.experiment
        if args.experiment == "E0":
            path = e0.write(args.artifacts, args.envs, report_dir)
        elif args.experiment == "E1":
            path = e1.write(args.artifacts, report_dir)
        elif args.experiment == "S1":
            path = s1.write(args.artifacts, report_dir)
        elif args.experiment == "P1":
            path = p1.write(args.artifacts, report_dir)
        else:
            path = s2.write(args.artifacts, report_dir)
    except Exception as error:  # noqa: BLE001 - reported as exit 1
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(path)
    return 0
