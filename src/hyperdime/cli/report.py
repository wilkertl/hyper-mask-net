"""``hyperdime report <experiment>``: generate a report and its criteria from result files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from hyperdime.reports import (
    dimensions,
    e0,
    e1,
    e2,
    overview,
    p1,
    p3,
    p3_oracle_l2s,
    p3_ranking,
    p3_shared,
    s1,
    s2,
    s3,
    s4,
)

COMMAND = "report"


def register(commands: Any) -> None:
    parser = commands.add_parser(COMMAND, help="generate a report from result files")
    parser.add_argument(
        "experiment",
        choices=[
            "E0",
            "E1",
            "E2",
            "S1",
            "P1",
            "P3",
            "P3-mask-ranking",
            "P3-oracle-l2s",
            "P3-shared-mask",
            "S2",
            "S3",
            "S4",
            "overview",
            "dimensions",
        ],
    )
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--envs", nargs="+", default=["scifact", "nfcorpus", "msmarco"])
    parser.add_argument(
        "--report-dir",
        type=Path,
        help=(
            "default: reports/<experiment> (P3 on non-F2 inputs: reports/P3-spike); "
            "for dimensions, the reports read and written to"
        ),
    )
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    try:
        report_dir = args.report_dir or Path("reports") / args.experiment
        if args.experiment == "dimensions":
            path = dimensions.write(args.report_dir or Path("reports"))
        elif args.experiment == "E0":
            path = e0.write(args.artifacts, args.envs, report_dir)
        elif args.experiment == "E1":
            path = e1.write(args.artifacts, report_dir)
        elif args.experiment == "E2":
            path = e2.write(args.artifacts, report_dir)
        elif args.experiment == "S1":
            path = s1.write(args.artifacts, report_dir)
        elif args.experiment == "P1":
            path = p1.write(args.artifacts, report_dir)
        elif args.experiment == "P3":
            path = p3.write(args.artifacts, args.report_dir)
        elif args.experiment == "P3-mask-ranking":
            path = p3_ranking.write(args.artifacts, report_dir)
        elif args.experiment == "P3-oracle-l2s":
            path = p3_oracle_l2s.write(args.artifacts, report_dir)
        elif args.experiment == "P3-shared-mask":
            path = p3_shared.write(args.artifacts, report_dir)
        elif args.experiment == "S2":
            path = s2.write(args.artifacts, report_dir)
        elif args.experiment == "S3":
            path = s3.write(args.artifacts, report_dir)
        elif args.experiment == "S4":
            path = s4.write(args.artifacts, report_dir)
        else:
            path = overview.write(args.artifacts, report_dir)
    except Exception as error:  # noqa: BLE001 - reported as exit 1
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(path)
    return 0
