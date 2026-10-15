"""``hyperdime decide G2|S1|S2``: draft a decision and its proposed ADR from result files.

The record goes to ``decisions/<G2|S1|S2>/<run_id>/`` with a manifest; the ADR is written as
"proposed" for the researcher, who decides. An ADR that is no longer proposed is never overwritten.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from hyperdime.contracts.hashing import NO_DATASET
from hyperdime.contracts.manifests import ArtifactWriter, start_manifest
from hyperdime.reports import e1, g2, s1, s2

COMMAND = "decide"


def register(commands: Any) -> None:
    parser = commands.add_parser(COMMAND, help="draft a gate decision and its ADR")
    parser.add_argument("gate", choices=["G2", "S1", "S2"])
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--adr", type=Path, help="default: the gate's ADR under docs/decisions")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    try:
        if args.gate == "G2":
            adr = args.adr or g2.DEFAULT_ADR
            results: Any = e1.load(args.artifacts)
            record = g2.decision(results, adr)
        elif args.gate == "S1":
            adr = args.adr or s1.DEFAULT_ADR
            results = s1.load(args.artifacts)
            record = s1.decision(results, adr)
        else:
            adr = args.adr or s2.DEFAULT_ADR
            results = s2.load(args.artifacts)
            record = s2.decision(results, adr)
        manifest = start_manifest(
            COMMAND,
            {"gate": args.gate, "evaluation": results.run, "adr": adr.as_posix()},
            dataset_hash=NO_DATASET,
            seed=0,
        )
        with ArtifactWriter(args.artifacts, f"decisions/{args.gate}", manifest) as writer:
            writer.write_record("gate_decision.json", record)
            directory = writer.commit()
        {"G2": g2, "S1": s1, "S2": s2}[args.gate].write_adr(results, record, adr)
    except Exception as error:  # noqa: BLE001 - reported as exit 1
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(directory)
    print(adr)
    return 0
