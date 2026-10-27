"""``hyperdime <command>`` (also ``python -m hyperdime.cli``); every command writes a manifest."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from hyperdime.cli import (
    compare_selectors,
    decide,
    diagnose,
    embed,
    evaluate,
    mine_negatives,
    prepare_data,
    probe_descriptors,
    probe_geometry,
    report,
    train_global,
    train_selector,
    transfer_matrix,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hyperdime", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    diagnose.register(commands)
    prepare_data.register(commands)
    embed.register(commands)
    mine_negatives.register(commands)
    train_selector.register(commands)
    train_global.register(commands)
    evaluate.register(commands)
    transfer_matrix.register(commands)
    compare_selectors.register(commands)
    probe_geometry.register(commands)
    probe_descriptors.register(commands)
    report.register(commands)
    decide.register(commands)
    args = parser.parse_args(argv)
    code: int = args.handler(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
