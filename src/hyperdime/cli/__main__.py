"""``hyperdime <command>`` (also ``python -m hyperdime.cli``); every command writes a manifest."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from hyperdime.cli import (
    diagnose,
    embed,
    evaluate,
    mine_negatives,
    prepare_data,
    report,
    train_global,
    train_selector,
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
    report.register(commands)
    args = parser.parse_args(argv)
    code: int = args.handler(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
