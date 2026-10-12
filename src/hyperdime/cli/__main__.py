"""``hyperdime <command>`` (also ``python -m hyperdime.cli``); every command writes a manifest."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from hyperdime.cli import diagnose, prepare_data


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hyperdime", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    diagnose.register(commands)
    prepare_data.register(commands)
    args = parser.parse_args(argv)
    code: int = args.handler(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
