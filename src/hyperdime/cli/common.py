"""Shared plumbing for commands that follow ``hyperdime diagnose`` (F1 cli contract)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hyperdime.contracts.config import RunConfig, load_config
from hyperdime.contracts.validation import ConfigError

INVALID_CONFIG = 2
FAILED = 1


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--config-name", default="experiment/prepare")
    parser.add_argument("--output", type=Path, help="artifact root (default: run.output_dir)")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. data=nfcorpus")


def run_with_config(
    args: argparse.Namespace,
    body: Callable[[RunConfig, Path], Path],
    validate: Callable[[RunConfig], Any] | None = None,
) -> int:
    """Load the config (exit 2 on any config problem, nothing written), run ``body``, and print
    the artifact directory it returns; any other failure is exit 1."""
    try:
        config = load_config(args.config_dir, args.config_name, args.overrides)
        if validate is not None:
            validate(config)
    except ConfigError as error:
        for line in str(error).splitlines():
            print(f"error: invalid config: {line}", file=sys.stderr)
        return INVALID_CONFIG
    output = args.output or Path(config.run.output_dir)
    try:
        directory = body(config, output)
    except Exception as error:  # noqa: BLE001 - reported as exit 1; writers leave nothing visible
        print(f"error: {error}", file=sys.stderr)
        return FAILED
    print(directory)
    return 0
