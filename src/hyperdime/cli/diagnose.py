"""``hyperdime diagnose``: load and validate a config and write its manifest (contracts/cli.md).

It never loads model weights, downloads data, or trains anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from hyperdime.contracts.config import load_config
from hyperdime.contracts.hashing import NO_DATASET
from hyperdime.contracts.manifests import ArtifactWriter, start_manifest
from hyperdime.contracts.seeding import seed_everything
from hyperdime.contracts.validation import ConfigError

KIND = "diagnose"


def register(commands: Any) -> None:
    parser = commands.add_parser(KIND, help="validate a config and write its manifest")
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--config-name", default="experiment/diagnose")
    parser.add_argument("--output", type=Path, help="artifact root (default: run.output_dir)")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. run.seed=7")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config_dir, args.config_name, args.overrides)
    except ConfigError as error:
        for line in str(error).splitlines():
            print(f"error: invalid config: {line}", file=sys.stderr)
        return 2
    try:
        determinism = seed_everything(config.run.seed)
        manifest = start_manifest(KIND, config, dataset_hash=NO_DATASET, determinism=determinism)
        output = args.output or Path(config.run.output_dir)
        with ArtifactWriter(output, KIND, manifest) as writer:
            directory = writer.commit()
    except Exception as error:  # noqa: BLE001 - any other failure is exit 1, nothing visible
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(directory)
    print(
        f"config {manifest.config_hash[:12]} | commit {manifest.git_commit[:12]}"
        f"{' (dirty)' if manifest.git_dirty else ''} | seed {manifest.seed} | {manifest.device}",
        file=sys.stderr,
    )
    return 0
