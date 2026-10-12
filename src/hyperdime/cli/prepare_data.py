"""``hyperdime prepare-data``: one environment into shared records (F2 US1, contracts/cli.md)."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hyperdime.cli.common import add_config_arguments, run_with_config
from hyperdime.contracts.config import RunConfig
from hyperdime.contracts.manifests import ArtifactWriter, start_manifest
from hyperdime.contracts.seeding import seed_everything
from hyperdime.data.environments import EnvironmentSpec, EvalQuerySet, spec_for
from hyperdime.data.prepare import prepare_environment
from hyperdime.data.sources import (
    HuggingFaceSource,
    IrDatasetsQuerySet,
    QuerySetReader,
    SourceReader,
)

COMMAND = "prepare-data"
# Replaced in tests by in-memory fakes, so no test touches the network.
SOURCE_FACTORY: Callable[[EnvironmentSpec], SourceReader] = HuggingFaceSource
QUERY_SET_FACTORY: Callable[[EvalQuerySet], QuerySetReader] = IrDatasetsQuerySet


def register(commands: Any) -> None:
    parser = commands.add_parser(COMMAND, help="prepare one environment's records and splits")
    add_config_arguments(parser)
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    return run_with_config(args, _prepare, validate=lambda config: spec_for(config.data))


def _prepare(config: RunConfig, output: Path) -> Path:
    spec = spec_for(config.data)
    determinism = seed_everything(config.run.seed)
    manifest = start_manifest(COMMAND, config, determinism=determinism)
    with ArtifactWriter(output, f"datasets/{spec.environment}", manifest) as writer:
        eval_sets = {s.name: QUERY_SET_FACTORY(s) for s in spec.eval_query_sets}
        prepare_environment(spec, config.data, writer, SOURCE_FACTORY(spec), eval_sets)
        return writer.commit()
