"""``hyperdime mine-negatives``: ADR-0002 pools for training and validation queries (F2 US4).

The environment is loaded without test judgments, so no test label can reach a pool.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hyperdime.artifacts import latest_artifact
from hyperdime.cli.common import add_config_arguments, run_with_config
from hyperdime.contracts.config import RunConfig
from hyperdime.contracts.manifests import (
    ArtifactWriter,
    read_manifest,
    resolve_device,
    start_manifest,
)
from hyperdime.contracts.seeding import seed_everything
from hyperdime.contracts.validation import ContractError
from hyperdime.data.environments import spec_for
from hyperdime.data.negatives import build_negative_pools
from hyperdime.data.prepare import load_environment
from hyperdime.embeddings.cache import load_embeddings

COMMAND = "mine-negatives"
SEED = 0


def register(commands: Any) -> None:
    parser = commands.add_parser(COMMAND, help="hard-negative pools for training queries")
    add_config_arguments(parser)
    parser.add_argument("--depth", type=int, default=1000)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    return run_with_config(
        args,
        lambda config, output: _mine(config, output, args),
        validate=lambda config: spec_for(config.data),
    )


def _mine(config: RunConfig, output: Path, args: argparse.Namespace) -> Path:
    spec = spec_for(config.data)
    dataset_dir = latest_artifact(output, "datasets", spec.environment)
    bundle = load_environment(dataset_dir)  # never test or evaluation-set judgments
    embeddings_dir = latest_artifact(output, "embeddings", spec.environment)
    if read_manifest(embeddings_dir).dataset_hash != bundle.manifest.fingerprint:
        raise ContractError(f"{embeddings_dir} was not made from {dataset_dir}; run embed first")
    documents, document_vectors = load_embeddings(embeddings_dir, "documents")
    queries, query_vectors = load_embeddings(embeddings_dir, "queries")
    device = resolve_device(config.run.device)
    determinism = seed_everything(config.run.seed)
    manifest = start_manifest(
        COMMAND,
        config,
        dataset_hash=bundle.manifest.fingerprint,
        device=device,
        determinism=determinism,
    )
    pools = build_negative_pools(
        bundle,
        query_vectors,
        queries.ids,
        document_vectors,
        documents.ids,
        depth=args.depth,
        count=args.count,
        seed=SEED,
        device=device,
        chunk_size=args.chunk_size,
    )
    with ArtifactWriter(output, f"negatives/{spec.environment}", manifest) as writer:
        with writer.path("negatives.jsonl").open("w", encoding="utf-8") as handle:
            for pool in pools:
                handle.write(pool.model_dump_json() + "\n")
        inputs = {
            "dataset": dataset_dir.name,
            "embeddings": embeddings_dir.name,
            "depth": args.depth,
            "count": args.count,
            "seed": SEED,
        }
        writer.path("inputs.json").write_text(json.dumps(inputs, indent=2, sort_keys=True) + "\n")
        return writer.commit()
