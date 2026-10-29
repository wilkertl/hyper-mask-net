"""``hyperdime describe-space data=<env> --protocol P1``: the environment's space descriptor.

Reads only the dataset manifest, the query list, and the embeddings, through
``space.inputs.load_unlabeled``: no judgments file is opened (F10 research D2). P1 also reads
training-split query embeddings, never validation or test ones (research D5). Writes the
``SpaceDescriptor`` and its payload (under P1, the P0 descriptor too: its first 18 columns), the
covariance as a diagnostic, and the inputs used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from hyperdime.cli.common import add_config_arguments, run_with_config
from hyperdime.contracts.config import RunConfig
from hyperdime.contracts.manifests import ArtifactWriter, resolve_device, start_manifest
from hyperdime.contracts.schemas import DatasetManifest, load_record
from hyperdime.contracts.seeding import seed_everything
from hyperdime.data.prepare import MANIFEST
from hyperdime.space.descriptor import FEATURES, PROTOCOLS, build_descriptors, descriptor_record
from hyperdime.space.inputs import load_unlabeled
from hyperdime.space.interaction import (
    BACKGROUND_SEED,
    BACKGROUND_SIZE,
    CALIBRATION_QUERIES,
    CALIBRATION_SEED,
    CALIBRATION_SPLIT,
    TOP_DEPTH,
    calibration_queries,
)
from hyperdime.space.streaming import QUANTILE_ROWS, QUANTILE_SEED

COMMAND = "describe-space"


def register(commands: Any) -> None:
    parser = commands.add_parser(COMMAND, help="experiment E2: an environment's space descriptor")
    add_config_arguments(parser)
    parser.add_argument("--protocol", choices=list(PROTOCOLS), default="P1")
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--calibration", type=int, default=CALIBRATION_QUERIES,
                        help="P1: at most this many training queries")  # fmt: skip
    parser.add_argument("--background", type=int, default=BACKGROUND_SIZE,
                        help="P1: background documents per calibration query")  # fmt: skip
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    return run_with_config(args, lambda config, output: _build(config, output, args))


def _build(config: RunConfig, output: Path, args: argparse.Namespace) -> Path:
    env = config.data.environment
    inputs = load_unlabeled(output, env)
    device = resolve_device(config.run.device)
    manifest = start_manifest(
        COMMAND,
        config,
        dataset_hash=load_record(DatasetManifest, inputs.dataset_dir / MANIFEST).fingerprint,
        device=device,
        determinism=seed_everything(config.run.seed),
    )
    query_ids: list[str] = []
    if args.protocol == "P1":
        query_ids = calibration_queries(inputs.splits, args.calibration)
    queries = inputs.queries(query_ids).numpy() if query_ids else None
    values, covariance, names, _ = build_descriptors(
        inputs.documents, inputs.doc_ids, args.protocol, args.chunk_size, device,
        queries=queries, query_ids=query_ids, background_size=args.background,
    )  # fmt: skip
    with ArtifactWriter(output, f"descriptors/{env}", manifest) as writer:
        for protocol in PROTOCOLS[: PROTOCOLS.index(args.protocol) + 1]:
            width = len(FEATURES[protocol])
            payload = writer.path(f"descriptor-{protocol}.npy")
            np.save(payload, values[:, :width])
            groups = sorted({group for _, group in FEATURES[protocol]})
            num_queries = 0 if protocol == "P0" else len(query_ids)
            record = descriptor_record(env, protocol, payload, names[:width], groups,
                                       len(inputs.doc_ids), num_queries)  # fmt: skip
            writer.write_record(f"descriptor-{protocol}.json", record)
        np.save(writer.path("covariance.npy"), covariance)
        sources = {
            "dataset": inputs.dataset_dir.name,
            "embeddings": inputs.embeddings_dir.name,
            "protocol": args.protocol,
            "num_documents": len(inputs.doc_ids),
            "quantile_rows": min(len(inputs.doc_ids), QUANTILE_ROWS),
            "quantile_seed": QUANTILE_SEED,
            "chunk_size": args.chunk_size,
        }
        if query_ids:
            sources["calibration"] = {
                "split": CALIBRATION_SPLIT,
                "num_queries": len(query_ids),
                "seed": CALIBRATION_SEED,
                "ids_sha256": hashlib.sha256("\n".join(query_ids).encode()).hexdigest(),
                "top_depth": TOP_DEPTH,
                "background_size": args.background,
                "background_seed": BACKGROUND_SEED,
            }
        writer.path("inputs.json").write_text(json.dumps(sources, indent=2, sort_keys=True) + "\n")
        return writer.commit()
