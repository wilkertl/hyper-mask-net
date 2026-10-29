"""``hyperdime describe-space data=<env> --protocol P0``: the environment's space descriptor.

Reads only the dataset manifest, the query list, and the embeddings, through
``space.inputs.load_unlabeled``: no judgments file is opened (F10 research D2). Writes the
``SpaceDescriptor`` and its payload, the covariance as a diagnostic, and the inputs used.
"""

from __future__ import annotations

import argparse
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
from hyperdime.space.descriptor import PROTOCOLS, build_descriptors, descriptor_record
from hyperdime.space.inputs import load_unlabeled
from hyperdime.space.streaming import QUANTILE_ROWS, QUANTILE_SEED

COMMAND = "describe-space"


def register(commands: Any) -> None:
    parser = commands.add_parser(COMMAND, help="experiment E2: an environment's space descriptor")
    add_config_arguments(parser)
    parser.add_argument("--protocol", choices=list(PROTOCOLS), default="P0")
    parser.add_argument("--chunk-size", type=int, default=250_000)
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
    values, covariance, names, groups = build_descriptors(
        inputs.documents, inputs.doc_ids, args.protocol, args.chunk_size, device
    )
    with ArtifactWriter(output, f"descriptors/{env}", manifest) as writer:
        payload = writer.path(f"descriptor-{args.protocol}.npy")
        np.save(payload, values)
        record = descriptor_record(env, args.protocol, payload, names, groups, len(inputs.doc_ids))
        writer.write_record(f"descriptor-{args.protocol}.json", record)
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
        writer.path("inputs.json").write_text(json.dumps(sources, indent=2, sort_keys=True) + "\n")
        return writer.commit()
