"""``hyperdime embed``: embed one prepared environment once (F2 US2/US3, contracts/cli.md).

Documents, the environment's queries (train, validation, and test), and each evaluation query set
are embedded in resumable shards under ``<output>/.work/`` and published together through
``ArtifactWriter``. An existing embedding of the same dataset with the same settings is reused;
one with different settings is refused.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from hyperdime.artifacts import latest_artifact
from hyperdime.cli.common import add_config_arguments, run_with_config
from hyperdime.contracts.config import ModelConfig, RunConfig
from hyperdime.contracts.hashing import config_hash
from hyperdime.contracts.manifests import (
    ArtifactWriter,
    read_manifest,
    resolve_device,
    start_manifest,
)
from hyperdime.contracts.schemas import QueryRecord
from hyperdime.contracts.seeding import seed_everything
from hyperdime.contracts.validation import ContractError
from hyperdime.data.environments import spec_for
from hyperdime.data.prepare import DatasetBundle, document_text, iter_documents, load_environment
from hyperdime.embeddings.cache import check_settings, embed_sharded, publish_embeddings
from hyperdime.embeddings.encoders import Encoder, build_encoder, compute_dtype, encoder_settings

COMMAND = "embed"
SETTINGS = "settings.json"
SPLIT_ORDER = ("train", "validation", "test")
# Replaced in tests by a stub encoder, so no test loads the model.
ENCODER_FACTORY: Callable[[ModelConfig, str, str, int, str | None], Encoder] = build_encoder


def register(commands: Any) -> None:
    parser = commands.add_parser(COMMAND, help="embed one prepared environment, resumably")
    add_config_arguments(parser)
    parser.add_argument("--backend", choices=["local", "vllm"], default="local")
    parser.add_argument("--base-url", default=os.environ.get("V4_EMBEDDING_BASE_URL"))
    parser.add_argument("--shard-size", type=int, default=65_536)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    return run_with_config(
        args,
        lambda config, output: _embed(config, output, args),
        validate=lambda config: spec_for(config.data),
    )


def _embed(config: RunConfig, output: Path, args: argparse.Namespace) -> Path:
    spec = spec_for(config.data)
    dataset_dir = latest_artifact(output, "datasets", spec.environment)
    bundle = load_environment(dataset_dir)
    device = resolve_device(config.run.device)
    dtype = compute_dtype(config.model, device)
    jobs = _jobs(bundle, dataset_dir)
    requested = {
        name: encoder_settings(config.model, args.backend, role, instruction, compute_dtype=dtype)
        for name, (role, instruction, _, _) in jobs.items()
    }
    existing = _reusable(output, spec.environment, bundle.manifest.fingerprint, requested)
    if existing is not None:
        print(f"{spec.environment}: reusing {existing}", file=sys.stderr)
        return existing

    encoder = ENCODER_FACTORY(config.model, args.backend, device, args.batch_size, args.base_url)
    determinism = seed_everything(config.run.seed)
    manifest = start_manifest(
        COMMAND,
        config,
        dataset_hash=bundle.manifest.fingerprint,
        device=device,
        determinism=determinism,
    )
    work_dirs = []
    with ArtifactWriter(output, f"embeddings/{spec.environment}", manifest) as writer:
        for name, (role, instruction, ids, texts) in jobs.items():
            settings = {**requested[name], "dataset": bundle.manifest.fingerprint}
            work = (
                output
                / ".work"
                / "embeddings"
                / spec.environment
                / name
                / config_hash(settings)[:16]
            )
            work_dirs.append(work)
            encode = _encoder_for(encoder, role, instruction)
            parts = embed_sharded(ids, texts(), encode, work, settings, args.shard_size)
            publish_embeddings(
                writer, name, ids, parts, role, spec.environment, instruction, config.model
            )
        writer.path(SETTINGS).write_text(
            json.dumps(requested, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        directory = writer.commit()
    for work in work_dirs:
        shutil.rmtree(work, ignore_errors=True)
    return directory


Job = tuple[str, str | None, list[str], Callable[[], Iterator[str]]]


def _jobs(bundle: DatasetBundle, dataset_dir: Path) -> dict[str, Job]:
    """What to embed: documents (streamed from disk), the environment's queries, each eval set."""
    instruction = str(bundle.manifest.spec["instruction"])
    doc_ids = [d.doc_id for d in iter_documents(dataset_dir)]
    query_ids = [q for split in SPLIT_ORDER for q in bundle.query_ids(split)]
    jobs: dict[str, Job] = {
        "documents": (
            "documents",
            None,
            doc_ids,
            lambda: (document_text(d) for d in iter_documents(dataset_dir)),
        ),
        "queries": (
            "queries",
            instruction,
            query_ids,
            lambda: (bundle.queries[q].text for q in query_ids),
        ),
    }
    for name, queries in sorted(bundle.eval_queries.items()):
        ids = sorted(queries)
        jobs[f"queries-{name}"] = ("queries", instruction, ids, _texts_of(queries, ids))
    return jobs


def _texts_of(
    queries: Mapping[str, QueryRecord], ids: Sequence[str]
) -> Callable[[], Iterator[str]]:
    return lambda: (queries[q].text for q in ids)


def _encoder_for(
    encoder: Encoder, role: str, instruction: str | None
) -> Callable[[Sequence[str]], Any]:
    if role == "queries":
        return lambda texts: encoder.encode_queries(texts, instruction)
    return encoder.encode


def _reusable(
    output: Path, environment: str, fingerprint: str, requested: dict[str, dict[str, Any]]
) -> Path | None:
    """The published embedding of this dataset if its settings match; refuse if they differ."""
    try:
        latest = latest_artifact(output, "embeddings", environment)
    except ContractError:
        return None
    if read_manifest(latest).dataset_hash != fingerprint:
        return None  # a re-prepared dataset: embed it anew
    stored: dict[str, dict[str, Any]] = json.loads((latest / SETTINGS).read_text(encoding="utf-8"))
    if set(stored) != set(requested):
        raise ContractError(
            f"{latest} holds {sorted(stored)}, but {sorted(requested)} were requested"
        )
    for name, settings in requested.items():
        check_settings(stored[name], settings)
    return latest
