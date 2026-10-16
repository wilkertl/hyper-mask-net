"""``hyperdime probe-geometry``: experiment P1, what the corpus's geometry alone can do.

Per environment: the corpus's moments and covariance spectrum from document embeddings only, and
the label-free method ``variance`` (``q_d² · Var(x_d)``) scored at every 10% budget on the same
query sets as the latest S1 run, so the report can pair it with S1's selectors per query
(F6 spec). Test and evaluation-set judgments only score rankings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from hyperdime.artifacts import latest_artifact
from hyperdime.baselines.methods import variance_importance
from hyperdime.cli.common import add_config_arguments, run_with_config
from hyperdime.cli.transfer_matrix import BUDGETS, SeededEnvironment
from hyperdime.contracts.config import RunConfig
from hyperdime.contracts.hashing import NO_DATASET
from hyperdime.contracts.manifests import ArtifactWriter, resolve_device, start_manifest
from hyperdime.contracts.schemas import dump_records
from hyperdime.contracts.seeding import seed_everything
from hyperdime.evaluation.runs import DIM, Importance, QuerySet, evaluate_methods
from hyperdime.space.geometry import corpus_geometry

COMMAND = "probe-geometry"
ENVIRONMENTS = ("scifact", "nfcorpus", "msmarco")


def register(commands: Any) -> None:
    parser = commands.add_parser(COMMAND, help="experiment P1: corpus geometry as a selector")
    add_config_arguments(parser)
    parser.add_argument("--envs", nargs="+", default=list(ENVIRONMENTS))
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    return run_with_config(args, lambda config, output: _build(config, output, args))


def _build(config: RunConfig, output: Path, args: argparse.Namespace) -> Path:
    device = resolve_device(config.run.device)
    s1_run = latest_artifact(output, "evaluation/S1", "all")
    manifest = start_manifest(
        COMMAND,
        config,
        dataset_hash=NO_DATASET,
        device=device,
        determinism=seed_everything(config.run.seed),
    )
    sources: dict[str, Any] = {"s1": s1_run.name}
    with ArtifactWriter(output, "evaluation/P1/all", manifest) as writer:
        for name in args.envs:
            env = SeededEnvironment(output, name, [0])
            geometry = corpus_geometry(env.inputs.documents, args.chunk_size, device)
            _write_json(writer, f"geometry-{name}.json", {
                "num_documents": geometry.num_documents,
                "participation_ratio": geometry.participation_ratio,
                "mean_energy": geometry.mean_energy,
                "eigenvalues": geometry.eigenvalues.tolist(),
                "variance": geometry.variance.tolist(),
                "mean": geometry.mean.tolist(),
            })  # fmt: skip
            variance = torch.from_numpy(geometry.variance)
            for set_name, matrix, ids, labels, threshold, depth in env.query_sets():
                queries = env.vectors(matrix, ids)
                methods: dict[str, Importance] = {
                    "full": None,
                    "variance": variance_importance(queries, variance),
                }
                pairs = [("full", DIM)] + [("variance", k) for k in BUDGETS]
                evaluation = evaluate_methods(
                    QuerySet(set_name, ids, queries, labels, threshold, depth),
                    methods, BUDGETS, env.inputs.documents, env.inputs.doc_ids,
                    environment=name, device=device, chunk_size=args.chunk_size, pairs=pairs,
                )  # fmt: skip
                dump_records(evaluation.results, writer.path(f"results-{name}-{set_name}.jsonl"))
            sources[name] = {
                "dataset": env.inputs.dataset_dir.name,
                "embeddings": env.inputs.embeddings_dir.name,
            }
        _write_json(writer, "inputs.json", sources)
        return writer.commit()


def _write_json(writer: ArtifactWriter, name: str, payload: Any) -> None:
    writer.path(name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
