"""``hyperdime train-global``: one selector over every environment's training labels.

It is the anchor of the ADR-0003 ``global-regularized`` candidate (research D4). Each environment's
targets use that environment's official temperature; epochs are chosen by the mean validation
nDCG@10 across environments. No test label is read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from hyperdime.artifacts import latest_artifact
from hyperdime.cli.common import add_config_arguments, run_with_config
from hyperdime.contracts.config import RunConfig
from hyperdime.contracts.hashing import NO_DATASET, config_hash
from hyperdime.contracts.manifests import ArtifactWriter, resolve_device, start_manifest
from hyperdime.contracts.seeding import seed_everything
from hyperdime.oracle.targets import targets_at
from hyperdime.training import selection
from hyperdime.training.inputs import load_inputs
from hyperdime.training.selection import Cell, Trained, save_selector, score_on_validation
from hyperdime.training.selector_trainer import TrainConfig, train_selector

COMMAND = "train-global"
ENVIRONMENTS = ("scifact", "nfcorpus", "msmarco")


def register(commands: Any) -> None:
    parser = commands.add_parser(COMMAND, help="train the cross-environment global selector")
    add_config_arguments(parser)
    parser.add_argument("--envs", nargs="+", default=list(ENVIRONMENTS))
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    return run_with_config(args, lambda config, output: _train(config, output, args))


def _train(config: RunConfig, output: Path, args: argparse.Namespace) -> Path:
    device = resolve_device(config.run.device)
    inputs = {env: load_inputs(output, env) for env in args.envs}
    temperatures = {}
    for env in args.envs:
        official = json.loads(
            (latest_artifact(output, "selectors/official", env) / "selection.json").read_text()
        )
        temperatures[env] = official["cells"][official["chosen"]]["cell"]["temperature"]
    parts: dict[str, list[torch.Tensor]] = {"tq": [], "tt": [], "vq": [], "vt": []}
    for env, data in inputs.items():
        parts["tq"].append(data.supervision.train[0])
        parts["tt"].append(targets_at(*data.supervision.train, temperatures[env]))
        parts["vq"].append(data.supervision.validation[0])
        parts["vt"].append(targets_at(*data.supervision.validation, temperatures[env]))
    tq, tt, vq, vt = (torch.cat(parts[key]) for key in ("tq", "tt", "vq", "vt"))
    manifest = start_manifest(
        COMMAND,
        config,
        dataset_hash=NO_DATASET,
        device=device,
        determinism=seed_everything(config.run.seed),
    )
    scores: dict[int, dict[str, float]] = {}
    models = {}
    for epochs in selection.OFFICIAL_EPOCHS:
        model, history = train_selector(
            tq, tt, vq, vt, TrainConfig(epochs=epochs, seed=config.run.seed, device=device)
        )
        models[epochs] = model
        scores[epochs] = {}
        for env, data in inputs.items():
            trained = [Trained(Cell("global", temperatures[env], epochs), model, 0.0, 0.0)]
            score_on_validation(
                trained, data.supervision, data.documents, data.doc_ids,
                environment=env, device=device, chunk_size=args.chunk_size,
            )  # fmt: skip
            scores[epochs][env] = trained[0].score
    mean = {e: sum(s.values()) / len(s) for e, s in scores.items()}
    chosen = max(mean, key=lambda e: (mean[e], -e))
    with ArtifactWriter(output, "selectors/global/all", manifest) as writer:
        save_selector(
            writer,
            "global",
            models[chosen],
            kind="global",
            environment=None,
            training_manifest_sha256=config_hash(manifest.model_dump(mode="json")),
        )
        record = {
            "envs": list(args.envs),
            "temperatures": temperatures,
            "validation_ndcg@10": {str(e): s for e, s in scores.items()},
            "mean_validation_ndcg@10": {str(e): s for e, s in mean.items()},
            "chosen_epochs": chosen,
            "inputs": {env: data.dataset_dir.name for env, data in inputs.items()},
        }
        writer.path("selection.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n"
        )
        return writer.commit()
