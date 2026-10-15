"""``hyperdime train-selector``: per-environment selectors (F3 US2/US3, research D4).

``--recipe official`` tunes the ADR-0002 recipe on validation; ``--recipe candidates`` trains
the ADR-0003 families and records the validation-best selector of all as ``selected``;
``--recipe seeded`` retrains F3's ``wide-grid`` cell with ``run.seed`` (F4 research D1), one
artifact per seed under ``selectors/seeded/<env>/seed-<n>/``.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from hyperdime.artifacts import latest_artifact
from hyperdime.cli.common import add_config_arguments, run_with_config
from hyperdime.contracts.config import RunConfig
from hyperdime.contracts.hashing import config_hash
from hyperdime.contracts.manifests import ArtifactWriter, resolve_device, start_manifest
from hyperdime.contracts.seeding import seed_everything
from hyperdime.data.environments import spec_for
from hyperdime.training.inputs import EnvironmentInputs, load_inputs
from hyperdime.training.selection import (
    Cell,
    Trained,
    best,
    candidate_cells,
    load_selector,
    official_cells,
    save_selector,
    score_on_validation,
    summary,
    train_cells,
)

COMMAND = "train-selector"


def register(commands: Any) -> None:
    parser = commands.add_parser(COMMAND, help="train per-environment selectors on validation")
    add_config_arguments(parser)
    parser.add_argument("--recipe", choices=["official", "candidates", "seeded"], required=True)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    return run_with_config(
        args,
        lambda config, output: _train(config, output, args),
        validate=lambda config: spec_for(config.data),
    )


def _train(config: RunConfig, output: Path, args: argparse.Namespace) -> Path:
    environment = spec_for(config.data).environment
    inputs = load_inputs(output, environment)
    device = resolve_device(config.run.device)
    manifest = start_manifest(
        COMMAND,
        config,
        dataset_hash=inputs.bundle.manifest.fingerprint,
        device=device,
        determinism=seed_everything(config.run.seed),
    )
    fingerprint = config_hash(manifest.model_dump(mode="json"))
    sources = {
        "dataset": inputs.dataset_dir.name,
        "embeddings": inputs.embeddings_dir.name,
        "negatives": inputs.negatives_dir.name,
    }
    kind = f"selectors/{args.recipe}/{environment}"
    if args.recipe == "seeded":
        kind += f"/seed-{config.run.seed}"
    with ArtifactWriter(output, kind, manifest) as writer:
        if args.recipe == "official":
            record = _official(inputs, config, device, args.chunk_size)
            models = {"official": record["model"]}
        elif args.recipe == "seeded":
            record = _seeded(inputs, output, config, device, args.chunk_size, sources)
            models = {"selector": record["model"]}
        else:
            record = _candidates(inputs, output, config, device, args.chunk_size, sources)
            models = record["models"]
        for name, model in models.items():
            save_selector(
                writer,
                name,
                model,
                kind="per_domain",
                environment=environment,
                training_manifest_sha256=fingerprint,
            )
        selection = {key: value for key, value in record.items() if key not in ("model", "models")}
        selection["inputs"] = sources
        writer.path("selection.json").write_text(
            json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return writer.commit()


def _scored(cells: list[Cell], inputs: EnvironmentInputs, config: RunConfig, device: str,
            chunk_size: int, anchor: Any = None) -> list[Trained]:  # fmt: skip
    trained = train_cells(
        cells, inputs.supervision, seed=config.run.seed, device=device, anchor=anchor
    )
    score_on_validation(
        trained, inputs.supervision, inputs.documents, inputs.doc_ids,
        environment=inputs.environment, device=device, chunk_size=chunk_size,
    )  # fmt: skip
    return trained


def _official(
    inputs: EnvironmentInputs, config: RunConfig, device: str, chunk_size: int
) -> dict[str, Any]:
    trained = _scored(official_cells(), inputs, config, device, chunk_size)
    chosen = best(trained)
    return {
        "recipe": "official",
        "cells": summary(trained),
        "chosen": chosen.cell.key,
        "model": chosen.model,
    }


def _candidates(
    inputs: EnvironmentInputs,
    output: Path,
    config: RunConfig,
    device: str,
    chunk_size: int,
    sources: dict[str, str],
) -> dict[str, Any]:
    official_dir = latest_artifact(output, "selectors/official", inputs.environment)
    official = json.loads((official_dir / "selection.json").read_text(encoding="utf-8"))
    if official["inputs"] != sources:
        raise ValueError(f"{official_dir} was trained on other inputs; rerun --recipe official")
    official_cell = Cell(**official["cells"][official["chosen"]]["cell"])
    official_score = official["cells"][official["chosen"]]["validation_ndcg@10"]
    _, anchor_model = load_selector(latest_artifact(output, "selectors/global", "all"), "global")
    trained = _scored(
        candidate_cells(official_cell), inputs, config, device, chunk_size,
        anchor=anchor_model.state_dict(),
    )  # fmt: skip
    families: dict[str, Trained] = {}
    for t in trained:
        families.setdefault(t.cell.family, t)
        if t.score > families[t.cell.family].score:
            families[t.cell.family] = t
    models = {family: t.model for family, t in families.items()}
    scores = {"official": official_score, **{f: t.score for f, t in families.items()}}
    selected = max(scores, key=lambda family: (scores[family], family == "official"))
    models["selected"] = (
        load_selector(official_dir, "official")[1] if selected == "official" else models[selected]
    )
    return {
        "recipe": "candidates",
        "official": {
            "run": official_dir.name,
            "chosen": official["chosen"],
            "validation_ndcg@10": official_score,
        },
        "cells": summary(trained),
        "family_best": {f: t.cell.key for f, t in families.items()},
        "family_scores": scores,
        "selected": selected,
        "models": models,
    }


SEEDED_FAMILY = "wide-grid"


def _seeded(
    inputs: EnvironmentInputs,
    output: Path,
    config: RunConfig,
    device: str,
    chunk_size: int,
    sources: dict[str, str],
) -> dict[str, Any]:
    """F3's validation-chosen ``wide-grid`` cell, trained once with ``run.seed``."""
    candidates_dir = latest_artifact(output, "selectors/candidates", inputs.environment)
    candidates = json.loads((candidates_dir / "selection.json").read_text(encoding="utf-8"))
    if candidates["inputs"] != sources:
        raise ValueError(f"{candidates_dir} was trained on other inputs; rerun F3 first")
    key = candidates["family_best"][SEEDED_FAMILY]
    cell = Cell(**candidates["cells"][key]["cell"])
    (trained,) = _scored([cell], inputs, config, device, chunk_size)
    if not all(math.isfinite(v) for v in (trained.score, trained.val_kl)):
        raise ValueError(f"seed {config.run.seed} of {key} gives a non-finite validation result")
    return {
        "recipe": "seeded",
        "family": SEEDED_FAMILY,
        "cell": asdict(cell),
        "seed": config.run.seed,
        "validation_ndcg@10": trained.score,
        "val_kl": trained.val_kl,
        "uniform_kl": trained.uniform_kl,
        "source": {
            "run": candidates_dir.name,
            "cell": key,
            "validation_ndcg@10": candidates["cells"][key]["validation_ndcg@10"],
            "selected": candidates["selected"],
            "family_scores": candidates["family_scores"],
        },
        "model": trained.model,
    }
