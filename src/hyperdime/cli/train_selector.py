"""``hyperdime train-selector``: per-environment selectors (F3 US2/US3, research D4).

``--recipe official`` tunes the ADR-0002 recipe on validation; ``--recipe candidates`` trains
the ADR-0003 families and records the validation-best selector of all as ``selected``;
``--recipe seeded`` retrains F3's ``wide-grid`` cell with ``run.seed`` (F4 research D1), one
artifact per seed under ``selectors/seeded/<env>/seed-<n>/``.

``--recipe retrieval-grid`` trains the F5 retrieval-objective grid with seed 0 and selects a cell
on validation (F5 research D6); ``--recipe retrieval-seeded`` retrains that cell with
``run.seed`` under ``selectors/retrieval-seeded/<env>/seed-<n>/``. ``--recipe geometric-grid``
and ``geometric-seeded`` do the same for S2's selectors that start from the corpus's geometric
prior (F7), under ``selectors/geometric-grid|geometric-seeded``, and ``weighted-grid`` and
``weighted-seeded`` for S3's, with a learned weight on that prior (F8). ``budget-grid`` and
``budget-seeded`` train S4's selector with one prior weight per budget, and
``weighted-aligned-grid`` and ``weighted-aligned-seeded`` its control, S3's selector on the same
8 cells; both keep checkpoints by full-corpus validation nDCG@10 (F9 research D3–D6).
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
from hyperdime.contracts.hashing import config_hash, file_sha256
from hyperdime.contracts.manifests import ArtifactWriter, resolve_device, start_manifest
from hyperdime.contracts.seeding import seed_everything
from hyperdime.data.environments import spec_for
from hyperdime.space.geometry import corpus_geometry
from hyperdime.training.inputs import EnvironmentInputs, load_examples, load_inputs
from hyperdime.training.retrieval_selection import (
    CANDIDATE_SEED,
    INITS,
    PRIOR_INITS,
    STEPS,
    RetrievalCell,
    TrainedRetrieval,
    best_retrieval,
    choose_candidates,
    family_bests,
    retrieval_cells,
    retrieval_summary,
    score_on_validation_budgets,
    train_retrieval_cells,
)
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
RECIPES = [
    "official", "candidates", "seeded", "retrieval-grid", "retrieval-seeded", "geometric-grid",
    "geometric-seeded", "weighted-grid", "weighted-seeded", "budget-grid", "budget-seeded",
    "weighted-aligned-grid", "weighted-aligned-seeded",
]  # fmt: skip
SEEDED_RECIPES = (
    "seeded", "retrieval-seeded", "geometric-seeded", "weighted-seeded", "budget-seeded",
    "weighted-aligned-seeded",
)  # fmt: skip
# Retrieval-objective tracks: S1 starts from scratch or L2S, S2 from the geometric prior, S3 from
# the prior with a learned weight on it; S4 with a weight per budget, against S3's selector under
# the same aligned rules.
TRACK_INITS = {
    "retrieval": INITS, "geometric": ("prior",), "weighted": ("weighted",),
    "budget": ("budget",), "weighted-aligned": ("weighted-aligned",),
}  # fmt: skip


def register(commands: Any) -> None:
    parser = commands.add_parser(COMMAND, help="train per-environment selectors on validation")
    add_config_arguments(parser)
    parser.add_argument("--recipe", choices=RECIPES, required=True)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument(
        "--steps", type=int, default=STEPS, help="retrieval recipes: steps per cell"
    )
    parser.add_argument(
        "--cells", type=int, default=None, help="retrieval-grid: train only the first N cells"
    )
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
    if args.recipe in SEEDED_RECIPES:
        kind += f"/seed-{config.run.seed}"
    with ArtifactWriter(output, kind, manifest) as writer:
        if args.recipe == "official":
            record = _official(inputs, config, device, args.chunk_size)
            models = {"official": record["model"]}
        elif args.recipe == "seeded":
            record = _seeded(inputs, output, config, device, args.chunk_size, sources)
            models = {"selector": record["model"]}
        elif args.recipe.endswith("-grid"):
            track = args.recipe.removesuffix("-grid")
            record = _retrieval_grid(inputs, output, config, device, args, sources, track)
            models = record["models"]
        elif args.recipe.endswith("-seeded") and args.recipe != "seeded":
            track = args.recipe.removesuffix("-seeded")
            record = _retrieval_seeded(inputs, output, config, device, args, sources, track)
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
        if "histories" in record:
            writer.path("history.json").write_text(
                json.dumps(record.pop("histories"), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
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


def _l2s_start(output: Path, environment: str, seed: int) -> tuple[dict[str, Any], Any]:
    """E1's seeded L2S selector of ``seed``, read-only; refuses a payload that changed."""
    directory = latest_artifact(output, f"selectors/seeded/{environment}", f"seed-{seed}")
    record, model = load_selector(directory, "selector")
    if file_sha256(directory / record.payload.path) != record.payload.sha256:
        raise ValueError(f"{directory} payload differs from its record")
    return {"run": directory.name, "sha256": record.payload.sha256}, model.state_dict()


def _prior(inputs: EnvironmentInputs, device: str, chunk_size: int) -> Any:
    """The corpus's per-dimension variance, from document embeddings only (S2)."""
    import torch

    return torch.from_numpy(corpus_geometry(inputs.documents, chunk_size, device).variance)


def _retrieval_grid(
    inputs: EnvironmentInputs,
    output: Path,
    config: RunConfig,
    device: str,
    args: argparse.Namespace,
    sources: dict[str, str],
    track: str = "retrieval",
) -> dict[str, Any]:
    """Every cell of the track's grid with ``run.seed``, scored on validation; one is selected."""
    cells = retrieval_cells(args.steps, TRACK_INITS[track])[: args.cells]
    train, validation = load_examples(inputs, "train"), load_examples(inputs, "validation")
    start_record, start = None, None
    if any(cell.init == "l2s" for cell in cells):
        start_record, start = _l2s_start(output, inputs.environment, config.run.seed)
    prior = _prior(inputs, device, args.chunk_size) if track != "retrieval" else None
    trained = train_retrieval_cells(
        cells, train, validation, seed=config.run.seed, device=device, start=start, prior=prior
    )
    aligned = _score(trained, inputs, device, args.chunk_size)
    chosen = best_retrieval(trained)
    return {
        "recipe": f"{track}-grid",
        "objective": "retrieval",
        **({"prior": "corpus variance q_d^2 Var(x_d)"} if prior is not None else {}),
        **aligned,
        "seed": config.run.seed,
        "cells": retrieval_summary(trained),
        "selected": chosen.cell.key,
        "family_best": {f: t.cell.key for f, t in family_bests(trained).items()},
        "skipped": {"train": train.skipped, "validation": validation.skipped},
        "l2s_start": start_record,
        "histories": {t.cell.key: t.history for t in trained},
        "models": {t.cell.key: t.model for t in trained},
    }


def _retrieval_seeded(
    inputs: EnvironmentInputs,
    output: Path,
    config: RunConfig,
    device: str,
    args: argparse.Namespace,
    sources: dict[str, str],
    track: str = "retrieval",
) -> dict[str, Any]:
    """The track grid's selected cell, trained once with ``run.seed`` (F5 research D9)."""
    grid_dir = latest_artifact(output, f"selectors/{track}-grid", inputs.environment)
    grid = json.loads((grid_dir / "selection.json").read_text(encoding="utf-8"))
    if grid["inputs"] != sources:
        raise ValueError(f"{grid_dir} was trained on other inputs; rerun --recipe {track}-grid")
    cell = RetrievalCell(**grid["cells"][grid["selected"]]["cell"])
    start_record, start = None, None
    if cell.init == "l2s":
        start_record, start = _l2s_start(output, inputs.environment, config.run.seed)
    prior = _prior(inputs, device, args.chunk_size) if cell.init in PRIOR_INITS else None
    train, validation = load_examples(inputs, "train"), load_examples(inputs, "validation")
    (trained,) = train_retrieval_cells(
        [cell], train, validation, seed=config.run.seed, device=device, start=start, prior=prior
    )
    aligned = _score([trained], inputs, device, args.chunk_size)
    return {
        "recipe": f"{track}-seeded",
        "objective": "retrieval",
        **aligned,
        "cell": asdict(cell),
        "seed": config.run.seed,
        "validation": retrieval_summary([trained])[cell.key],
        "source": {"run": grid_dir.name, "cell": grid["selected"]},
        "l2s_start": start_record,
        "histories": {cell.key: trained.history},
        "model": trained.model,
    }


def _score(
    trained: list[TrainedRetrieval], inputs: EnvironmentInputs, device: str, chunk_size: int
) -> dict[str, Any]:
    """Keep the aligned cells' best candidates on the validation subset, then score every kept
    state on all validation queries; returns the S4 keys of ``selection.json`` (none for F5–F8)."""
    found: dict[str, Any] = {}
    if any(t.cell.aligned for t in trained):
        ids = choose_candidates(
            trained, inputs.supervision, inputs.documents, inputs.doc_ids,
            environment=inputs.environment, device=device, chunk_size=chunk_size,
        )  # fmt: skip
        found = {
            "checkpoint_rule": "validation",
            "candidate_queries": {"seed": CANDIDATE_SEED, "ids": ids},
        }
    score_on_validation_budgets(
        trained, inputs.supervision, inputs.documents, inputs.doc_ids,
        environment=inputs.environment, device=device, chunk_size=chunk_size,
    )  # fmt: skip
    return found
