"""``hyperdime probe-descriptors --experiment E2``: do space descriptors predict selector shift?

Per environment: target profiles of the S1 and S4 seeded selectors on at most 10,000 validation
queries (embeddings only; judgments are never read), and the shift check (F10 research D0, D6).
With ``--targets-only`` that is all; otherwise the descriptor probes follow (D7-D10), on the
latest descriptor of each environment, with the environment-level context of D11.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from hyperdime.artifacts import latest_artifact
from hyperdime.baselines.learning_to_select import BUDGETS, BudgetGeometricSelector
from hyperdime.cli.common import add_config_arguments, run_with_config
from hyperdime.contracts.config import RunConfig
from hyperdime.contracts.hashing import NO_DATASET
from hyperdime.contracts.manifests import ArtifactWriter, resolve_device, start_manifest
from hyperdime.contracts.schemas import SpaceDescriptor, TransferCell, load_record, load_records
from hyperdime.contracts.seeding import seed_everything
from hyperdime.contracts.validation import ContractError
from hyperdime.evaluation.descriptor_probe import e2_probes, shift_check
from hyperdime.evaluation.targets import (
    corpus_variance,
    seed_profiles,
    seeded_selectors,
    target_profiles,
    validation_sample,
)
from hyperdime.space.inputs import load_unlabeled

COMMAND = "probe-descriptors"
ENVIRONMENTS = ("scifact", "nfcorpus", "msmarco")


def register(commands: Any) -> None:
    parser = commands.add_parser(COMMAND, help="experiment E2: descriptors against selector shift")
    add_config_arguments(parser)
    parser.add_argument("--experiment", choices=["E2"], required=True)
    parser.add_argument("--envs", nargs="+", default=list(ENVIRONMENTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--targets-only", action="store_true", help="targets and the shift check")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    return run_with_config(args, lambda config, output: _build(config, output, args))


def _build(config: RunConfig, output: Path, args: argparse.Namespace) -> Path:
    if len(set(args.envs)) < 3:
        raise ValueError("E2 holds one environment out at a time and needs at least three")
    if len(set(args.seeds)) < 2:
        raise ValueError("the noise ceiling needs at least two seeds per environment")
    manifest = start_manifest(
        COMMAND,
        config,
        dataset_hash=NO_DATASET,
        device=resolve_device(config.run.device),
        determinism=seed_everything(config.run.seed),
    )
    sources: dict[str, Any] = {
        "environments": args.envs,
        "targets_only": args.targets_only,
        "seeds": args.seeds,
    }
    profiles: dict[str, dict[str, np.ndarray]] = {}
    with ArtifactWriter(output, "evaluation/E2/all", manifest) as writer:
        for env in args.envs:
            profiles[env], sources[env] = _targets(output, env, args.seeds)
            keys = sorted(profiles[env])
            np.save(writer.path(f"targets-{env}.npy"), np.stack([profiles[env][k] for k in keys]))
            _write_json(writer, f"targets-{env}.json", {"keys": keys, **sources[env]})
        check = shift_check({env: seed_profiles(p) for env, p in profiles.items()})
        _write_json(writer, "check.json", check)
        if not args.targets_only:
            descriptors, names, sources["descriptors"] = _descriptors(output, sources)
            rows, coefficients = e2_probes(
                descriptors, names, profiles, args.seeds, BUDGETS, args.permutations
            )
            writer.path("probe.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
            )
            _write_json(writer, "coefficients.json", coefficients)
            _write_json(writer, "context.json", _context(output, descriptors, names, sources))
        _write_json(writer, "inputs.json", sources)
        return writer.commit()


def _targets(
    root: Path, env: str, seeds: list[int]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    inputs = load_unlabeled(root, env)
    ids = validation_sample(inputs.splits["validation"])
    s1, s1_sources = seeded_selectors(root, env, "retrieval", seeds)
    s4, s4_sources = seeded_selectors(root, env, "budget", seeds)
    found = target_profiles(s1, s4, corpus_variance(s4), inputs.queries(ids))  # type: ignore[arg-type]
    weights = [m.prior_weights.detach().double().numpy() for m in s4.values()
               if isinstance(m, BudgetGeometricSelector)]  # fmt: skip
    return found, {
        "dataset": inputs.dataset_dir.name,
        "embeddings": inputs.embeddings_dir.name,
        "validation_queries": len(ids),
        "s1": s1_sources,
        "s4": s4_sources,
        "s4_prior_weights": np.mean(weights, axis=0).tolist(),
    }


def _descriptors(
    root: Path, sources: dict[str, Any]
) -> tuple[dict[str, np.ndarray], list[str], dict[str, Any]]:
    """The latest descriptor of each environment, made from the embeddings the targets used; the
    richest protocol it holds (P1 over P0)."""
    found: dict[str, np.ndarray] = {}
    names: list[str] | None = None
    used: dict[str, Any] = {}
    for env in sources["environments"]:
        directory = latest_artifact(root, "descriptors", env)
        made_from = json.loads((directory / "inputs.json").read_text(encoding="utf-8"))
        if made_from["embeddings"] != sources[env]["embeddings"]:
            raise ContractError(f"{directory} was not made from the current embeddings of {env}")
        protocol = "P1" if (directory / "descriptor-P1.json").is_file() else "P0"
        record = load_record(SpaceDescriptor, directory / f"descriptor-{protocol}.json")
        if names is not None and record.feature_names != names:
            raise ContractError(f"{directory} has other descriptor columns than {names}")
        names = record.feature_names
        found[env] = np.load(directory / record.payload.path)
        used[env] = {"run": directory.name, "protocol": protocol}
    assert names is not None
    return found, names, used


def _context(
    root: Path, descriptors: dict[str, np.ndarray], names: list[str], sources: dict[str, Any]
) -> dict[str, Any]:
    """Research D11, descriptive only: per environment, the covariance's participation ratio and
    mean energy, the descriptor column means, S4's prior weights, and E1's degradation."""
    try:
        cells = load_records(
            TransferCell, latest_artifact(root, "evaluation/E1", "all") / "cells.jsonl"
        )
    except ContractError:
        cells = []
    context: dict[str, Any] = {}
    for env, values in descriptors.items():
        run = sources["descriptors"][env]["run"]
        covariance = np.load(root / "descriptors" / env / run / "covariance.npy")
        eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
        mean_square = float(np.sum(values[:, names.index("doc_mean")] ** 2))
        context[env] = {
            "participation_ratio": float(eigenvalues.sum() ** 2 / np.sum(eigenvalues**2)),
            "mean_energy": mean_square / (mean_square + float(np.trace(covariance))),
            "descriptor_means": dict(zip(names, values.mean(axis=0).tolist(), strict=True)),
            "s4_prior_weights": sources[env]["s4_prior_weights"],
            "e1_degradation": {
                c.train_environment: c.degradation
                for c in cells
                if c.test_environment == env and c.query_set == "test"
            },
        }
    return context


def _write_json(writer: ArtifactWriter, name: str, payload: Any) -> None:
    writer.path(name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
