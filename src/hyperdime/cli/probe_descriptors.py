"""``hyperdime probe-descriptors --experiment E2``: do space descriptors predict selector shift?

Per environment: target profiles of the S1 and S4 seeded selectors on at most 10,000 validation
queries (embeddings only; judgments are never read), and the shift check (F10 research D0, D6).
With ``--targets-only`` that is all; otherwise the descriptor probes follow (D7-D10).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from hyperdime.cli.common import add_config_arguments, run_with_config
from hyperdime.contracts.config import RunConfig
from hyperdime.contracts.hashing import NO_DATASET
from hyperdime.contracts.manifests import ArtifactWriter, resolve_device, start_manifest
from hyperdime.contracts.seeding import seed_everything
from hyperdime.evaluation.descriptor_probe import shift_check
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
    return found, {
        "dataset": inputs.dataset_dir.name,
        "embeddings": inputs.embeddings_dir.name,
        "validation_queries": len(ids),
        "s1": s1_sources,
        "s4": s4_sources,
    }


def _write_json(writer: ArtifactWriter, name: str, payload: Any) -> None:
    writer.path(name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
