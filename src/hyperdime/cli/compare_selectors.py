"""``hyperdime compare-selectors``: experiment E2, the retrieval-trained selector against L2S.

Per environment, every method is scored on the test set (and MS MARCO's TREC DL sets) at each
10% budget in one corpus pass per query set (F5 research D7): the retrieval-trained selector and
E1's L2S selector (seeds averaged later, in the report), the grid's family bests (seed 0), the
global selector, prefix, random, global importance, and the full query. All in-domain.

No model is trained or updated: checkpoints are loaded read-only and their payload hashes checked.
Test and evaluation-set judgments only score rankings; the fit diagnostic uses validation queries
with training-split labels (F4 research D2).
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from hyperdime.artifacts import latest_artifact
from hyperdime.baselines.learning_to_select import LinearSelector
from hyperdime.baselines.methods import (
    global_importance,
    prefix_importance,
    random_importance,
    selector_importance,
)
from hyperdime.cli.common import add_config_arguments, run_with_config
from hyperdime.cli.transfer_matrix import BUDGETS, SeededEnvironment, fit_results
from hyperdime.contracts.config import RunConfig
from hyperdime.contracts.hashing import NO_DATASET, file_sha256
from hyperdime.contracts.manifests import ArtifactWriter, resolve_device, start_manifest
from hyperdime.contracts.schemas import SelectorCheckpoint, dump_records, load_record
from hyperdime.contracts.seeding import seed_everything
from hyperdime.contracts.validation import ContractError
from hyperdime.evaluation.runs import DIM, Importance, QuerySet, evaluate_methods
from hyperdime.evaluation.selector_shift import overlap_at_k
from hyperdime.selection.topk import top_k_mask
from hyperdime.training.inputs import EnvironmentInputs
from hyperdime.training.selection import load_selector

COMMAND = "compare-selectors"
ENVIRONMENTS = ("scifact", "nfcorpus", "msmarco")
SEEDS = (0, 1, 2)


def register(commands: Any) -> None:
    parser = commands.add_parser(COMMAND, help="experiment E2: retrieval selector against L2S")
    add_config_arguments(parser)
    parser.add_argument("--envs", nargs="+", default=list(ENVIRONMENTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    return run_with_config(args, lambda config, output: _build(config, output, args))


def _checked(directory: Path, name: str) -> tuple[LinearSelector, str]:
    record = load_record(SelectorCheckpoint, directory / f"{name}.json")
    if file_sha256(directory / record.payload.path) != record.payload.sha256:
        raise ContractError(f"{directory}: {name} payload does not match its record")
    return load_selector(directory, name)[1], record.payload.sha256


class RetrievalSelectors:
    """One environment's seeded retrieval selectors and its grid's family bests, read-only."""

    def __init__(self, root: Path, env: SeededEnvironment, seeds: Sequence[int]) -> None:
        self.name = env.name
        self.inputs: EnvironmentInputs = env.inputs
        self.mean_target = env.mean_target
        grid_dir = latest_artifact(root, "selectors/retrieval-grid", self.name)
        grid = json.loads((grid_dir / "selection.json").read_text(encoding="utf-8"))
        self.sources: dict[str, Any] = {"grid": grid_dir.name, "retrieval": {}}
        self.selectors: dict[int, LinearSelector] = {}
        for seed in seeds:
            directory = latest_artifact(
                root, f"selectors/retrieval-seeded/{self.name}", f"seed-{seed}"
            )
            record = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
            if record["source"] != {"run": grid_dir.name, "cell": grid["selected"]}:
                raise ContractError(f"{directory} is not the latest grid's selected cell")
            self.selectors[seed] = _checked(directory, "selector")[0]
            self.sources["retrieval"][f"seed-{seed}"] = directory.name
        self.families = {
            family: _checked(grid_dir, key)[0] for family, key in grid["family_best"].items()
        }

    def scores(self, queries: Tensor) -> dict[int, Tensor]:
        found = {seed: selector_importance(queries, m) for seed, m in self.selectors.items()}
        for seed, importance in found.items():
            if not bool(torch.isfinite(importance).all()):
                raise ContractError(f"retrieval:{self.name}:s{seed} gives non-finite scores")
        return found


def overlaps(
    l2s: Mapping[int, Tensor], retrieval: Mapping[int, Tensor]
) -> dict[str, dict[str, float]]:
    """Per budget: seed-matched retrieval-L2S top-k overlap, and each selector's seed noise
    floor (mean over seed pairs), on the same queries (F5 research D8)."""

    def mean(pairs: list[tuple[Tensor, Tensor]], k: int) -> float:
        values = [float(overlap_at_k(top_k_mask(a, k), top_k_mask(b, k)).mean()) for a, b in pairs]
        return sum(values) / len(values)

    matched = [(retrieval[s], l2s[s]) for s in sorted(retrieval)]
    return {
        str(k): {
            "retrieval_l2s": mean(matched, k),
            "l2s_seeds": mean(list(itertools.combinations(l2s.values(), 2)), k),
            "retrieval_seeds": mean(list(itertools.combinations(retrieval.values(), 2)), k),
        }
        for k in BUDGETS
    }


def _build(config: RunConfig, output: Path, args: argparse.Namespace) -> Path:
    if len(set(args.seeds)) < 2:
        raise ValueError("the seed noise floor needs at least two seeds per environment")
    device = resolve_device(config.run.device)
    envs = {name: SeededEnvironment(output, name, args.seeds) for name in args.envs}
    retrieval = {name: RetrievalSelectors(output, env, args.seeds) for name, env in envs.items()}
    global_dir = latest_artifact(output, "selectors/global", "all")
    global_model = load_selector(global_dir, "global")[1]
    manifest = start_manifest(
        COMMAND,
        config,
        dataset_hash=NO_DATASET,
        device=device,
        determinism=seed_everything(config.run.seed),
    )
    with ArtifactWriter(output, "evaluation/E2/all", manifest) as writer:
        found_overlaps: dict[str, Any] = {}
        for name, env in envs.items():
            for set_name, matrix, ids, labels, threshold, depth in env.query_sets():
                queries = env.vectors(matrix, ids)
                l2s, new = env.scores(queries), retrieval[name].scores(queries)
                methods: dict[str, Importance] = {
                    "full": None,
                    "prefix": prefix_importance(queries),
                    "random": random_importance(queries, seed=0),
                    "global-importance": global_importance(queries, env.mean_target),
                    "global-selector": selector_importance(queries, global_model),
                    **{f"l2s:{name}:s{seed}": scores for seed, scores in l2s.items()},
                    **{f"retrieval:{name}:s{seed}": scores for seed, scores in new.items()},
                    **{
                        f"family:{family}:{name}": selector_importance(queries, model)
                        for family, model in retrieval[name].families.items()
                    },
                }
                pairs = [("full", DIM)] + [(m, k) for m in methods if m != "full" for k in BUDGETS]
                evaluation = evaluate_methods(
                    QuerySet(set_name, ids, queries, labels, threshold, depth),
                    methods, BUDGETS, env.inputs.documents, env.inputs.doc_ids,
                    environment=name, device=device, chunk_size=args.chunk_size, pairs=pairs,
                )  # fmt: skip
                dump_records(evaluation.results, writer.path(f"results-{name}-{set_name}.jsonl"))
                if set_name == "test":
                    found_overlaps[name] = overlaps(l2s, new)
        _write_json(writer, "overlap.json", found_overlaps)
        _write_json(
            writer, "fit.json", {"l2s": fit_results(envs), "retrieval": fit_results(retrieval)}
        )
        sources = {
            "global": global_dir.name,
            **{
                name: {
                    "l2s": {k: v for k, v in env.sources.items() if k.startswith("seed-")},
                    **{k: v for k, v in env.sources.items() if not k.startswith("seed-")},
                    **retrieval[name].sources,
                }
                for name, env in envs.items()
            },
        }
        _write_json(writer, "inputs.json", sources)
        return writer.commit()


def _write_json(writer: ArtifactWriter, name: str, payload: Any) -> None:
    writer.path(name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
