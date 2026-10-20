"""``hyperdime compare-selectors``: experiment S1, the retrieval-trained selector against L2S.

Per environment, every method is scored on the test set (and MS MARCO's TREC DL sets) at each
10% budget in one corpus pass per query set (F5 research D7): the retrieval-trained selector and
E1's L2S selector (seeds averaged later, in the report), the grid's family bests (seed 0), the
global selector, prefix, random, global importance, and the full query. All in-domain.

No model is trained or updated: checkpoints are loaded read-only and their payload hashes checked.
Test and evaluation-set judgments only score rankings; the fit diagnostic uses validation queries
with training-split labels (F4 research D2).

S4 (F9) scores the budget selector with its per-budget importance in the same pass and adds a
validation pass for its diagnostics: the prior's share of the score, overlaps with ``variance`` and
with the correction alone, and nDCG@10 with every prior weight removed (F9 research D5).
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
from hyperdime.baselines.learning_to_select import (
    BudgetGeometricSelector,
    GeometricSelector,
    LinearSelector,
)
from hyperdime.baselines.methods import (
    budget_importance,
    global_importance,
    prefix_importance,
    random_importance,
    selector_importance,
    variance_importance,
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
from hyperdime.selection.diagnostics import (
    overlap_with_correction,
    overlap_with_variance,
    prior_share,
    without_prior,
)
from hyperdime.selection.topk import top_k_mask
from hyperdime.space.geometry import corpus_geometry
from hyperdime.training.inputs import EnvironmentInputs
from hyperdime.training.retrieval_trainer import HEADLINE_K
from hyperdime.training.selection import load_selector

COMMAND = "compare-selectors"
# Selector tracks scored beyond L2S and S1's retrieval selector; the last one is the candidate.
EXPERIMENT_TRACKS: dict[str, tuple[str, ...]] = {
    "S1": (),
    "S2": ("geometric",),
    "S3": ("geometric", "weighted"),
    "S4": ("geometric", "weighted", "weighted-aligned", "budget"),
}
# S4's tracks with a learned prior weight, diagnosed on validation queries (F9 research D5).
DIAGNOSED = ("weighted-aligned", "budget")
ENVIRONMENTS = ("scifact", "nfcorpus", "msmarco")
SEEDS = (0, 1, 2)


def register(commands: Any) -> None:
    parser = commands.add_parser(COMMAND, help="experiment S1: retrieval selector against L2S")
    add_config_arguments(parser)
    parser.add_argument("--envs", nargs="+", default=list(ENVIRONMENTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument(
        "--experiment", choices=list(EXPERIMENT_TRACKS), default="S1",
        help="S2 adds the geometric selectors (F7) and the variance method; S3 the weighted (F8); "
        "S4 the aligned weighted and the budget selectors (F9)",
    )  # fmt: skip
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

    def __init__(
        self, root: Path, env: SeededEnvironment, seeds: Sequence[int], track: str = "retrieval"
    ) -> None:
        """``track`` is ``retrieval`` (S1) or ``geometric`` (S2); its sources are recorded under
        ``grid`` and ``retrieval`` for S1 and ``geometric_grid`` and ``geometric`` for S2."""
        self.name = env.name
        self.track = track
        self.inputs: EnvironmentInputs = env.inputs
        self.mean_target = env.mean_target
        grid_dir = latest_artifact(root, f"selectors/{track}-grid", self.name)
        grid = json.loads((grid_dir / "selection.json").read_text(encoding="utf-8"))
        grid_key = "grid" if track == "retrieval" else f"{track}_grid"
        self.sources: dict[str, Any] = {grid_key: grid_dir.name, track: {}}
        self.hashes: dict[str, str] = {}
        self.selectors: dict[int, LinearSelector] = {}
        for seed in seeds:
            directory = latest_artifact(
                root, f"selectors/{track}-seeded/{self.name}", f"seed-{seed}"
            )
            record = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
            if record["source"] != {"run": grid_dir.name, "cell": grid["selected"]}:
                raise ContractError(f"{directory} is not the latest grid's selected cell")
            self.selectors[seed], self.hashes[f"seed-{seed}"] = _checked(directory, "selector")
            self.sources[track][f"seed-{seed}"] = directory.name
        self.families = {
            family: _checked(grid_dir, key)[0] for family, key in grid["family_best"].items()
        }

    def scores(self, queries: Tensor) -> dict[int, Tensor]:
        """One importance per seed; a budget selector's at the headline budget (the fit)."""
        found = {
            seed: _at(importance_of(queries, m), HEADLINE_K) for seed, m in self.selectors.items()
        }
        for seed, importance in found.items():
            if not bool(torch.isfinite(importance).all()):
                raise ContractError(f"{self.track}:{self.name}:s{seed} gives non-finite scores")
        return found

    def importance(self, queries: Tensor) -> dict[int, Importance]:
        """One importance per seed as the evaluation takes it: per budget for a budget selector."""
        found = {seed: importance_of(queries, m) for seed, m in self.selectors.items()}
        for seed, importance in found.items():
            values = importance.values() if isinstance(importance, Mapping) else [importance]
            if not all(bool(torch.isfinite(v).all()) for v in values if v is not None):
                raise ContractError(f"{self.track}:{self.name}:s{seed} gives non-finite scores")
        return found


def importance_of(queries: Tensor, model: LinearSelector) -> Importance:
    """A selector's log-importance; per budget for a budget selector (S4)."""
    if isinstance(model, BudgetGeometricSelector):
        return budget_importance(queries, model)
    return selector_importance(queries, model)


def _at(importance: Importance, k: int) -> Tensor:
    found = importance[k] if isinstance(importance, Mapping) else importance
    if found is None:
        raise ValueError("the full query has no importance")
    return found


def overlaps(
    l2s: Mapping[int, Importance], retrieval: Mapping[int, Importance], name: str = "retrieval"
) -> dict[str, dict[str, float]]:
    """Per budget: seed-matched top-k overlap of a candidate (``name``) with L2S, and each
    selector's seed noise floor (mean over seed pairs), on the same queries (F5 research D8)."""

    def mean(pairs: list[tuple[Importance, Importance]], k: int) -> float:
        values = [
            float(overlap_at_k(top_k_mask(_at(a, k), k), top_k_mask(_at(b, k), k)).mean())
            for a, b in pairs
        ]
        return sum(values) / len(values)

    matched: list[tuple[Importance, Importance]] = [
        (retrieval[s], l2s[s]) for s in sorted(retrieval)
    ]
    return {
        str(k): {
            f"{name}_l2s": mean(matched, k),
            "l2s_seeds": mean(list(itertools.combinations(l2s.values(), 2)), k),
            f"{name}_seeds": mean(list(itertools.combinations(retrieval.values(), 2)), k),
        }
        for k in BUDGETS
    }


def _build(config: RunConfig, output: Path, args: argparse.Namespace) -> Path:
    if len(set(args.seeds)) < 2:
        raise ValueError("the seed noise floor needs at least two seeds per environment")
    device = resolve_device(config.run.device)
    envs = {name: SeededEnvironment(output, name, args.seeds) for name in args.envs}
    retrieval = {name: RetrievalSelectors(output, env, args.seeds) for name, env in envs.items()}
    tracks = EXPERIMENT_TRACKS[args.experiment]
    extra_selectors = {
        track: {
            name: RetrievalSelectors(output, env, args.seeds, track=track)
            for name, env in envs.items()
        }
        for track in tracks
    }
    global_dir = latest_artifact(output, "selectors/global", "all")
    global_model = load_selector(global_dir, "global")[1]
    manifest = start_manifest(
        COMMAND,
        config,
        dataset_hash=NO_DATASET,
        device=device,
        determinism=seed_everything(config.run.seed),
    )
    with ArtifactWriter(output, f"evaluation/{args.experiment}/all", manifest) as writer:
        found_overlaps: dict[str, Any] = {}
        for name, env in envs.items():
            variance = (
                torch.from_numpy(
                    corpus_geometry(env.inputs.documents, args.chunk_size, device).variance
                )
                if tracks
                else None
            )
            for set_name, matrix, ids, labels, threshold, depth in env.query_sets():
                queries = env.vectors(matrix, ids)
                l2s, new = env.scores(queries), retrieval[name].scores(queries)
                extra: dict[str, Importance] = {}
                candidate: dict[int, Importance] = dict(new)
                if variance is not None:
                    extra["variance"] = variance_importance(queries, variance)
                for track, selectors in extra_selectors.items():
                    candidate = selectors[name].importance(queries)
                    extra.update({f"{track}:{name}:s{seed}": v for seed, v in candidate.items()})
                    extra.update({
                        f"family:{family}:{name}": importance_of(queries, model)
                        for family, model in selectors[name].families.items()
                    })  # fmt: skip
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
                    **extra,
                }
                pairs = [("full", DIM)] + [(m, k) for m in methods if m != "full" for k in BUDGETS]
                evaluation = evaluate_methods(
                    QuerySet(set_name, ids, queries, labels, threshold, depth),
                    methods, BUDGETS, env.inputs.documents, env.inputs.doc_ids,
                    environment=name, device=device, chunk_size=args.chunk_size, pairs=pairs,
                )  # fmt: skip
                dump_records(evaluation.results, writer.path(f"results-{name}-{set_name}.jsonl"))
                if set_name == "test":
                    found_overlaps[name] = overlaps(
                        l2s, candidate, name=tracks[-1] if tracks else "retrieval"
                    )
        if args.experiment == "S4":
            diagnosed = {t: extra_selectors[t] for t in DIAGNOSED}
            diagnostics, ablation = _diagnose(diagnosed, envs, device, args.chunk_size)
            _write_json(writer, "diagnostics.json", diagnostics)
            _write_json(writer, "ablation.json", ablation)
            _write_json(writer, "sources.json", {
                name: {
                    track: {
                        seed: {"run": run, "sha256": selectors[name].hashes[seed]}
                        for seed, run in selectors[name].sources[track].items()
                    }
                    for track, selectors in extra_selectors.items()
                }
                for name in envs
            })  # fmt: skip
        _write_json(writer, "overlap.json", found_overlaps)
        fit = {"l2s": fit_results(envs), "retrieval": fit_results(retrieval)}
        for track, selectors in extra_selectors.items():
            fit[track] = fit_results(selectors)
        _write_json(writer, "fit.json", fit)
        sources = {
            "global": global_dir.name,
            **{
                name: {
                    "l2s": {k: v for k, v in env.sources.items() if k.startswith("seed-")},
                    **{k: v for k, v in env.sources.items() if not k.startswith("seed-")},
                    **retrieval[name].sources,
                    **{
                        key: value
                        for selectors in extra_selectors.values()
                        for key, value in selectors[name].sources.items()
                    },
                }
                for name, env in envs.items()
            },
        }
        _write_json(writer, "inputs.json", sources)
        return writer.commit()


def _diagnose(
    tracks: Mapping[str, Mapping[str, RetrievalSelectors]],
    envs: Mapping[str, SeededEnvironment],
    device: str,
    chunk_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Per track, environment, seed, and budget, on validation queries only (F9 research D5):
    the prior's share, overlaps with ``variance`` and with the correction alone, the prior weight,
    and exact validation nDCG@10 of the kept state and of the same state without its prior."""
    diagnostics: dict[str, Any] = {}
    ablation: dict[str, Any] = {}
    for name, env in envs.items():
        supervision = env.inputs.supervision
        queries = supervision.validation[0]
        methods: dict[str, Importance] = {}
        for track, by_env in tracks.items():
            for seed, model in by_env[name].selectors.items():
                if not isinstance(model, GeometricSelector):
                    raise ContractError(f"{track}:{name}:s{seed} has no geometric prior")
                diagnostics.setdefault(track, {}).setdefault(name, {})[f"seed-{seed}"] = {
                    str(k): _diagnostics(model, queries, k) for k in BUDGETS
                }
                methods[f"{track}:s{seed}:kept"] = importance_of(queries, model)
                methods[f"{track}:s{seed}:no-prior"] = importance_of(queries, without_prior(model))
        evaluation = evaluate_methods(
            QuerySet("validation", supervision.validation_ids, queries,
                     supervision.validation_judgments),
            methods, BUDGETS, env.inputs.documents, env.inputs.doc_ids,
            environment=name, device=device, chunk_size=chunk_size,
        )  # fmt: skip
        for r in evaluation.results:
            track, tag, state = r.selector.split(":")
            seeds = ablation.setdefault(track, {}).setdefault(name, {})
            seeds.setdefault(f"seed-{tag[1:]}", {}).setdefault(str(r.k), {})[state] = r.metrics[
                "ndcg@10"
            ]
    return diagnostics, ablation


def _diagnostics(model: GeometricSelector, queries: Tensor, k: int) -> dict[str, float]:
    weight = model.weight_at(k)
    return {
        "prior_share": prior_share(model, queries, k),
        "overlap_variance": overlap_with_variance(model, queries, k),
        "overlap_correction": overlap_with_correction(model, queries, k),
        "prior_weight": 1.0 if weight is None else float(weight.detach()),
    }


def _write_json(writer: ArtifactWriter, name: str, payload: Any) -> None:
    writer.path(name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
