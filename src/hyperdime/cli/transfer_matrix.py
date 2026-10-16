"""``hyperdime transfer-matrix``: every seeded selector on every environment (F4 US2, US3).

Every method is scored at each 10% budget (k = 102 ... 921; k = 1024 is the full query) so the
report can show nDCG@10 against the budget; transfer cells, the fit test, and the behavior
battery stay at the headline k = 307 that gate G2 reads.

No model is trained or updated: checkpoints are loaded read-only and their payload hashes checked.
Test and evaluation-set judgments are read only to score rankings; fixed subsets and the fit test
use training labels; the behavior battery reads query embeddings only.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch import Tensor

from hyperdime.artifacts import latest_artifact
from hyperdime.baselines.learning_to_select import LinearSelector
from hyperdime.baselines.methods import (
    fixed_subset,
    global_importance,
    oracle_scores,
    prefix_importance,
    random_importance,
    rank_importance,
    selector_importance,
)
from hyperdime.cli.common import add_config_arguments, run_with_config
from hyperdime.cli.evaluate import TREC_THRESHOLD
from hyperdime.contracts.config import RunConfig
from hyperdime.contracts.hashing import NO_DATASET, file_sha256
from hyperdime.contracts.manifests import ArtifactWriter, resolve_device, start_manifest
from hyperdime.contracts.schemas import (
    EvaluationResult,
    PayloadRef,
    SelectorCheckpoint,
    SelectorProperties,
    dump_records,
    load_record,
)
from hyperdime.contracts.seeding import seed_everything
from hyperdime.contracts.validation import ContractError
from hyperdime.data.prepare import DatasetBundle, load_environment
from hyperdime.embeddings.cache import load_embeddings
from hyperdime.evaluation.runs import (
    DIM,
    METRICS,
    Importance,
    QuerySet,
    evaluate_methods,
    ratio_budgets,
)
from hyperdime.evaluation.selector_shift import (
    group_permutation_test,
    overlap_at_k,
    pair_measures,
    selection_frequency,
)
from hyperdime.evaluation.statistics import holm, paired_randomization_test
from hyperdime.evaluation.transfer import (
    ALPHA,
    degradation_tests,
    fit_gain,
    fits,
    sample_battery,
    seed_average,
    transfer_cell,
)
from hyperdime.oracle.targets import targets_at
from hyperdime.selection.topk import top_k_mask
from hyperdime.training.inputs import EnvironmentInputs, judgments, load_inputs
from hyperdime.training.selection import HEADLINE_K, load_selector

COMMAND = "transfer-matrix"
ENVIRONMENTS = ("scifact", "nfcorpus", "msmarco")
SEEDS = (0, 1, 2)
BUDGETS = tuple(k for k in ratio_budgets() if k < DIM)  # k = DIM is the unmasked query


def register(commands: Any) -> None:
    parser = commands.add_parser(COMMAND, help="evaluate every selector on every environment")
    add_config_arguments(parser)
    parser.add_argument("--envs", nargs="+", default=list(ENVIRONMENTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    return run_with_config(args, lambda config, output: _build(config, output, args))


def selector_name(environment: str, seed: int) -> str:
    return f"selector:{environment}:s{seed}"


class FitSource(Protocol):
    """What the ranking-fit test reads: training-label inputs, a global-importance target, and
    per-seed selector scores."""

    inputs: EnvironmentInputs
    mean_target: Tensor

    def scores(self, queries: Tensor) -> dict[int, Tensor]: ...


class SeededEnvironment:
    """One environment's inputs, selectors, and baselines, loaded read-only."""

    def __init__(self, root: Path, name: str, seeds: Sequence[int]) -> None:
        self.name = name
        self.inputs: EnvironmentInputs = load_inputs(root, name)
        self.bundle: DatasetBundle = load_environment(
            self.inputs.dataset_dir, include_test_qrels=True
        )
        self.selectors: dict[int, LinearSelector] = {}
        self.payloads: dict[int, str] = {}
        self.sources: dict[str, str] = {}
        for seed in seeds:
            directory = latest_artifact(root, f"selectors/seeded/{name}", f"seed-{seed}")
            record = load_record(SelectorCheckpoint, directory / "selector.json")
            if file_sha256(directory / record.payload.path) != record.payload.sha256:
                raise ContractError(f"{directory}: selector payload does not match its record")
            _, model = load_selector(directory, "selector")
            self.selectors[seed] = model
            self.payloads[seed] = record.payload.sha256
            self.sources[f"seed-{seed}"] = directory.name
        official_dir = latest_artifact(root, "selectors/official", name)
        official = json.loads((official_dir / "selection.json").read_text(encoding="utf-8"))
        temperature = official["cells"][official["chosen"]]["cell"]["temperature"]
        self.mean_target = targets_at(*self.inputs.supervision.train, temperature).mean(dim=0)
        self.sources["official"] = official_dir.name
        self.sources["dataset"] = self.inputs.dataset_dir.name
        self.sources["embeddings"] = self.inputs.embeddings_dir.name

    def scores(self, queries: Tensor) -> dict[int, Tensor]:
        """Each seed's log-importance; a non-finite output stops the run (spec edge cases)."""
        found = {seed: selector_importance(queries, m) for seed, m in self.selectors.items()}
        for seed, importance in found.items():
            if not bool(torch.isfinite(importance).all()):
                raise ContractError(f"{selector_name(self.name, seed)} gives non-finite scores")
        return found

    def fixed_subsets(self, budgets: Sequence[int]) -> dict[int, Tensor]:
        """The fixed subset at each budget, from that budget's selection frequencies."""
        seeds = list(self.scores(self.inputs.supervision.train[0]).values())
        return {k: fixed_subset([selection_frequency(s, k) for s in seeds], k) for k in budgets}

    def query_sets(self) -> list[tuple[str, str, list[str], dict[str, dict[str, int]], int, int]]:
        sets = [("test", "queries", self.bundle.query_ids("test"), judgments(self.bundle, "test"),
                 1, 100)]  # fmt: skip
        for name, queries in sorted(self.bundle.eval_queries.items()):
            sets.append((name, f"queries-{name}", sorted(queries),
                         judgments(self.bundle, name), TREC_THRESHOLD, 1000))  # fmt: skip
        return sets

    def vectors(self, matrix: str, ids: Sequence[str]) -> Tensor:
        record, vectors = load_embeddings(self.inputs.embeddings_dir, matrix)
        rows = {qid: i for i, qid in enumerate(record.ids)}
        return torch.from_numpy(np.asarray(vectors[[rows[q] for q in ids]], dtype=np.float32))


def _build(config: RunConfig, output: Path, args: argparse.Namespace) -> Path:
    if len(set(args.seeds)) < 2:
        raise ValueError("the seed noise floor needs at least two seeds per environment")
    device = resolve_device(config.run.device)
    envs = {name: SeededEnvironment(output, name, args.seeds) for name in args.envs}
    global_dir = latest_artifact(output, "selectors/global", "all")
    global_model = load_selector(global_dir, "global")[1]
    manifest = start_manifest(
        COMMAND,
        config,
        dataset_hash=NO_DATASET,
        device=device,
        determinism=seed_everything(config.run.seed),
    )
    subsets = {name: env.fixed_subsets(BUDGETS) for name, env in envs.items()}
    with ArtifactWriter(output, "evaluation/E1/all", manifest) as writer:
        scores: dict[tuple[str, str, str], dict[str, float]] = {}
        averaged: dict[tuple[str, str, str], dict[str, dict[str, float]]] = {}
        overlaps: dict[tuple[str, str, str], float] = {}
        for test_name, test in envs.items():
            for set_name, matrix, ids, labels, threshold, depth in test.query_sets():
                queries = test.vectors(matrix, ids)
                per_seed = {name: env.scores(queries) for name, env in envs.items()}
                methods: dict[str, Importance] = {
                    "full": None,
                    "prefix": prefix_importance(queries),
                    "random": random_importance(queries, seed=0),
                    "global-importance": global_importance(queries, test.mean_target),
                    "global-selector": selector_importance(queries, global_model),
                }
                for name in envs:
                    for seed, importance in per_seed[name].items():
                        methods[selector_name(name, seed)] = importance
                    methods[f"fixed:{name}"] = {
                        k: rank_importance(subset, queries) for k, subset in subsets[name].items()
                    }
                pairs = [("full", DIM)] + [(m, k) for m in methods if m != "full" for k in BUDGETS]
                evaluation = evaluate_methods(
                    QuerySet(set_name, ids, queries, labels, threshold, depth),
                    methods, BUDGETS, test.inputs.documents, test.inputs.doc_ids,
                    environment=test_name, device=device, chunk_size=args.chunk_size, pairs=pairs,
                )  # fmt: skip
                results = [_protocol(r, test_name) for r in evaluation.results]
                dump_records(results, writer.path(f"results-{test_name}-{set_name}.jsonl"))
                by_name = {r.selector: r for r in results if r.k == HEADLINE_K}
                for name in envs:
                    key = (name, test_name, set_name)
                    averaged[key] = seed_average(
                        [by_name[selector_name(name, s)].per_query for s in args.seeds]
                    )
                    scores[key] = {q: v["ndcg@10"] for q, v in averaged[key].items()}
                    overlaps[key] = _mask_overlap(per_seed[name], per_seed[test_name])
        rows = degradation_tests(scores)
        cells = [
            transfer_cell(
                row,
                _means(averaged[(row.train, row.test, row.query_set)]),
                overlaps[(row.train, row.test, row.query_set)],
                args.seeds,
                HEADLINE_K,
                [selector_name(row.train, s) for s in args.seeds],
            )
            for row in rows
        ]
        dump_records(cells, writer.path("cells.jsonl"))
        _write_json(writer, "fixed_subsets.json", {
            n: {str(k): s.tolist() for k, s in by_k.items()} for n, by_k in subsets.items()
        })  # fmt: skip
        _write_json(writer, "fit.json", fit_results(envs))
        _behavior(writer, envs)
        sources = {"global": global_dir.name, **{n: e.sources for n, e in envs.items()}}
        _write_json(writer, "inputs.json", sources)
        return writer.commit()


def _protocol(result: EvaluationResult, test: str) -> EvaluationResult:
    """Selectors and fixed subsets from another environment are transfer results."""
    source = result.selector.split(":")[1] if ":" in result.selector else test
    return result.model_copy(update={"protocol": "transfer"}) if source != test else result


def _mask_overlap(a: Mapping[int, Tensor], b: Mapping[int, Tensor]) -> float:
    """Mean overlap@k over every seed pair of two selectors on the same queries (research D7)."""
    values: list[float] = [
        float(overlap_at_k(top_k_mask(sa, HEADLINE_K), top_k_mask(sb, HEADLINE_K)).mean())
        for sa in a.values()
        for sb in b.values()
    ]
    return sum(values) / len(values)


def _means(per_query: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    return {m: sum(v[m] for v in per_query.values()) / len(per_query) for m in METRICS}


def fit_results(envs: Mapping[str, FitSource]) -> dict[str, Any]:
    """Each environment's ranking fit on validation queries, Holm over environments (F4 D2)."""
    found: dict[str, dict[str, Any]] = {}
    for name, env in envs.items():
        queries, positive, negative = env.inputs.supervision.validation
        gain = fit_gain(
            list(env.scores(queries).values()),
            global_importance(queries, env.mean_target),
            oracle_scores(queries, positive, negative),
            HEADLINE_K,
        )
        values = gain.tolist()
        found[name] = {
            "mean_gain": sum(values) / len(values),
            "p_value": paired_randomization_test(values, [0.0] * len(values)),
            "num_queries": len(values),
        }
    adjusted = holm({name: row["p_value"] for name, row in found.items()})
    for name, row in found.items():
        row["p_holm"] = adjusted[name]
        row["fits"] = fits(row["mean_gain"], adjusted[name])
    return found


def _behavior(writer: ArtifactWriter, envs: Mapping[str, SeededEnvironment]) -> None:
    """Every seeded selector on one balanced battery: properties, 36 pairs, and criterion 2."""
    battery = sample_battery({name: env.bundle.query_ids("test") for name, env in envs.items()})
    queries = torch.cat([envs[n].vectors("queries", ids) for n, ids in battery.ids.items()])
    selectors = [(name, seed) for name, env in envs.items() for seed in env.selectors]
    scores = {(n, s): t for n in envs for s, t in envs[n].scores(queries).items()}
    for (name, seed), importance in scores.items():
        stem = f"properties-{name}-s{seed}"
        payloads = {
            "mean_importance": importance.exp().mean(dim=0).double().numpy(),
            "selection_frequency": selection_frequency(importance, HEADLINE_K)[None]
            .double()
            .numpy(),
        }
        refs = {}
        for field, array in payloads.items():
            path = writer.path(f"{stem}-{field}.npy")
            np.save(path, array)
            refs[field] = PayloadRef(path=path.name, sha256=file_sha256(path), format="npy")
        writer.write_record(f"{stem}.json", SelectorProperties(
            environment=name, seed=seed,
            selector_sha256=envs[name].payloads[seed],
            battery_sha256=battery.sha256, k_values=[HEADLINE_K], **refs,
        ))  # fmt: skip
    pairs = []
    size = len(selectors)
    matrix = torch.eye(size, dtype=torch.float64)
    for i in range(size):
        for j in range(i + 1, size):
            measures = pair_measures(scores[selectors[i]], scores[selectors[j]], HEADLINE_K)
            matrix[i, j] = matrix[j, i] = measures["overlap"]
            pairs.append({
                "a": selector_name(*selectors[i]), "b": selector_name(*selectors[j]),
                "within": selectors[i][0] == selectors[j][0], **measures,
            })  # fmt: skip
    names = list(envs)
    statistic, p_value, assignments = group_permutation_test(
        matrix, [names.index(name) for name, _ in selectors]
    )
    by_env_pair = {}
    for a in names:
        for b in names[names.index(a) + 1 :]:
            by_env_pair[f"{a}|{b}"] = {
                "cross": _mean_overlap(pairs, {a, b}),
                f"within_{a}": _mean_overlap(pairs, {a}),
                f"within_{b}": _mean_overlap(pairs, {b}),
            }
    _write_json(
        writer,
        "battery.json",
        {"ids": battery.ids, "n": battery.n, "seed": battery.seed, "sha256": battery.sha256},
    )
    _write_json(writer, "behavior.json", {
        "k": HEADLINE_K, "pairs": pairs, "statistic": statistic, "p_value": p_value,
        "assignments": assignments, "holds": statistic > 0 and p_value <= ALPHA,
        "environment_pairs": by_env_pair,
    })  # fmt: skip


def _mean_overlap(pairs: Sequence[Mapping[str, Any]], environments: set[str]) -> float:
    """Mean overlap of the selector pairs whose two environments are exactly ``environments``
    (one environment: within pairs; two: cross pairs)."""
    values: list[float] = [
        p["overlap"] for p in pairs if {p["a"].split(":")[1], p["b"].split(":")[1]} == environments
    ]
    return sum(values) / len(values)


def _write_json(writer: ArtifactWriter, name: str, payload: Any) -> None:
    writer.path(name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
