"""Synthetic `compare-selectors` result files for the E2 report and decision tests."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from hyperdime.contracts.hashing import NO_DATASET
from hyperdime.contracts.manifests import ArtifactWriter, start_manifest
from hyperdime.contracts.schemas import EvaluationResult, dump_records
from hyperdime.evaluation.runs import METRICS

ENVS = ("msmarco", "nfcorpus", "scifact")
BUDGETS = (102, 307, 512)
SEEDS = (0, 1, 2)
FAMILIES = ("fixed-307-scratch", "multi-budget-l2s")
BASELINES = {"prefix": 0.15, "random": 0.2, "global-importance": 0.1, "global-selector": 0.05}
QUERIES = 40


def _result(
    env: str, set_name: str, method: str, k: int, values: Sequence[float]
) -> EvaluationResult:
    per_query = {f"q{i}": dict.fromkeys(METRICS, v) for i, v in enumerate(values)}
    means = {m: sum(values) / len(values) for m in METRICS}
    return EvaluationResult(
        environment=env, protocol="in-domain", split="test", selector=method, k=k,
        metrics=means, per_query=per_query, num_queries=len(values),
    )  # fmt: skip


def _write(directory: Path, name: str, payload: object) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build(root: Path, gains: Mapping[str, float], *, seeds: Sequence[int] = SEEDS) -> Path:
    """The retrieval selector scores ``gains[env]`` above L2S on every query of ``env``, with
    seed noise; L2S equals the baseline. Returns the evaluation directory."""
    sources: dict[str, object] = {"global": "global-run"}
    for env in ENVS:
        cells = {
            f"{family}-lr0.001-ts0.05-b0.1": {
                "validation_mean": 0.5 + i / 10, "soft_mass": 0.95, "selectable": True,
            }
            for i, family in enumerate(FAMILIES)
        }  # fmt: skip
        selected = next(iter(cells))
        _write(root / "selectors" / "retrieval-grid" / env / f"grid-{env}", "selection.json", {
            "cells": cells, "selected": selected,
            "family_best": {f: f"{f}-lr0.001-ts0.05-b0.1" for f in FAMILIES},
        })  # fmt: skip
        retrieval = {}
        for seed in seeds:
            run = f"retrieval-{env}-{seed}"
            history = [
                {"step": 0.0, "val_rerank_ndcg10": 0.4, "soft_mass": 0.7, "beta": 1.0},
                {"step": 1000.0, "val_rerank_ndcg10": 0.6, "soft_mass": 0.96, "beta": 0.1},
            ]
            directory = root / "selectors" / "retrieval-seeded" / env / f"seed-{seed}" / run
            _write(directory, "history.json", {selected: history})
            retrieval[f"seed-{seed}"] = run
        sources[env] = {
            "grid": f"grid-{env}",
            "retrieval": retrieval,
            "l2s": {f"seed-{s}": f"seeded-{env}-{s}" for s in seeds},
        }
    gen = torch.Generator().manual_seed(0)
    base = (torch.rand(QUERIES, generator=gen) * 0.6 + 0.3).tolist()
    manifest = start_manifest(
        "compare-selectors", {"synthetic": True}, dataset_hash=NO_DATASET, seed=0
    )
    with ArtifactWriter(root, "evaluation/E2/all", manifest) as writer:
        for env in ENVS:
            for set_name in ("test", "trec-dl-2019") if env == "msmarco" else ("test",):
                results = [_result(env, set_name, "full", 1024, [v + 0.02 for v in base])]
                for k in BUDGETS:
                    step = (k - 307) / 1024
                    results += [
                        _result(env, set_name, m, k, [v - d + step for v in base])
                        for m, d in BASELINES.items()
                    ]
                    for seed in seeds:
                        noise = (torch.rand(QUERIES, generator=gen) * 0.01).tolist()
                        results.append(_result(env, set_name, f"l2s:{env}:s{seed}", k, [
                            v + step + e for v, e in zip(base, noise, strict=True)
                        ]))  # fmt: skip
                        results.append(_result(env, set_name, f"retrieval:{env}:s{seed}", k, [
                            v + step + gains.get(env, 0.0) + e
                            for v, e in zip(base, noise, strict=True)
                        ]))  # fmt: skip
                    results += [
                        _result(env, set_name, f"family:{f}:{env}", k, [v + step for v in base])
                        for f in FAMILIES
                    ]
                dump_records(results, writer.path(f"results-{env}-{set_name}.jsonl"))
        overlap = {
            env: {
                str(k): {"retrieval_l2s": 0.5, "l2s_seeds": 0.9, "retrieval_seeds": 0.8}
                for k in BUDGETS
            }
            for env in ENVS
        }
        fit_row = {"mean_gain": 0.1, "p_value": 0.001, "p_holm": 0.003, "fits": True}
        fit = {s: dict.fromkeys(ENVS, fit_row) for s in ("l2s", "retrieval")}
        for name, payload in (
            ("overlap.json", overlap),
            ("fit.json", fit),
            ("inputs.json", sources),
        ):
            writer.path(name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return writer.commit()
