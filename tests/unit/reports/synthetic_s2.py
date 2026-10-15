"""Synthetic `compare-selectors --experiment S2` result files for the S2 report tests."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from synthetic_s1 import BASELINES, BUDGETS, ENVS, QUERIES, SEEDS, _result

from hyperdime.contracts.hashing import NO_DATASET
from hyperdime.contracts.manifests import ArtifactWriter, start_manifest
from hyperdime.contracts.schemas import dump_records

FAMILIES = ("fixed-307-prior", "multi-budget-prior")


def _shifted(base: Sequence[float], shift: float, noise: Sequence[float] = ()) -> list[float]:
    noise = noise or [0.0] * len(base)
    return [v + shift + e for v, e in zip(base, noise, strict=True)]


def _write(directory: Path, name: str, payload: object) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build(
    root: Path,
    over_retrieval: Mapping[str, float],
    over_variance: Mapping[str, float],
    *,
    unchanged: Sequence[str] = (),
) -> Path:
    """Per query, the geometric selector scores ``over_retrieval[env]`` above the S1 selector and
    ``over_variance[env]`` above ``variance``; in ``unchanged`` environments no seed improves on
    step 0. Returns the evaluation directory."""
    sources: dict[str, object] = {"global": "global-run"}
    for env in ENVS:
        cells = {
            f"{family}-lr0.001-ts0.05-b0.1": {
                "validation_mean": 0.5, "soft_mass": 0.95, "selectable": True,
                "cell": {"init": "prior"},
            }
            for family in FAMILIES
        }  # fmt: skip
        selected = next(iter(cells))
        _write(root / "selectors" / "geometric-grid" / env / f"ggrid-{env}", "selection.json", {
            "cells": cells, "selected": selected,
            "family_best": {f: f"{f}-lr0.001-ts0.05-b0.1" for f in FAMILIES},
        })  # fmt: skip
        geometric = {}
        for seed in SEEDS:
            run = f"geometric-{env}-{seed}"
            later = 0.3 if env in unchanged else 0.6
            history = [
                {"step": 0.0, "val_rerank_ndcg10": 0.4, "soft_mass": 0.7, "beta": 1.0},
                {"step": 1000.0, "val_rerank_ndcg10": later, "soft_mass": 0.96, "beta": 0.1},
            ]
            directory = root / "selectors" / "geometric-seeded" / env / f"seed-{seed}" / run
            _write(directory, "history.json", {selected: history})
            geometric[f"seed-{seed}"] = run
        sources[env] = {
            "grid": f"grid-{env}", "geometric_grid": f"ggrid-{env}", "geometric": geometric,
            "retrieval": {f"seed-{s}": f"retrieval-{env}-{s}" for s in SEEDS},
            "l2s": {f"seed-{s}": f"seeded-{env}-{s}" for s in SEEDS},
        }  # fmt: skip
    gen = torch.Generator().manual_seed(0)
    base = (torch.rand(QUERIES, generator=gen) * 0.6 + 0.3).tolist()
    manifest = start_manifest(
        "compare-selectors", {"synthetic": True}, dataset_hash=NO_DATASET, seed=0
    )
    with ArtifactWriter(root, "evaluation/S2/all", manifest) as writer:
        for env in ENVS:
            for set_name in ("test", "trec-dl-2019") if env == "msmarco" else ("test",):
                results = [_result(env, set_name, "full", 1024, [v + 0.02 for v in base])]
                for k in BUDGETS:
                    step = (k - 307) / 1024
                    results += [
                        _result(env, set_name, m, k, _shifted(base, step - d))
                        for m, d in BASELINES.items()
                    ]
                    variance = _shifted(base, step - over_variance.get(env, 0.0))
                    results.append(_result(env, set_name, "variance", k, variance))
                    for seed in SEEDS:
                        noise = (torch.rand(QUERIES, generator=gen) * 0.01).tolist()
                        for name, shift in (
                            ("l2s", 0.0), ("retrieval", -over_retrieval.get(env, 0.0)),
                            ("geometric", 0.0),
                        ):  # fmt: skip
                            values = _shifted(base, step + shift, noise)
                            results.append(
                                _result(env, set_name, f"{name}:{env}:s{seed}", k, values)
                            )
                    results += [
                        _result(env, set_name, f"family:{f}:{env}", k, _shifted(base, step))
                        for f in FAMILIES
                    ]
                dump_records(results, writer.path(f"results-{env}-{set_name}.jsonl"))
        overlap = {
            env: {
                str(k): {"geometric_l2s": 0.5, "l2s_seeds": 0.9, "geometric_seeds": 0.8}
                for k in BUDGETS
            }
            for env in ENVS
        }
        fit_row = {"mean_gain": 0.1, "p_value": 0.001, "p_holm": 0.003, "fits": True}
        fit = {s: dict.fromkeys(ENVS, fit_row) for s in ("l2s", "retrieval", "geometric")}
        for name, payload in (
            ("overlap.json", overlap),
            ("fit.json", fit),
            ("inputs.json", sources),
        ):
            writer.path(name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return writer.commit()
