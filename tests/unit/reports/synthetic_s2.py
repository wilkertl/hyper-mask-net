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
    return build_track(
        root, "S2", "geometric", "prior", FAMILIES,
        {"retrieval": over_retrieval, "variance": over_variance}, unchanged=unchanged,
    )  # fmt: skip


def build_track(
    root: Path,
    experiment: str,
    candidate: str,
    init: str,
    families: Sequence[str],
    over: Mapping[str, Mapping[str, float]],
    *,
    others: Sequence[str] = ("retrieval",),
    unchanged: Sequence[str] = (),
    extra_sources: Mapping[str, Mapping[str, object]] | None = None,
    files: Mapping[str, object] | None = None,
) -> Path:
    """``candidate`` (the last seeded track) scores ``over[reference][env]`` above each reference
    (seeded tracks in ``others``, or ``variance``) per query; in ``unchanged`` environments no seed
    improves on step 0; candidate histories carry a ``prior_weight`` when ``init`` is weighted.
    ``extra_sources`` adds entries to each environment's sources, ``files`` more result files."""
    sources: dict[str, object] = {"global": "global-run"}
    for env in ENVS:
        cells = {
            f"{family}-lr0.001-ts0.05-b0.1": {
                "validation_mean": 0.5, "soft_mass": 0.95, "selectable": True,
                "cell": {"init": init},
            }
            for family in families
        }  # fmt: skip
        selected = next(iter(cells))
        grid_dir = root / "selectors" / f"{candidate}-grid" / env / f"{candidate}grid-{env}"
        _write(grid_dir, "selection.json", {
            "cells": cells, "selected": selected,
            "family_best": {f: f"{f}-lr0.001-ts0.05-b0.1" for f in families},
        })  # fmt: skip
        seeded = {}
        for seed in SEEDS:
            run = f"{candidate}-{env}-{seed}"
            later = 0.3 if env in unchanged else 0.6
            history = [
                {"step": 0.0, "val_rerank_ndcg10": 0.4, "soft_mass": 0.7, "beta": 1.0},
                {"step": 1000.0, "val_rerank_ndcg10": later, "soft_mass": 0.96, "beta": 0.1},
            ]
            if init == "weighted":
                for row, weight in zip(history, (1.0, 0.25 + 0.25 * seed), strict=True):
                    row["prior_weight"] = weight
            directory = root / "selectors" / f"{candidate}-seeded" / env / f"seed-{seed}" / run
            _write(directory, "history.json", {selected: history})
            seeded[f"seed-{seed}"] = run
        sources[env] = {
            "grid": f"grid-{env}", f"{candidate}_grid": f"{candidate}grid-{env}",
            candidate: seeded,
            **{t: {f"seed-{s}": f"{t}-{env}-{s}" for s in SEEDS} for t in others},
            "l2s": {f"seed-{s}": f"seeded-{env}-{s}" for s in SEEDS},
            **(extra_sources or {}).get(env, {}),
        }  # fmt: skip
    gen = torch.Generator().manual_seed(0)
    base = (torch.rand(QUERIES, generator=gen) * 0.6 + 0.3).tolist()
    manifest = start_manifest(
        "compare-selectors", {"synthetic": True}, dataset_hash=NO_DATASET, seed=0
    )
    with ArtifactWriter(root, f"evaluation/{experiment}/all", manifest) as writer:
        for env in ENVS:
            for set_name in ("test", "trec-dl-2019") if env == "msmarco" else ("test",):
                results = [_result(env, set_name, "full", 1024, [v + 0.02 for v in base])]
                for k in BUDGETS:
                    step = (k - 307) / 1024
                    results += [
                        _result(env, set_name, m, k, _shifted(base, step - d))
                        for m, d in BASELINES.items()
                    ]
                    variance = _shifted(base, step - over.get("variance", {}).get(env, 0.0))
                    results.append(_result(env, set_name, "variance", k, variance))
                    for seed in SEEDS:
                        noise = (torch.rand(QUERIES, generator=gen) * 0.01).tolist()
                        shifts = [("l2s", 0.0)]
                        shifts += [(t, -over.get(t, {}).get(env, 0.0)) for t in others]
                        shifts.append((candidate, 0.0))
                        for name, shift in shifts:
                            values = _shifted(base, step + shift, noise)
                            results.append(
                                _result(env, set_name, f"{name}:{env}:s{seed}", k, values)
                            )
                    results += [
                        _result(env, set_name, f"family:{f}:{env}", k, _shifted(base, step))
                        for f in families
                    ]
                dump_records(results, writer.path(f"results-{env}-{set_name}.jsonl"))
        overlap = {
            env: {
                str(k): {f"{candidate}_l2s": 0.5, "l2s_seeds": 0.9, f"{candidate}_seeds": 0.8}
                for k in BUDGETS
            }
            for env in ENVS
        }
        fit_row = {"mean_gain": 0.1, "p_value": 0.001, "p_holm": 0.003, "fits": True}
        fit = {s: dict.fromkeys(ENVS, fit_row) for s in ("l2s", *others, candidate)}
        for name, payload in (
            ("overlap.json", overlap),
            ("fit.json", fit),
            ("inputs.json", sources),
            *(files or {}).items(),
        ):
            writer.path(name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return writer.commit()
