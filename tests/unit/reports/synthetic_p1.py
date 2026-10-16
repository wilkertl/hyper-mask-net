"""Synthetic `probe-geometry` result files on top of the synthetic S1 run."""

import json
from collections.abc import Mapping
from pathlib import Path

import torch
from synthetic_s1 import BASELINES, BUDGETS, ENVS, QUERIES, _result
from synthetic_s1 import build as build_s1

from hyperdime.contracts.hashing import NO_DATASET
from hyperdime.contracts.manifests import ArtifactWriter, start_manifest
from hyperdime.contracts.schemas import dump_records

RATIOS = {"msmarco": 300.0, "nfcorpus": 200.0, "scifact": 100.0}


def build(root: Path, shares: Mapping[str, float], *, s1_run: str | None = None) -> Path:
    """``variance`` recovers ``shares[env]`` of L2S's gain over global importance per query."""
    s1 = build_s1(root, {})
    base = (torch.rand(QUERIES, generator=torch.Generator().manual_seed(0)) * 0.6 + 0.3).tolist()
    gap = BASELINES["global-importance"]  # L2S sits this far above global importance
    manifest = start_manifest(
        "probe-geometry", {"synthetic": True}, dataset_hash=NO_DATASET, seed=0
    )
    with ArtifactWriter(root, "evaluation/P1/all", manifest) as writer:
        for env in ENVS:
            share = shares.get(env, 0.0)
            for set_name in ("test", "trec-dl-2019") if env == "msmarco" else ("test",):
                results = [
                    _result(env, set_name, "variance", k,
                            [v + (k - 307) / 1024 - gap * (1 - share) for v in base])
                    for k in BUDGETS
                ]  # fmt: skip
                dump_records(results, writer.path(f"results-{env}-{set_name}.jsonl"))
            geometry = {
                "num_documents": 1000, "participation_ratio": RATIOS[env], "mean_energy": 0.25,
                "eigenvalues": [], "variance": [], "mean": [],
            }  # fmt: skip
            writer.path(f"geometry-{env}.json").write_text(json.dumps(geometry))
        inputs = {"s1": s1_run or s1.name, **{env: {} for env in ENVS}}
        writer.path("inputs.json").write_text(json.dumps(inputs))
        return writer.commit()
