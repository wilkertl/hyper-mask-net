"""Exact positive masks beside L2S, end to end on a tiny corpus (ADR-0017)."""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hyperdime.artifacts import latest_artifact
from hyperdime.baselines.learning_to_select import LinearSelector
from hyperdime.cli.__main__ import main
from hyperdime.contracts.manifests import ArtifactWriter, start_manifest
from hyperdime.contracts.schemas import NegativePool, dump_records
from hyperdime.evaluation.oracle_comparison import ARMS
from hyperdime.training.selection import save_selector

REPOSITORY = Path(__file__).resolve().parents[2]
CONFIGS = REPOSITORY / "configs"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PIPELINE = _load(REPOSITORY / "tests/integration/test_p3_pipeline.py", "p3_oracle_fixtures")
ENV = PIPELINE.ENV
TRAINING = "training-seed-recorded"


def _clean_manifest(command: str, fingerprint: str) -> Any:
    config = {"model": {"revision": PIPELINE.REVISION}, "data": {"environment": ENV}}
    start = start_manifest(command, config, seed=0, dataset_hash=fingerprint)
    return start.model_copy(update={"git_dirty": False})


def _checkpoint(root: Path, dataset: Path, embeddings: Path) -> Path:
    from hyperdime.contracts.manifests import read_manifest

    fingerprint = read_manifest(embeddings).dataset_hash
    with ArtifactWriter(root, f"negatives/{ENV}", _clean_manifest("t", fingerprint)) as writer:
        dump_records(
            [
                NegativePool(
                    query_id="q0",
                    environment=ENV,
                    split="train",
                    doc_ids=["d0"],
                    depth=1,
                    pool_size=1,
                    seed=0,
                )
            ],
            writer.path("negatives.jsonl"),
        )
        negatives = writer.commit()
    manifest = _clean_manifest("train-selector", fingerprint)
    with ArtifactWriter(root, f"selectors/official/{ENV}", manifest) as writer:
        save_selector(
            writer, "official", LinearSelector(), kind="per_domain", environment=ENV,
            training_manifest_sha256="0" * 64,
        )  # fmt: skip
        writer.path("selection.json").write_text(
            json.dumps({
                "recipe": "official", "chosen": "c", "cells": {"c": {}},
                "inputs": {"dataset": dataset.name, "embeddings": embeddings.name,
                           "negatives": negatives.name},
            })
        )  # fmt: skip
        return writer.commit()


def _analyze(root: Path, *extra: str) -> int:
    return main([
        "analyze-oracle-l2s", "--config-dir", str(CONFIGS), "--output", str(root),
        "run.device=cpu", "--chunk-docs", "16", *extra, "--envs", ENV,
    ])  # fmt: skip


@pytest.fixture(scope="module")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    found = tmp_path_factory.mktemp("oracle")
    dataset, embeddings = PIPELINE.build_inputs(found)
    _checkpoint(found, dataset, embeddings)
    assert _analyze(found) == 0
    return found


@pytest.fixture(scope="module")
def run(root: Path) -> Path:
    return latest_artifact(root, "evaluation/P3-oracle-l2s", "all")


def _summary(run: Path) -> dict[str, Any]:
    return json.loads((run / ENV / "summary.json").read_text())


def test_provenance_is_recorded_and_labels_the_sample(run: Path) -> None:
    provenance = _summary(run)["provenance"]
    assert provenance["failed"] == []
    assert provenance["sample"].startswith("in-sample")


def test_a_lone_positive_shares_its_mask_with_the_oracle(run: Path) -> None:
    checks = _summary(run)["checks"]
    assert checks["single_positive_queries"] > 0
    assert checks["single_positive_masks_identical_to_oracle"] == checks["single_positive_queries"]


def test_invariants_hold_on_the_stored_run(run: Path) -> None:
    checks = _summary(run)["checks"]
    assert checks["ranks_out_of_bounds"] == 0
    assert checks["oracle_ceiling_violations"] == 0
    assert checks["k1024_rank_mismatches"] == 0


def test_float32_evaluation_agrees_with_the_exact_scan(run: Path) -> None:
    assert _summary(run)["checks"]["float32_evaluation_gap"]["largest_ndcg10_gap"] < 1e-5


def test_every_query_has_both_cohort_arms_and_held_only_where_eligible(run: Path) -> None:
    with np.load(run / ENV / "per-query.npz") as found:
        ndcg = found["ndcg@10"]
    with np.load(run / ENV / "selection.npz") as selection:
        multi = selection["strata"] == 2
    assert np.isfinite(ndcg[ARMS.index("oracle"), 0]).all()
    assert np.isfinite(ndcg[ARMS.index("l2s"), 0]).all()
    held = np.isfinite(ndcg[ARMS.index("held"), 0])
    assert held.tolist() == multi.tolist()


def test_a_tampered_checkpoint_stops_the_run_before_any_scan(tmp_path: Path, root: Path) -> None:
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(root, copy)
    official = latest_artifact(copy, "selectors/official", ENV)
    (official / "selection.json").write_text(
        (official / "selection.json").read_text().replace("official", "other", 1)
    )
    before = {p.name for p in (copy / "evaluation/P3-oracle-l2s/all").iterdir()}
    assert _analyze(copy) != 0
    assert {p.name for p in (copy / "evaluation/P3-oracle-l2s/all").iterdir()} == before


def test_provenance_only_scans_nothing(root: Path, tmp_path: Path) -> None:
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(root, copy)
    assert _analyze(copy, "--provenance-only") == 0
    found = latest_artifact(copy, "evaluation/P3-oracle-l2s/provenance", "")
    assert not (found / ENV).exists()


def test_report_labels_in_sample_l2s_and_the_score_ceiling(root: Path, tmp_path: Path) -> None:
    target = tmp_path / "report"
    assert (
        main(["report", "P3-oracle-l2s", "--artifacts", str(root), "--report-dir", str(target)])
        == 0
    )
    text = (target / "README.md").read_text()
    assert "**in-sample**" in text
    assert "It is not a ranking ceiling" in text
    assert "single-positive" in text
    assert (target / "tables" / "paired.csv").read_text().count("\n") > 1
    first = (target / "tables" / "metrics.csv").read_bytes()
    main(["report", "P3-oracle-l2s", "--artifacts", str(root), "--report-dir", str(target)])
    assert (target / "tables" / "metrics.csv").read_bytes() == first
