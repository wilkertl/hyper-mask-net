"""The shared-mask follow-up, end to end on a tiny corpus (ADR-0015)."""

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from hyperdime.cli.__main__ import main

REPOSITORY = Path(__file__).resolve().parents[2]
CONFIGS = REPOSITORY / "configs"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PIPELINE = _load(REPOSITORY / "tests/integration/test_p3_pipeline.py", "p3_shared_fixtures")
ENV = PIPELINE.ENV


def _shared(root: Path, *extra: str, device: str = "cpu") -> int:
    return main([
        "analyze-shared-masks", "--config-dir", str(CONFIGS), "--output", str(root),
        f"run.device={device}", "--envs", ENV, *extra,
    ])  # fmt: skip


def _latest(root: Path) -> Path:
    return sorted((root / "evaluation/P3-shared-mask/all").iterdir())[-1]


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("shared")
    PIPELINE.build_inputs(root)
    assert PIPELINE.analyze(root) == 0
    assert _shared(root) == 0
    return _latest(root)


def _summary(run: Path) -> dict[str, Any]:
    return json.loads((run / ENV / "summary.json").read_text())


def test_identities_hold_and_every_k_is_reported(run: Path) -> None:
    summary = _summary(run)
    checks = summary["checks"]
    assert checks["lowest_raw_regret"] > -1e-9
    assert checks["largest_full_k_regret"] < 1e-9
    assert checks["largest_mean_identity_error"] < 1e-9
    assert len(summary["shared"]["mean_norm"]["multi"]["mean"]) == 1024
    assert summary["pair_weighted"]["mean_norm"][-1] == pytest.approx(0.0, abs=1e-9)


def test_run_agrees_with_what_p3_recorded_at_its_checkpoints(run: Path) -> None:
    checks = _summary(run)["checks"]
    assert checks["orderings_differing_from_canonical"] == 0
    assert checks["documents_compared_with_canonical"] > 0
    assert checks["canonical"]["largest_worst_regret_difference"] < 1e-5


def test_transfer_reports_eligible_queries_and_partition_variation(run: Path) -> None:
    transfer = _summary(run)["transfer"]
    cell = transfer["1-1"]
    assert cell["queries"] > 0 and cell["partitions_evaluated"] >= 2 * cell["queries"]
    assert len(cell["repeat_band"]["min"]) == 1024
    assert cell["held_mean_norm"]["queries"][0] == cell["queries"]
    assert cell["privileged_mean_norm"]["n"] == cell["queries"]


def test_report_regenerates_identically_and_states_its_limits(run: Path, tmp_path: Path) -> None:
    root = run.parents[3]
    arguments = ["report", "P3-shared-mask", "--artifacts", str(root), "--report-dir",
                 str(tmp_path)]  # fmt: skip
    assert main(arguments) == 0
    first = (tmp_path / "README.md").read_bytes()
    assert main(arguments) == 0
    assert (tmp_path / "README.md").read_bytes() == first
    text = " ".join(first.decode().split())
    assert "privileged" in text and "not negatives" in text
    assert "pointwise and exploratory" in text
    assert "does not show retrieval damage" in text
    assert "not attributed to relevance itself" in text
    assert "not minimax" in text and "irreducible" in text
    assert "mix construction size with which queries enter" in text
    assert "does not describe the rest of the regret distribution" in text
    curves = (tmp_path / "tables" / f"curves-{ENV}.csv").read_text().splitlines()
    assert len(curves) == 1025
    assert (tmp_path / "tables" / "document-similarity.csv").read_text().count(ENV) == 10


def test_seeded_run_repeats_its_results(run: Path, tmp_path: Path) -> None:
    PIPELINE.build_inputs(tmp_path)
    assert PIPELINE.analyze(tmp_path) == 0
    assert _shared(tmp_path) == 0
    found = np.load(_latest(tmp_path) / ENV / "transfer1-1-held_mean_norm.npy")
    assert np.array_equal(found, np.load(run / ENV / "transfer1-1-held_mean_norm.npy"),
                          equal_nan=True)  # fmt: skip


def test_analysis_opens_no_other_split_and_ranks_nothing() -> None:
    for name in ("src/hyperdime/cli/analyze_shared_masks.py",
                 "src/hyperdime/evaluation/representation_shared.py"):  # fmt: skip
        source = (REPOSITORY / name).read_text()
        assert 'qrels_file("validation")' not in source
        assert 'qrels_file("test")' not in source
        assert "faiss" not in source.lower() and "selector" not in source.lower()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gpu_run_matches_the_cpu_run(run: Path, tmp_path: Path) -> None:
    PIPELINE.build_inputs(tmp_path)
    assert PIPELINE.analyze(tmp_path) == 0
    assert _shared(tmp_path, device="cuda") == 0
    other = _latest(tmp_path)
    assert _summary(other)["cost"]["device"].startswith("cuda")
    for name in ("shared-mean_norm.npy", "transfer1-1-held_mean_norm.npy"):
        found, wanted = np.load(other / ENV / name), np.load(run / ENV / name)
        assert np.allclose(found, wanted, atol=1e-6, equal_nan=True)
