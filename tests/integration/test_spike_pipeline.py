"""Synthetic end-to-end spike run: cached embeddings → train → global → analyze → report."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from hyperdime.spike import diagnostics, pipeline
from hyperdime.spike.manifest import read_json, write_json, write_manifest
from hyperdime.spike.report import write_report


@pytest.fixture(autouse=True)
def _small_selection_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    """The full temperature x epoch grid is exercised with two values each to keep CPU time low."""
    monkeypatch.setattr(pipeline, "TEMPERATURES", (0.05, 0.1))
    monkeypatch.setattr(pipeline, "EPOCH_BUDGETS", (2, 5))
    monkeypatch.setattr(diagnostics, "EPOCHS_ABOVE", (6,))
    monkeypatch.setattr(diagnostics, "EPOCHS_BELOW", (1,))


def _write_env(root: Path, env: str, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    dim, num_docs = pipeline.DIM, 40
    docs = F.normalize(torch.randn(num_docs, dim, generator=generator), dim=-1)
    query_ids = [f"{env}-q{i}" for i in range(24)]
    targets = [i % num_docs for i in range(24)]
    queries = F.normalize(docs[targets] + 0.5 * torch.randn(24, dim, generator=generator), dim=-1)
    doc_ids = [f"{env}-d{i}" for i in range(num_docs)]
    for name, ids, values in (("corpus", doc_ids, docs), ("queries", query_ids, queries)):
        (root / env / name).mkdir(parents=True)
        np.save(root / env / name / "embeddings.npy", values.numpy().astype(np.float16))
        write_json(root / env / name / "ids.json", ids)
    labels = {qid: {doc_ids[t]: 1} for qid, t in zip(query_ids, targets, strict=True)}
    splits = {
        "train": query_ids[:12],
        "validation": query_ids[12:16],
        "evaluation": query_ids[16:],
        "train_labels": {q: labels[q] for q in query_ids[:16]},
        "eval_labels": {q: labels[q] for q in query_ids[16:]},
    }
    write_json(root / env / "splits.json", splits)
    labelled = query_ids[:16]
    negatives = pipeline._mine(queries[:16], docs, doc_ids, labelled, splits["train_labels"])
    write_json(root / env / "negatives.json", negatives)
    counts = {"train": 12, "validation": 4, "evaluation": 8}
    write_manifest(root / env, "prepare", {"env": env, "counts": counts})


def test_spike_pipeline_runs_end_to_end_on_cpu(tmp_path: Path) -> None:
    envs, seeds = ["alpha", "beta"], [0, 1]
    for index, env in enumerate(envs):
        _write_env(tmp_path, env, seed=index)
    for env in envs:
        pipeline.train(env, tmp_path, seeds)
    pipeline.train_global(envs, tmp_path, seeds)
    analysis = pipeline.analyze(envs, tmp_path, seeds)

    criteria = analysis["criteria"]
    assert set(criteria["transfer"]) == {"alpha->beta", "beta->alpha"}
    assert criteria["verdict"] in {"supported", "not supported"}
    full = analysis["summary"]["alpha"]["full"]["1024"]["ndcg@10"]
    oracle = analysis["summary"]["alpha"]["oracle"][str(pipeline.HEADLINE_K)]["ndcg@10"]
    assert 0.0 <= full <= 1.0 and 0.0 <= oracle <= 1.0
    assert 0.0 <= analysis["mask"]["overlap_cross_env"] <= 1.0

    report = write_report(tmp_path, tmp_path / "report")
    text = report.read_text()
    assert f"## Verdict: H1 {criteria['verdict']}" in text
    assert "| alpha | 8 |" in text
    assert write_report(tmp_path, tmp_path / "report").read_text() == text


def _selector_weights(
    root: Path, envs: list[str], seeds: list[int]
) -> list[dict[str, torch.Tensor]]:
    paths = [root / env / "selectors" / f"seed-{s}.pt" for env in envs for s in seeds]
    paths += [root / "global" / f"seed-{s}.pt" for s in seeds]
    return [torch.load(path, weights_only=True) for path in paths]


def test_training_never_reads_evaluation_labels(tmp_path: Path) -> None:
    """US0/AC4, Constitution II: rewriting every evaluation label leaves all selectors unchanged."""
    envs, seeds = ["alpha", "beta"], [0]
    for variant in ("original", "corrupted"):
        for index, env in enumerate(envs):
            _write_env(tmp_path / variant, env, seed=index)
    for env in envs:
        path = tmp_path / "corrupted" / env / "splits.json"
        splits = read_json(path)
        # Point each evaluation query at a different document than its true positive.
        splits["eval_labels"] = {
            qid: {f"{env}-d{(int(qid.split('q')[-1]) + 7) % 40}": 1} for qid in splits["evaluation"]
        }
        write_json(path, splits)
    for variant in ("original", "corrupted"):
        root = tmp_path / variant
        for env in envs:
            pipeline.train(env, root, seeds)
        pipeline.train_global(envs, root, seeds)
    original = _selector_weights(tmp_path / "original", envs, seeds)
    corrupted = _selector_weights(tmp_path / "corrupted", envs, seeds)
    for a, b in zip(original, corrupted, strict=True):
        assert all(torch.equal(a[name], b[name]) for name in a)
    for env in envs:
        assert read_json(
            tmp_path / "original" / env / "selectors" / "temperature.json"
        ) == read_json(tmp_path / "corrupted" / env / "selectors" / "temperature.json")


def test_steps_refuse_inputs_from_interrupted_runs(tmp_path: Path) -> None:
    """Constitution III: an unfinished step leaves a "running" manifest; its outputs are refused."""
    envs, seeds = ["alpha", "beta"], [0]
    for index, env in enumerate(envs):
        _write_env(tmp_path, env, seed=index)
    for env in envs:
        pipeline.train(env, tmp_path, seeds)
    pipeline.train_global(envs, tmp_path, seeds)
    for env in envs:
        assert read_json(tmp_path / env / "selectors" / "manifest.json")["status"] == "complete"

    write_manifest(tmp_path / "beta" / "selectors", "train", {"env": "beta"}, status="running")
    with pytest.raises(ValueError, match="was not completed"):
        pipeline.analyze(envs, tmp_path, seeds)
    with pytest.raises(ValueError, match="was not completed"):
        pipeline.train_global(envs, tmp_path, seeds)
    write_manifest(tmp_path / "alpha", "prepare", {"env": "alpha"}, status="running")
    with pytest.raises(ValueError, match="was not completed"):
        pipeline.train("alpha", tmp_path, seeds)
    # analyze was refused before writing anything, so there is no analysis to report on.
    with pytest.raises(ValueError, match="has no manifest"):
        write_report(tmp_path, tmp_path / "report")


def _corrupt_evaluation_labels(root: Path, envs: list[str]) -> None:
    """Point each evaluation query at a different document than its true positive."""
    for env in envs:
        path = root / env / "splits.json"
        splits = read_json(path)
        splits["eval_labels"] = {
            qid: {f"{env}-d{(int(qid.split('q')[-1]) + 7) % 40}": 1} for qid in splits["evaluation"]
        }
        write_json(path, splits)


def _run_through_analyze(root: Path, envs: list[str], seeds: list[int]) -> None:
    for env in envs:
        pipeline.train(env, root, seeds)
    pipeline.train_global(envs, root, seeds)
    pipeline.analyze(envs, root, seeds)


DIAGNOSE_ENVS, DIAGNOSE_SEEDS = ["alpha", "beta"], [0, 1]
DIAGNOSE_OPTIONS = {"permutations": 30, "half_splits": 8}


@pytest.fixture(scope="module")
def diagnosed(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One synthetic run through `diagnose`, plus a copy whose evaluation labels are corrupted."""
    root = tmp_path_factory.mktemp("diagnose")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pipeline, "TEMPERATURES", (0.05, 0.1))
        patch.setattr(pipeline, "EPOCH_BUDGETS", (2, 5))
        patch.setattr(diagnostics, "EPOCHS_ABOVE", (6,))
        patch.setattr(diagnostics, "EPOCHS_BELOW", (1,))
        for variant in ("original", "corrupted"):
            for index, env in enumerate(DIAGNOSE_ENVS):
                _write_env(root / variant, env, seed=index)
        _corrupt_evaluation_labels(root / "corrupted", DIAGNOSE_ENVS)
        results = {}
        for variant in ("original", "corrupted"):
            _run_through_analyze(root / variant, DIAGNOSE_ENVS, DIAGNOSE_SEEDS)
            if variant == "original":
                before = write_report(root / variant, root / "report-before").read_text()
            results[variant] = diagnostics.diagnose(
                DIAGNOSE_ENVS, root / variant, DIAGNOSE_SEEDS, **DIAGNOSE_OPTIONS
            )
    return {"root": root, "results": results, "report_before": before}


def test_diagnose_writes_a_complete_result(diagnosed: dict[str, Any]) -> None:
    """FR-000d, research R20: the result has every section and a complete manifest."""
    result = diagnosed["results"]["original"]
    root = diagnosed["root"] / "original"
    assert read_json(root / "diagnostics" / "manifest.json")["status"] == "complete"
    profile = result["oracle_shift"]["profiles"]["alpha"]
    assert profile["labels"] == "train" and profile["num_queries"] == 16
    assert sum(profile["frequency"]) == pytest.approx(pipeline.HEADLINE_K)
    assert set(result["oracle_shift"]["pairs"]) == {"alpha|beta"}
    assert set(result["trainer"]["envs"]) == set(DIAGNOSE_ENVS)
    assert result["recommendation"] in {"stop_or_pivot", "continue_strengthen_selector", "continue"}


def test_diagnose_ignores_evaluation_labels_in_oracle_profiles_and_probe(
    diagnosed: dict[str, Any],
) -> None:
    """Constitution II: D1 and D3 read training labels only; D2 scores with evaluation labels."""
    original, corrupted = diagnosed["results"]["original"], diagnosed["results"]["corrupted"]
    assert corrupted["oracle_shift"] == original["oracle_shift"]
    assert corrupted["probe"] == original["probe"]


def test_report_gains_the_diagnostics_section_only_after_diagnose(
    diagnosed: dict[str, Any],
) -> None:
    root = diagnosed["root"] / "original"
    before = diagnosed["report_before"]
    after = write_report(root, diagnosed["root"] / "report-after").read_text()
    assert "Diagnostics (research R20)" not in before
    assert after.startswith(before.rstrip("\n"))
    assert diagnosed["results"]["original"]["recommendation"] in after


def test_diagnose_is_deterministic(tmp_path: Path) -> None:
    for index, env in enumerate(DIAGNOSE_ENVS):
        _write_env(tmp_path, env, seed=index)
    _run_through_analyze(tmp_path, DIAGNOSE_ENVS, DIAGNOSE_SEEDS)
    result = tmp_path / "diagnostics" / "diagnostics.json"
    diagnostics.diagnose(DIAGNOSE_ENVS, tmp_path, DIAGNOSE_SEEDS, **DIAGNOSE_OPTIONS)
    first = result.read_bytes()
    diagnostics.diagnose(DIAGNOSE_ENVS, tmp_path, DIAGNOSE_SEEDS, **DIAGNOSE_OPTIONS)
    assert result.read_bytes() == first


def test_diagnose_refuses_an_unfinished_analysis(tmp_path: Path) -> None:
    for index, env in enumerate(DIAGNOSE_ENVS):
        _write_env(tmp_path, env, seed=index)
    _run_through_analyze(tmp_path, DIAGNOSE_ENVS, DIAGNOSE_SEEDS)
    write_manifest(tmp_path / "analysis", "analyze", {"envs": DIAGNOSE_ENVS}, status="running")
    with pytest.raises(ValueError, match="was not completed"):
        diagnostics.diagnose(DIAGNOSE_ENVS, tmp_path, DIAGNOSE_SEEDS, **DIAGNOSE_OPTIONS)


def test_report_skips_an_unfinished_diagnose(tmp_path: Path) -> None:
    for index, env in enumerate(DIAGNOSE_ENVS):
        _write_env(tmp_path, env, seed=index)
    _run_through_analyze(tmp_path, DIAGNOSE_ENVS, DIAGNOSE_SEEDS)
    write_manifest(tmp_path / "diagnostics", "diagnose", {"envs": DIAGNOSE_ENVS}, status="running")
    assert "Diagnostics" not in write_report(tmp_path, tmp_path / "report").read_text()
