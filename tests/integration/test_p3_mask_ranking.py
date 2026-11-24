"""The ranking diagnostic, end to end on a tiny corpus (ADR-0016)."""

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hyperdime.cli.__main__ import main
from hyperdime.evaluation.representation_ranking import ROWS

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "scripts/checks"))  # sibling import in the checks
CONFIGS = REPOSITORY / "configs"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up while being defined
    spec.loader.exec_module(module)
    return module


PIPELINE = _load(REPOSITORY / "tests/integration/test_p3_pipeline.py", "p3_ranking_fixtures")
ENV = PIPELINE.ENV


def _shared(root: Path) -> int:
    return main([
        "analyze-shared-masks", "--config-dir", str(CONFIGS), "--output", str(root),
        "run.device=cpu", "--envs", ENV,
    ])  # fmt: skip


def _rankings(root: Path, *extra: str, device: str = "cpu") -> int:
    return main([
        "analyze-mask-rankings", "--config-dir", str(CONFIGS), "--output", str(root),
        f"run.device={device}", *extra, "--envs", ENV,
    ])  # fmt: skip


def _latest(root: Path) -> Path:
    return sorted((root / "evaluation/P3-mask-ranking/all").iterdir())[-1]


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("rankings")
    PIPELINE.build_inputs(root)
    assert PIPELINE.analyze(root) == 0
    assert _shared(root) == 0
    assert _rankings(root, "--budgets", "4", "256", "921") == 0
    return _latest(root)


def _summary(run: Path) -> dict[str, Any]:
    return json.loads((run / ENV / "summary.json").read_text())


def test_masks_are_those_the_shared_run_scored(run: Path) -> None:
    checks = _summary(run)["checks"]
    assert checks["stored_regrets_compared"] > 0
    assert checks["largest_stored_regret_difference"] < 1e-6


def test_inputs_are_recorded_with_the_shared_run_they_were_checked_against(run: Path) -> None:
    inputs = json.loads((run / "inputs.json").read_text())
    assert inputs["budgets"] == [4, 256, 921]
    assert inputs[ENV]["shared_mask_run"]
    assert inputs[ENV]["qrels_train_sha256"]


def test_every_row_is_ranked_and_the_full_row_appears_once_per_query(run: Path) -> None:
    summary = _summary(run)
    per_row = summary["coverage"]["masks_per_row"]
    queries = summary["coverage"]["queries"]
    assert per_row["full"] == queries
    assert per_row["prefix"] == queries * 3
    assert per_row["shared"] == queries * 3
    assert per_row["own"] == summary["coverage"]["positives"] * 3


def test_full_row_compared_with_itself_never_moves_a_rank(run: Path) -> None:
    with np.load(run / ENV / "per-query.npz") as found:
        improved = found["improved"]
        worsened = found["worsened"]
        rr = found["rr"]
        base = found["base_rr"]
    row = ROWS.index("full")
    assert np.nansum(improved[row, -1]) == 0
    assert np.nansum(worsened[row, -1]) == 0
    assert np.array_equal(rr[row, -1], base[row, -1])


def test_a_positive_under_its_own_mask_is_measured_against_its_full_query_rank(run: Path) -> None:
    with np.load(run / ENV / "per-query.npz") as found:
        evaluated = found["evaluated"][ROWS.index("own")]
    assert np.nanmin(evaluated) == 1
    assert np.nanmax(evaluated) == 1


def test_runs_are_deterministic_and_never_open_other_judgments(
    run: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    # The fixture's validation and test judgments are not JSON: reading them would fail.
    root = tmp_path_factory.mktemp("again")
    PIPELINE.build_inputs(root)
    assert _rankings(root, "--budgets", "4", "256", "921", "--no-own-masks") == 0
    again = _latest(root)
    with np.load(run / ENV / "tracked.npz") as a, np.load(again / ENV / "tracked.npz") as b:
        with np.load(run / ENV / "masks.npz") as ma, np.load(again / ENV / "masks.npz") as mb:
            keep = ma["row"] != ROWS.index("own")
            assert ma["ndcg_at_10"][keep].tolist() == mb["ndcg_at_10"].tolist()
        kept = np.repeat(keep, np.diff(a["offsets"]))
        assert a["rank"][kept].tolist() == b["rank"].tolist()


INTEGRITY = _load(REPOSITORY / "scripts/checks/p3_ranking_integrity.py", "p3_ranking_integrity")
EXCLUSION = _load(
    REPOSITORY / "scripts/checks/p3_ranking_exclusion_sensitivity.py",
    "p3_ranking_exclusion_sensitivity",
)


def test_integrity_check_passes_a_stored_run_and_matches_its_report(
    run: Path, tmp_path: Path
) -> None:
    root = run.parents[3]
    report = tmp_path / "report"
    main(["report", "P3-mask-ranking", "--artifacts", str(root), "--report-dir", str(report)])
    found = INTEGRITY.check_run(root, run, 4, report)
    assert found["failed"] == []
    assert found["paired"]["primary"]
    assert found["environments"][ENV]["spot"]["masks_rescored"] > 0


def test_integrity_check_reports_a_rank_that_was_altered(run: Path, tmp_path: Path) -> None:
    root = run.parents[3]
    copy = tmp_path / run.name
    shutil.copytree(run, copy)
    with np.load(copy / ENV / "tracked.npz") as found:
        arrays = {name: found[name] for name in found.files}
    arrays["rank"][arrays["rank"] > 0] += 2**32
    np.savez(copy / ENV / "tracked.npz", **arrays)  # type: ignore[arg-type]
    failed = INTEGRITY.check_run(root, copy, 4, None)["failed"]
    assert f"{ENV}: ranks are within the corpus" in failed
    assert f"{ENV}: spot: full-corpus ranks agree exactly" in failed


def test_a_passage_outside_the_competition_is_charged_only_where_it_beats_the_positive() -> None:
    # Positive scores 0.5 (row 7), 0.5 (row 2), 0.1 (row 7), and one positive already excluded.
    rank = np.array([4, 4, 9, 0])
    score = np.array([0.5, 0.5, 0.1, 0.5])
    row = np.array([7, 2, 7, 3])
    corrected = EXCLUSION.corrected_ranks(rank, score, row, np.full(4, 0.5), np.full(4, 3))
    # Tied with a lower row: charged. Tied with a higher row: not. Beaten outright: charged.
    assert corrected.tolist() == [5, 4, 10, 0]


def test_exclusion_analysis_of_a_run_without_collisions_changes_no_rank(run: Path) -> None:
    found = EXCLUSION.check_run(run.parents[3], run)
    assert found["environments"][ENV]["queries_with_excluded_document"] == 0
    assert found["exposure"] == []
    assert all(row["stored_difference"] == row["corrected_difference"] for row in found["primary"])
    assert not any(row["holm_0_05_changed"] for row in found["primary"])


def test_report_is_generated_and_labels_the_privileged_diagnostic(
    run: Path, tmp_path: Path
) -> None:
    root = run.parents[3]
    target = tmp_path / "report"
    assert (
        main(["report", "P3-mask-ranking", "--artifacts", str(root), "--report-dir", str(target)])
        == 0
    )
    text = (target / "README.md").read_text()
    assert "Privileged offline diagnostic" in text
    assert "not confirmed irrelevant" in text
    assert "neither minimax nor ranking optimal" in text
    assert (target / "tables" / "claims.csv").read_text().count("\n") > 1
    first = (target / "tables" / "list-metrics.csv").read_bytes()
    assert (
        main(["report", "P3-mask-ranking", "--artifacts", str(root), "--report-dir", str(target)])
        == 0
    )
    assert (target / "tables" / "list-metrics.csv").read_bytes() == first
