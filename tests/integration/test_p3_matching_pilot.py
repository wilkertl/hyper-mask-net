"""The score-matching coverage pilot, end to end on a tiny corpus (ADR-0014)."""

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hyperdime.cli.analyze_representations import load_training_judgments
from hyperdime.space.inputs import load_unlabeled

REPOSITORY = Path(__file__).resolve().parents[2]


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PIPELINE = _load(REPOSITORY / "tests/integration/test_p3_pipeline.py", "p3_pipeline_fixtures")
SCRIPT = _load(REPOSITORY / "scripts/checks/p3_score_matching.py", "p3_score_matching")
ENV = PIPELINE.ENV


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("matching")
    PIPELINE.build_inputs(root)
    arguments = [str(root), "--envs", ENV, "--device", "cpu", "--repeats", "2"]
    assert SCRIPT.main(arguments) == 0
    return next((root / "evaluation/P3-matching-pilot/all").iterdir())


def test_pilot_records_its_fixed_settings_and_exact_checks(run: Path) -> None:
    summary = json.loads((run / ENV / "summary.json").read_text())
    assert summary["exact_check_disagreements"] == 0
    assert summary["settings"]["caliper"] == 0.02 and summary["settings"]["repeats"] == 2
    assert summary["sample"]["queries"] == sum(r["selected_queries"] for r in summary["strata"])


def test_matched_documents_are_never_judged_and_positives_are_always_positive(run: Path) -> None:
    root = run.parents[3]
    found = np.load(run / ENV / "matched-rows.npz")
    assert found["document_row"].size > 0
    inputs = load_unlabeled(root, ENV)
    labels = load_training_judgments(inputs.dataset_dir, inputs.splits["train"])
    rows = {doc: row for row, doc in enumerate(inputs.doc_ids)}
    ids = json.loads((run / ENV / "selected-queries.json").read_text())
    for query, positive, document in zip(
        found["query"], found["positive_row"], found["document_row"], strict=True
    ):
        own = labels[ids[query]]
        assert document not in {rows[d] for d in own}
        assert positive in {rows[d] for d, g in own.items() if g > 0}


def test_pilot_report_regenerates_and_states_its_limits(run: Path, tmp_path: Path) -> None:
    root = str(run.parents[3])
    assert SCRIPT.main([root, "--report-to", str(tmp_path)]) == 0
    first = (tmp_path / "README.md").read_bytes()
    assert SCRIPT.main([root, "--report-to", str(tmp_path)]) == 0
    assert (tmp_path / "README.md").read_bytes() == first
    assert "unjudged, not negatives" in first.decode()
    assert "not population coverage" in first.decode()
