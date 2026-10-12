"""Latest-artifact lookup over the F1 artifact layout."""

from pathlib import Path

import pytest

from hyperdime.artifacts import latest_artifact
from hyperdime.contracts.manifests import ArtifactWriter, start_manifest
from hyperdime.contracts.validation import ContractError


def _publish(root: Path) -> Path:
    with ArtifactWriter(root, "datasets/scifact", start_manifest("prepare-data", {}, seed=0)) as w:
        w.path("x.txt").write_text("x")
        return w.commit()


def test_latest_published_artifact_is_returned(tmp_path: Path) -> None:
    first = _publish(tmp_path)
    second = _publish(tmp_path)
    assert first != second
    assert latest_artifact(tmp_path, "datasets", "scifact") == second


def test_directories_without_a_manifest_are_skipped(tmp_path: Path) -> None:
    published = _publish(tmp_path)
    (tmp_path / "datasets" / "scifact" / "99999999T999999Z-00000000").mkdir()
    assert latest_artifact(tmp_path, "datasets", "scifact") == published


def test_missing_artifacts_raise(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="no published datasets artifact for 'nfcorpus'"):
        latest_artifact(tmp_path, "datasets", "nfcorpus")
