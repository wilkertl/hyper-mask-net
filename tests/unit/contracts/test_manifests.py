"""Run manifests and the artifact writer: the M0 gate (spec US2, contracts/python-api.md)."""

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from hyperdime.contracts import manifests
from hyperdime.contracts.config import RunConfig
from hyperdime.contracts.hashing import config_hash, file_sha256
from hyperdime.contracts.manifests import (
    ArtifactWriter,
    RunManifest,
    finish_manifest,
    read_manifest,
    start_manifest,
)
from hyperdime.contracts.schemas import QueryRecord, dump_record, load_record
from hyperdime.contracts.validation import ContractError

REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
CONFIG: dict[str, Any] = {
    "model": {
        "name": "Qwen/Qwen3-Embedding-0.6B",
        "revision": REVISION,
        "embedding_dim": 1024,
        "frozen": True,
        "normalize": True,
        "max_length": 512,
        "dtype": "bfloat16",
    },
    "data": {
        "environment": "scifact",
        "source": "BeIR/scifact",
        "instruction": "Given a claim",
        "validation_fraction": 0.1,
    },
    "run": {"seed": 3, "device": "cpu", "dtype": "float32", "output_dir": "artifacts"},
    "config_version": 1,
}
QUERY = QueryRecord(query_id="q1", text="t", instruction=None, environment="e", split="train")


def _config() -> RunConfig:
    return RunConfig.model_validate(CONFIG)


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _visible(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if not p.name.startswith("."))


def test_start_manifest_fills_every_required_field() -> None:
    manifest = start_manifest("diagnose", _config(), device="cpu")
    assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}", manifest.run_id)
    assert manifest.config_hash == config_hash(_config())
    assert manifest.config == _config().model_dump(mode="json")
    assert manifest.dataset_hash == "none"
    assert manifest.model_revision == REVISION
    assert (manifest.seed, manifest.device, manifest.dtype) == (3, "cpu", "float32")
    assert set(manifest.versions) >= {"python", "torch", "numpy", "hyperdime"}
    assert manifest.started_at.tzinfo is not None and manifest.finished_at is None


def test_manifest_accepts_a_plain_config_mapping_with_an_explicit_seed() -> None:
    manifest = start_manifest("spike", {"env": "scifact"}, seed=0)
    assert manifest.config == {"env": "scifact"}
    assert manifest.model_revision == "none"
    with pytest.raises(ContractError, match="seed"):
        start_manifest("spike", {"env": "scifact"})


@pytest.fixture
def git_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A temporary checkout with one committed file, as the working directory."""
    _git(tmp_path, "init", "-q")
    (tmp_path / "tracked.txt").write_text("a")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_a_clean_checkout_records_its_commit(git_repo: Path) -> None:
    manifest = start_manifest("diagnose", _config())
    assert re.fullmatch(r"[0-9a-f]{40}", manifest.git_commit)
    assert manifest.git_dirty is False


def test_untracked_files_do_not_make_the_tree_dirty(git_repo: Path) -> None:
    (git_repo / "untracked.txt").write_text("b")
    assert start_manifest("diagnose", _config()).git_dirty is False


def test_a_changed_tracked_file_makes_the_tree_dirty(git_repo: Path) -> None:
    (git_repo / "tracked.txt").write_text("changed")
    assert start_manifest("diagnose", _config()).git_dirty is True


def test_git_unavailable_is_recorded_not_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr(manifests.subprocess, "run", missing)
    manifest = start_manifest("diagnose", _config())
    assert manifest.git_commit == "unknown" and manifest.git_dirty is None


def test_finish_manifest_sets_the_end_time_and_files() -> None:
    started = start_manifest("diagnose", _config())
    finished = finish_manifest(started, [("queries.json", "ab" * 32)])
    assert finished.finished_at is not None and finished.finished_at >= finished.started_at
    assert [(a.name, a.sha256) for a in finished.artifacts] == [("queries.json", "ab" * 32)]


def test_manifest_round_trips(tmp_path: Path) -> None:
    finished = finish_manifest(start_manifest("diagnose", _config()), [])
    dump_record(finished, tmp_path / "manifest.json")
    assert load_record(RunManifest, tmp_path / "manifest.json") == finished


def test_writer_refuses_to_run_without_a_manifest(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="manifest"):
        ArtifactWriter(tmp_path, "queries", None)  # type: ignore[arg-type]
    assert not (tmp_path / "queries").exists()


def test_nothing_is_visible_before_commit(tmp_path: Path) -> None:
    writer = ArtifactWriter(tmp_path, "queries", start_manifest("embed", _config()))
    writer.write_record("queries.json", QUERY)
    assert _visible(tmp_path / "queries") == []


def test_writer_publishes_payloads_with_their_hashes(tmp_path: Path) -> None:
    writer = ArtifactWriter(tmp_path, "queries", start_manifest("embed", _config()))
    writer.write_record("queries.json", QUERY)
    writer.path("vectors.bin").write_bytes(b"hi")
    published = writer.commit()
    assert published.parent == tmp_path / "queries"
    assert _visible(published) == ["manifest.json", "queries.json", "vectors.bin"]
    manifest = read_manifest(published)
    hashes = {a.name: a.sha256 for a in manifest.artifacts}
    # sha256(b"hi"), computed by hand once and fixed here
    assert (
        hashes["vectors.bin"] == "8f434346648f6b96df89dda901c5176b10a6d83961dd3c1ac88b59b2dc327aa4"
    )
    assert hashes["queries.json"] == file_sha256(published / "queries.json")
    assert manifest.finished_at is not None
    assert _visible(tmp_path / "queries") == [published.name]


def test_an_exception_leaves_nothing_behind(tmp_path: Path) -> None:
    with (
        pytest.raises(RuntimeError),
        ArtifactWriter(tmp_path, "queries", start_manifest("embed", _config())) as writer,
    ):
        writer.write_record("queries.json", QUERY)
        raise RuntimeError("interrupted")
    assert list((tmp_path / "queries").iterdir()) == []


def test_commit_refuses_a_staged_file_that_was_never_written(tmp_path: Path) -> None:
    writer = ArtifactWriter(tmp_path, "queries", start_manifest("embed", _config()))
    writer.path("never-written.npy")
    with pytest.raises(ContractError, match="never-written.npy"):
        writer.commit()


def test_published_artifacts_are_never_overwritten(tmp_path: Path) -> None:
    manifest = start_manifest("embed", _config())
    with ArtifactWriter(tmp_path, "queries", manifest) as first:
        first.write_record("queries.json", QUERY)
    published = tmp_path / "queries" / manifest.run_id
    before = (published / "manifest.json").read_bytes()
    second = ArtifactWriter(tmp_path, "queries", manifest)
    second.path("other.txt").write_text("x")
    with pytest.raises(ContractError, match="already exists"):
        second.commit()
    assert (published / "manifest.json").read_bytes() == before


def test_read_manifest_requires_a_manifest(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="manifest.json"):
        read_manifest(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"schema_version": 1}))
    with pytest.raises(ContractError, match="RunManifest"):
        read_manifest(tmp_path)
