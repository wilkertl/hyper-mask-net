"""``hyperdime diagnose``: config in, manifest out, nothing loaded or trained (contracts/cli.md)."""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hyperdime.cli.__main__ import main

FIXTURES = Path(__file__).parents[1] / "fixtures" / "configs"
REQUIRED = (
    "config_hash",
    "git_commit",
    "git_dirty",
    "dataset_hash",
    "model_revision",
    "seed",
    "device",
    "dtype",
    "started_at",
    "finished_at",
)


def _diagnose(output: Path, *extra: str, config_dir: Path = FIXTURES / "valid") -> int:
    return main(["diagnose", "--config-dir", str(config_dir), "--output", str(output), *extra])


def test_diagnose_writes_one_complete_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    start = time.perf_counter()
    assert _diagnose(tmp_path) == 0
    elapsed = time.perf_counter() - start
    stdout = capsys.readouterr().out.splitlines()
    assert len(stdout) == 1
    directory = Path(stdout[0])
    assert directory.parent == tmp_path / "diagnose"
    assert sorted(p.name for p in directory.iterdir()) == ["manifest.json"]
    manifest = json.loads((directory / "manifest.json").read_text())
    assert all(manifest[field] is not None for field in REQUIRED if field != "git_dirty")
    assert manifest["dataset_hash"] == "none"
    assert manifest["model_revision"] == "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
    assert manifest["command"] == "diagnose" and manifest["artifacts"] == []
    assert elapsed < 5.0  # SC-005


def test_two_runs_share_the_config_hash_and_nothing_else(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _diagnose(tmp_path)
    _diagnose(tmp_path)
    directories = capsys.readouterr().out.splitlines()
    first, second = (json.loads((Path(d) / "manifest.json").read_text()) for d in directories)
    assert first["config_hash"] == second["config_hash"]
    assert first["run_id"] != second["run_id"]


@pytest.mark.parametrize(
    ("name", "field"),
    [
        ("missing_revision", "model.revision"),
        ("unknown_key", "model.unknown_key"),
        ("wrong_dim", "model.embedding_dim"),
    ],
)
def test_invalid_configs_exit_2_and_write_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], name: str, field: str
) -> None:
    code = _diagnose(
        tmp_path, "--config-name", f"experiment/{name}", config_dir=FIXTURES / "invalid"
    )
    assert code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"error: invalid config: {field}: " in captured.err
    assert not (tmp_path / "diagnose").exists()


def test_the_cli_imports_no_heavy_libraries() -> None:
    code = (
        "import sys, hyperdime.cli.__main__, hyperdime.cli.diagnose; "
        "print([m for m in ('transformers', 'datasets', 'faiss') if m in sys.modules])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"
