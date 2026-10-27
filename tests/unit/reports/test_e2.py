"""The E2 report from a synthetic probe-descriptors run (F10 research D0)."""

import json
from pathlib import Path

from hyperdime.contracts.manifests import ArtifactWriter, start_manifest
from hyperdime.reports import e2

ENVS = ("scifact", "nfcorpus", "msmarco")


def _fold(b1: float, ceiling: float) -> dict[str, float]:
    return {"b1": b1, "seed_spearman": ceiling / (3 - 2 * ceiling), "ceiling": ceiling,
            "gap": ceiling - b1}  # fmt: skip


def write_check_run(root: Path, folds: dict[str, dict[str, float]], no_shift: bool) -> Path:
    inputs = {"environments": list(ENVS), "targets_only": True, "seeds": [0, 1, 2]}
    inputs |= {env: {"validation_queries": 100} for env in ENVS}
    check = {"folds": folds, "no_shift": no_shift, "threshold": {"gap": 0.05, "min_folds": 2}}
    manifest = start_manifest("probe-descriptors", {}, seed=0)
    with ArtifactWriter(root, "evaluation/E2/all", manifest) as w:
        for name, payload in (("inputs.json", inputs), ("check.json", check)):
            w.path(name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return w.commit()


def test_a_targets_only_run_gives_the_shift_check_and_says_descriptors_were_not_evaluated(
    tmp_path: Path,
) -> None:
    folds = {"scifact": _fold(0.80, 0.95), "nfcorpus": _fold(0.70, 0.90),
             "msmarco": _fold(0.93, 0.95)}  # fmt: skip
    write_check_run(tmp_path / "artifacts", folds, no_shift=False)
    readme = e2.write(tmp_path / "artifacts", tmp_path / "E2")
    text = readme.read_text()
    rows = (tmp_path / "E2" / "tables" / "check.csv").read_text().splitlines()
    assert rows[0] == "held-out,B1 spearman,seed spearman,ceiling,gap,room (gap >= 0.05)"
    assert rows[1] == "scifact,0.8000,0.8636,0.9500,+0.1500,yes"
    assert rows[3].endswith(",+0.0200,no")
    assert "there is an environment-specific shift for descriptors to explain" in text
    assert "Descriptors were not evaluated in this run" in text
    assert "exploratory" in text and "no full or prefix rows" in text


def test_the_report_states_no_shift_and_regenerates_identically(tmp_path: Path) -> None:
    folds = {env: _fold(0.94, 0.95) for env in ENVS}
    write_check_run(tmp_path / "artifacts", folds, no_shift=True)
    first = e2.write(tmp_path / "artifacts", tmp_path / "E2").read_bytes()
    assert b"no environment-specific shift beyond seed noise" in first
    second = e2.write(tmp_path / "artifacts", tmp_path / "E2").read_bytes()
    assert first == second
