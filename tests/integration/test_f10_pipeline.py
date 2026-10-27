"""F10 commands end to end on the F5 toy stack (S1 and S4 seeded selectors per environment):
targets and the shift check, the space descriptor, the probes, the E2 report, and decide G3."""

import json
import shutil
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
from test_f4_pipeline import TOY_ENVS, f4_stack, run_command  # noqa: F401 - fixtures
from test_f5_pipeline import (  # noqa: F401 - fixtures
    Comparison,
    S4Stack,
    comparison,
    f5_stack,
    s2_comparison,
    s4_stack,
)

from hyperdime.artifacts import latest_artifact
from hyperdime.cli.__main__ import main


class Stack(NamedTuple):
    configs: Path
    out: Path


def _probe(stack: Stack, *extra: str) -> int:
    return run_command(stack.configs, stack.out, "probe-descriptors", "--experiment", "E2",
                       *extra)  # fmt: skip


@pytest.fixture(scope="module")
def targets_run(
    s4_stack: S4Stack,  # noqa: F811
    comparison: Comparison,  # noqa: F811
    tmp_path_factory: pytest.TempPathFactory,
) -> Stack:
    """A copy of the S4 stack with every judgments file deleted, then a targets-only run."""
    stack = Stack(comparison.configs, tmp_path_factory.mktemp("f10") / "out")
    shutil.copytree(s4_stack.out, stack.out)
    for path in stack.out.glob("datasets/*/*/qrels-*.jsonl"):
        path.unlink()  # nothing in E2 may read a judgment
    assert _probe(stack, "--envs", *TOY_ENVS, "--targets-only") == 0
    return stack


def test_targets_only_writes_targets_and_the_shift_check_without_any_judgment(
    targets_run: Stack,
) -> None:
    run = latest_artifact(targets_run.out, "evaluation/E2", "all")
    inputs = json.loads((run / "inputs.json").read_text())
    assert inputs["targets_only"] is True and inputs["environments"] == list(TOY_ENVS)
    for env in TOY_ENVS:
        index = json.loads((run / f"targets-{env}.json").read_text())
        values = np.load(run / f"targets-{env}.npy")
        assert values.shape == (len(index["keys"]), 1024)
        assert {"s1/mean", "s1/bias", "s4_correction/mean", "variance/mean"} <= set(index["keys"])
        primary = values[index["keys"].index("s1/mean")]
        assert primary.sum() == pytest.approx(1.0)  # each budget's profile sums to 1
        assert sorted(index["s1"]["seeds"]) == ["0", "1", "2"]
        assert index["validation_queries"] >= 1
    check = json.loads((run / "check.json").read_text())
    assert set(check["folds"]) == set(TOY_ENVS) and isinstance(check["no_shift"], bool)
    assert not (run / "probe.jsonl").exists()


def test_report_e2_reads_a_targets_only_run(targets_run: Stack, tmp_path: Path) -> None:
    report_dir = tmp_path / "E2"
    assert (
        main(["report", "E2", "--artifacts", str(targets_run.out), "--report-dir", str(report_dir)])
        == 0
    )
    assert (tmp_path / "E2" / "tables" / "check.csv").read_text().count("\n") == 4
    assert "not evaluated" in (tmp_path / "E2" / "README.md").read_text()


def test_probe_descriptors_refuses_two_environments(targets_run: Stack) -> None:
    assert _probe(targets_run, "--envs", *TOY_ENVS[:2], "--targets-only") == 1


def test_probe_descriptors_refuses_a_missing_seed(targets_run: Stack) -> None:
    assert _probe(targets_run, "--envs", *TOY_ENVS, "--targets-only", "--seeds", "0", "1", "7") == 1
