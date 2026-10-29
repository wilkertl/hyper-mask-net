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


@pytest.fixture(scope="module")
def probe_run(targets_run: Stack) -> Stack:
    """A P0 descriptor per environment (still without any judgment), then the full probe."""
    for env in TOY_ENVS:
        assert run_command(targets_run.configs, targets_run.out, "describe-space",
                           f"data={env}", "--protocol", "P0", "--chunk-size", "5") == 0  # fmt: skip
    assert _probe(targets_run, "--envs", *TOY_ENVS, "--permutations", "5") == 0
    return targets_run


def test_describe_space_writes_a_p0_descriptor_without_any_judgment(probe_run: Stack) -> None:
    directory = latest_artifact(probe_run.out, "descriptors", TOY_ENVS[0])
    record = json.loads((directory / "descriptor-P0.json").read_text())
    assert record["protocol"] == "P0" and record["num_queries"] == 0
    assert record["groups"] == ["A", "E"] and len(record["feature_names"]) == 18
    values = np.load(directory / "descriptor-P0.npy")
    assert values.shape == (1024, 18) and np.isfinite(values).all()
    assert np.load(directory / "covariance.npy").shape == (1024, 1024)
    inputs = json.loads((directory / "inputs.json").read_text())
    assert inputs["protocol"] == "P0" and inputs["quantile_rows"] == inputs["num_documents"]
    assert not list(probe_run.out.glob("datasets/*/*/qrels-*.jsonl"))


def test_describe_space_refuses_embeddings_of_another_dataset(
    probe_run: Stack, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    shutil.copytree(probe_run.out / "datasets", out / "datasets")
    shutil.copytree(probe_run.out / "embeddings" / TOY_ENVS[0], out / "embeddings" / TOY_ENVS[1])
    shutil.copytree(probe_run.out / "embeddings" / TOY_ENVS[0], out / "embeddings" / TOY_ENVS[0])
    code = run_command(probe_run.configs, out, "describe-space", f"data={TOY_ENVS[1]}")
    assert code == 1 and not (out / "descriptors").exists()


def test_the_full_probe_writes_rows_coefficients_and_context(probe_run: Stack) -> None:
    run = latest_artifact(probe_run.out, "evaluation/E2", "all")
    rows = [json.loads(line) for line in (run / "probe.jsonl").read_text().splitlines()]
    primary = [r for r in rows if r["target"] == "s1" and r["seed"] == "mean"]
    assert {r["set"] for r in primary} == {"B1", "B2", "1", "2", "3", "4"}
    assert all(r["n_permutations"] == 5 for r in primary if r["form"] != "—")
    assert {r["heldout"] for r in rows} == set(TOY_ENVS)
    context = json.loads((run / "context.json").read_text())
    assert set(context) == set(TOY_ENVS) and len(context[TOY_ENVS[0]]["s4_prior_weights"]) == 9
    inputs = json.loads((run / "inputs.json").read_text())
    assert {d["protocol"] for d in inputs["descriptors"].values()} == {"P0"}
    assert set(json.loads((run / "coefficients.json").read_text())) == set(TOY_ENVS)


def test_report_e2_and_decide_g3_read_the_full_run(probe_run: Stack, tmp_path: Path) -> None:
    report_dir, adr = tmp_path / "E2", tmp_path / "ADR-0013.md"
    args = ["--artifacts", str(probe_run.out)]
    assert main(["report", "E2", *args, "--report-dir", str(report_dir)]) == 0
    first = (report_dir / "README.md").read_bytes()
    assert b"## G3 criteria" in first and (report_dir / "tables" / "criteria.csv").is_file()
    assert main(["report", "E2", *args, "--report-dir", str(report_dir)]) == 0
    assert (report_dir / "README.md").read_bytes() == first
    assert main(["decide", "G3", *args, "--adr", str(adr)]) == 0
    decision = latest_artifact(probe_run.out, "decisions", "G3")
    assert json.loads((decision / "gate_decision.json").read_text())["gate"] == "G3"
    assert "- **Status:** proposed" in adr.read_text()


def test_the_probe_refuses_a_descriptor_of_other_embeddings(
    probe_run: Stack, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    shutil.copytree(probe_run.out, out)
    for path in out.glob(f"descriptors/{TOY_ENVS[0]}/*/inputs.json"):
        path.write_text(json.dumps({**json.loads(path.read_text()), "embeddings": "other"}))
    stack = Stack(probe_run.configs, out)
    assert _probe(stack, "--envs", *TOY_ENVS, "--permutations", "2") == 1
