"""Constitution II: training loads never open test or evaluation-set judgments (FR-006, FR-007)."""

from pathlib import Path

from hyperdime.contracts.manifests import ArtifactWriter, start_manifest
from hyperdime.data.prepare import load_environment, prepare_environment

from conftest import toy_data, toy_eval_sets, toy_source, toy_spec


def _prepared(root: Path) -> Path:
    spec = toy_spec()
    with ArtifactWriter(root, "datasets/toy", start_manifest("prepare-data", {}, seed=0)) as writer:
        prepare_environment(spec, toy_data(spec), writer, toy_source(), toy_eval_sets())
        return writer.commit()


def test_training_load_returns_no_test_or_evaluation_judgments(tmp_path: Path) -> None:
    bundle = load_environment(_prepared(tmp_path))
    assert set(bundle.qrels) == {"train", "validation"}
    assert bundle.query_ids("test") and bundle.eval_queries["evalset"]  # texts, not labels


def test_training_load_never_opens_label_files(tmp_path: Path) -> None:
    directory = _prepared(tmp_path)
    before = load_environment(directory)
    (directory / "qrels-test.jsonl").unlink()
    (directory / "qrels-evalset.jsonl").unlink()
    after = load_environment(directory)
    assert after.qrels == before.qrels and after.queries == before.queries


def test_evaluation_asks_for_test_judgments_explicitly(tmp_path: Path) -> None:
    bundle = load_environment(_prepared(tmp_path), include_test_qrels=True)
    assert set(bundle.qrels) == {"train", "validation", "test", "evalset"}
    assert bundle.positives("evalset") == {"e0": {"d1"}, "e1": {"d4"}}
