"""P3 end to end on a synthetic environment whose non-training judgments cannot be read."""

import hashlib
import importlib.util
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pytest
import torch

from hyperdime.artifacts import latest_artifact
from hyperdime.cli.__main__ import main
from hyperdime.contracts.hashing import config_hash, file_sha256
from hyperdime.contracts.manifests import ArtifactWriter, read_manifest, start_manifest
from hyperdime.contracts.schemas import (
    DatasetManifest,
    EmbeddingArtifact,
    PayloadRef,
    QrelsRecord,
    QueryRecord,
    dump_records,
)

REPOSITORY = Path(__file__).resolve().parents[2]
CONFIGS = REPOSITORY / "configs"
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
ENV = "scifact"
SECRET = "private-query-text"
NOT_JSON = "this judgments file must never be opened\n"
TRAIN = 24
DOCUMENTS = 40
BUDGETS = ("--budgets", "4", "256", "921")


def _unit(rng: np.random.Generator, rows: int) -> np.ndarray:
    vectors = rng.normal(size=(rows, 1024))
    return (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(np.float16)


def _judgments() -> list[QrelsRecord]:
    """Query i has 1 + i % 4 positives, graded 1 or 2; every fourth query has a judged zero."""
    records = []
    for i in range(TRAIN):
        for j in range(1 + i % 4):
            records.append((i, (i + j) % DOCUMENTS, 1 + j % 2))
        if i % 4 == 0:
            records.append((i, (i + 20) % DOCUMENTS, 0))
    return [
        QrelsRecord(query_id=f"q{i}", doc_id=f"d{d}", relevance=g, environment=ENV, split="train")
        for i, d, g in records
    ]


def build_inputs(
    root: Path, extra_train: list[QrelsRecord] | None = None, env: str = ENV
) -> tuple[Path, Path]:
    rng = np.random.default_rng(0)
    query_ids = [f"q{i}" for i in range(TRAIN + 8)]
    doc_ids = [f"d{i}" for i in range(DOCUMENTS)]
    splits = {"train": query_ids[:TRAIN], "validation": query_ids[TRAIN:TRAIN + 4],
              "test": query_ids[TRAIN + 4:]}  # fmt: skip
    with ArtifactWriter(root, f"datasets/{env}", start_manifest("t", {"e": env}, seed=0)) as w:
        dump_records(
            [QueryRecord(query_id=q, text=SECRET, instruction="x", environment=env, split=s)
             for s, ids in splits.items() for q in ids],
            w.path("queries.jsonl"),
        )  # fmt: skip
        dump_records([*_judgments(), *(extra_train or [])], w.path("qrels-train.jsonl"))
        w.path("qrels-validation.jsonl").write_text(NOT_JSON)
        w.path("qrels-test.jsonl").write_text(NOT_JSON)
        files = {n: file_sha256(w.path(n)) for n in ("queries.jsonl", "qrels-train.jsonl")}
        fingerprint = config_hash(files)
        w.write_record(
            "dataset.json",
            DatasetManifest(
                environment=env, spec={}, num_documents=DOCUMENTS, dropped_documents=0,
                num_queries={k: len(v) for k, v in splits.items()}, num_qrels={}, dropped_qrels=0,
                excluded_queries={}, split_seed=0, validation_fraction=0.1, files=files,
                fingerprint=fingerprint,
            ),
        )  # fmt: skip
        dataset = w.commit()
    manifest = start_manifest("t", {"e": env}, seed=0, dataset_hash=fingerprint)
    with ArtifactWriter(root, f"embeddings/{env}", manifest) as w:
        for name, ids in (("documents", doc_ids), ("queries", query_ids)):
            target = w.path(f"{name}.npy")
            np.save(target, _unit(rng, len(ids)))
            w.write_record(
                f"{name}.json",
                EmbeddingArtifact(
                    role=name, environment=env, ids=ids, num_rows=len(ids), dtype="float16",
                    normalized=True, model_revision=REVISION,
                    instruction="x" if name == "queries" else None,
                    payload=PayloadRef(
                        path=f"{name}.npy", sha256=file_sha256(target), format="npy"
                    ),
                ),
            )  # fmt: skip
        embeddings = w.commit()
    return dataset, embeddings


def analyze(root: Path, *extra: str, device: str = "cpu") -> int:
    return main([
        "analyze-representations", "--config-dir", str(CONFIGS), "--output", str(root),
        "--envs", ENV, *BUDGETS, "--stability-draws", "4", "--across-pairs", "50",
        "--reference-samples", "3", *extra, f"run.device={device}",
    ])  # fmt: skip


def _script(name: str) -> Any:
    path = REPOSITORY / f"scripts/checks/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Run(NamedTuple):
    root: Path
    directory: Path
    embeddings: Path
    summary: dict[str, Any]


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Run:
    root = tmp_path_factory.mktemp("p3")
    _, embeddings = build_inputs(root)
    assert analyze(root) == 0
    directory = latest_artifact(root, "evaluation/P3", "all")
    summary = json.loads((directory / ENV / "summary.json").read_text())
    summary["curves"] = json.loads((directory / ENV / "curves.json").read_text())
    return Run(root, directory, embeddings, summary)


def test_run_reads_no_validation_or_test_judgment(run: Run) -> None:
    # The two files hold no JSON, so any attempt to parse one would have failed the run.
    assert run.summary["coverage"]["selected_queries"] == TRAIN


def test_a_judgment_of_another_split_among_the_training_labels_is_refused(tmp_path: Path) -> None:
    leak = QrelsRecord(query_id=f"q{TRAIN}", doc_id="d0", relevance=1, environment=ENV,
                       split="train")  # fmt: skip
    build_inputs(tmp_path, [leak])
    assert analyze(tmp_path) == 1
    assert not (tmp_path / "evaluation" / "P3" / "all").exists() or not list(
        p for p in (tmp_path / "evaluation" / "P3" / "all").iterdir() if not p.name.startswith(".")
    )


def test_every_relevant_document_stays_an_individual_pair(run: Run) -> None:
    pairs = [
        json.loads(line) for line in (run.directory / ENV / "pairs.jsonl").read_text().splitlines()
    ]
    expected = {(r.query_id, r.doc_id, r.relevance) for r in _judgments() if r.relevance > 0}
    assert {(p["query_id"], p["doc_id"], p["grade"]) for p in pairs} == expected
    contributions = np.load(run.directory / ENV / "pair-contributions.npy")
    order = np.load(run.directory / ENV / "pair-order.npy")
    assert contributions.shape == order.shape == (len(expected), 1024)
    assert (np.sort(order, axis=1) == np.arange(1024)).all()
    values = np.take_along_axis(contributions, order.astype(np.int64), axis=1)
    assert (np.diff(values, axis=1) <= 0).all()


def test_identities_hold_on_every_pair_and_query(run: Run) -> None:
    assert run.summary["identities"]["max_decomposition_error"] < 1e-14
    assert run.summary["identities"]["max_aggregation_identity_error"] < 1e-14


def test_query_balanced_frequency_rows_sum_to_their_budget(run: Run) -> None:
    frequency = np.load(run.directory / ENV / "frequency.npy")
    assert frequency.sum(axis=1) == pytest.approx([4, 256, 921])


def test_single_and_multiple_positive_queries_are_reported_apart(run: Run) -> None:
    coverage = run.summary["coverage"]
    assert coverage["single_positive_queries"] == 6
    assert coverage["multi_positive_queries"] == 18
    scalar = run.summary["pairs"]["scalar"]["score"]
    assert scalar["single"]["n"] == 6 and scalar["multi"]["n"] == 18 and scalar["all"]["n"] == 24
    assert run.summary["agreement"]["mean_jaccard"]["multi"][0]["n"] == 18
    assert run.summary["agreement"]["mean_jaccard"]["single"][0]["n"] == 0


def test_judged_nonrelevant_documents_are_analyzed_apart_and_unjudged_ones_not_at_all(
    run: Run,
) -> None:
    nonrelevant = run.summary["nonrelevant"]
    assert nonrelevant["queries_with_judged_nonrelevant"] == 6
    assert nonrelevant["pairs"] == 6
    assert nonrelevant["coverage"] == 0.25
    stored = np.load(run.directory / ENV / "nonrelevant-stats.npz")
    assert stored["score"].size == 6  # only judged rows, not the 30-odd unjudged per query
    grades = run.summary["coverage"]["grades"]
    assert "0" not in grades  # no grade-0 document joined the relevant pairs


def test_grades_are_kept_in_the_grade_table(run: Run) -> None:
    assert [row["grade"] for row in run.summary["grades"]] == [1, 2]
    assert sum(row["pairs"] for row in run.summary["grades"]) == 60


def test_stored_embeddings_are_not_modified(run: Run) -> None:
    for name in ("documents", "queries"):
        record = json.loads((run.embeddings / f"{name}.json").read_text())
        assert file_sha256(run.embeddings / f"{name}.npy") == record["payload"]["sha256"]


def test_manifest_records_inputs_and_lists_every_file(run: Run) -> None:
    manifest = read_manifest(run.directory)
    assert manifest.command == "analyze-representations"
    assert manifest.model_revision == REVISION
    names = {entry.name for entry in manifest.artifacts}
    assert f"{ENV}/pair-contributions.npy" in names and "inputs.json" in names
    assert {f"{ENV}/{p.name}" for p in (run.directory / ENV).iterdir()} <= names
    inputs = json.loads((run.directory / "inputs.json").read_text())
    assert inputs["settings"]["budgets"] == [4, 256, 921]
    assert inputs[ENV]["qrels_train_sha256"]
    assert inputs[ENV]["model_revision"] == REVISION


def test_costs_are_recorded(run: Run) -> None:
    cost = run.summary["cost"]
    assert cost["total_seconds"] > 0 and cost["peak_rss_mib"] > 0
    assert cost["artifact_bytes"]["pair-order.npy"] > 0
    assert cost["device"] == "cpu" and cost["device_memory"] == {"name": "cpu"}
    assert {"read", "transfer", "compute", "write"} <= set(cost["phase_seconds"])


def test_a_rerun_gives_identical_results_apart_from_cost(run: Run, tmp_path: Path) -> None:
    build_inputs(tmp_path)
    assert analyze(tmp_path) == 0
    again = latest_artifact(tmp_path, "evaluation/P3", "all")
    other = json.loads((again / ENV / "summary.json").read_text())
    other.pop("cost")
    assert other == {k: v for k, v in run.summary.items() if k not in ("cost", "curves")}
    curves = json.loads((again / ENV / "curves.json").read_text())
    curves.pop("cost")
    assert curves == {k: v for k, v in run.summary["curves"].items() if k != "cost"}
    for name in ("pair-contributions.npy", "pair-order.npy", "frequency.npy",
                 "curve-jaccard-reference.npy", "reference-documents.npz"):  # fmt: skip
        assert _sha(again / ENV / name) == _sha(run.directory / ENV / name)


def test_batch_size_changes_neither_draws_nor_results(run: Run, tmp_path: Path) -> None:
    build_inputs(tmp_path)
    assert analyze(tmp_path, "--batch-pairs", "1") == 0
    again = latest_artifact(tmp_path, "evaluation/P3", "all") / ENV
    for name in ("pair-order.npy", "reference-documents.npz"):
        assert _sha(again / name) == _sha(run.directory / ENV / name)
    for name in ("curve-similarity-reference.npy", "curve-jaccard-positive.npy",
                 "curve-scaled_jaccard-reference.npy", "pair-contributions.npy"):  # fmt: skip
        found = np.load(again / name)
        assert np.allclose(found, np.load(run.directory / ENV / name), atol=1e-9, equal_nan=True)
    with (
        np.load(again / "control-stats.npz") as new,
        np.load(run.directory / ENV / "control-stats.npz") as old,
    ):
        for name in old.files:
            assert np.allclose(new[name], old[name], atol=1e-9, equal_nan=True), name


def test_parity_check_passes_on_equal_runs_and_fails_on_a_changed_array(
    run: Run, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    build_inputs(tmp_path)
    assert analyze(tmp_path, "--batch-pairs", "1") == 0
    other = latest_artifact(tmp_path, "evaluation/P3", "all")
    target = run.root / "evaluation" / "P3" / "all" / other.name
    shutil.copytree(other, target)
    parity = _script("p3_device_parity")
    try:
        assert parity.main([str(run.root), run.directory.name, other.name]) == 0
        curve = target / ENV / "curve-jaccard-positive.npy"
        changed = np.load(curve)
        changed[0, 0] += 0.5
        np.save(curve, changed)
        assert parity.main([str(run.root), run.directory.name, other.name]) == 1
        assert "curve-jaccard-positive" in capsys.readouterr().out
    finally:
        shutil.rmtree(target)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gpu_run_matches_the_cpu_run_and_records_its_device(run: Run, tmp_path: Path) -> None:
    build_inputs(tmp_path)
    assert analyze(tmp_path, device="cuda") == 0
    other = latest_artifact(tmp_path, "evaluation/P3", "all")
    cost = json.loads((other / ENV / "summary.json").read_text())["cost"]
    assert cost["device"].startswith("cuda") and cost["device_memory"]["peak_allocated_mib"] > 0
    for name in ("pair-order.npy", "reference-documents.npz"):
        assert _sha(other / ENV / name) == _sha(run.directory / ENV / name)
    found = np.load(other / ENV / "curve-jaccard-reference.npy")
    wanted = np.load(run.directory / ENV / "curve-jaccard-reference.npy")
    assert np.allclose(found, wanted, atol=1e-6, equal_nan=True)


def test_run_index_labels_every_run_from_its_manifest(run: Run, tmp_path: Path) -> None:
    output = tmp_path / "RUNS.md"
    index = _script("p3_run_index")
    assert (
        index.main([str(run.root), str(output), "--label", f"{run.directory.name}=canonical"]) == 0
    )
    text = output.read_text()
    assert run.directory.name in text and "canonical" in text and "cpu" in text
    assert "not reviewed" not in text.split(run.directory.name)[1].splitlines()[0]


def test_max_queries_bounds_the_run_with_a_seeded_subset(tmp_path: Path) -> None:
    build_inputs(tmp_path)
    assert analyze(tmp_path, "--max-queries", "5") == 0
    directory = latest_artifact(tmp_path, "evaluation/P3", "all")
    summary = json.loads((directory / ENV / "summary.json").read_text())
    assert summary["coverage"]["selected_queries"] == 5
    assert summary["coverage"]["training_queries"] == TRAIN


def test_embeddings_of_another_revision_are_refused(tmp_path: Path) -> None:
    build_inputs(tmp_path)
    assert (
        main(
            [
                "analyze-representations",
                "--config-dir",
                str(CONFIGS),
                "--output",
                str(tmp_path),
                "--envs",
                ENV,
                "model.revision=" + "0" * 40,
                *BUDGETS,
            ]
        )  # fmt: skip
        == 1
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def report(run: Run, tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("report")
    assert main(["report", "P3", "--artifacts", str(run.root), "--report-dir", str(output)]) == 0
    return output


def test_report_regenerates_byte_identically(run: Run, report: Path, tmp_path: Path) -> None:
    assert main(["report", "P3", "--artifacts", str(run.root), "--report-dir", str(tmp_path)]) == 0
    first = sorted(p.relative_to(report) for p in report.rglob("*") if p.is_file())
    assert first == sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*") if p.is_file())
    for name in first:
        assert (report / name).read_bytes() == (tmp_path / name).read_bytes()


def test_report_is_organized_around_the_five_questions(report: Path) -> None:
    text = (report / "README.md").read_text()
    headings = [
        "## 1. What was directly observed",
        "## 2. What is stable within the analyzed training data",
        "## 3. What aggregation conceals",
        "## 4. Questions that remain unanswered",
        "## 5. Follow-up hypotheses these observations motivate",
    ]
    positions = [text.index(heading) for heading in headings]
    assert positions == sorted(positions)
    assert "**Improving a ranking**: not measured" in text
    assert "not claim to hold for unseen queries" in " ".join(text.split())


def test_report_marks_what_is_missing_and_keeps_query_text_out(report: Path) -> None:
    text = (report / "README.md").read_text()
    assert "**Not analyzed yet:** msmarco, nfcorpus." in text
    assert "Validation only" not in text  # these inputs carry no spike provenance
    for path in report.rglob("*"):
        if path.is_file():
            assert SECRET not in path.read_text(encoding="utf-8")


def test_report_figures_are_well_formed_and_finite(report: Path) -> None:
    figures = sorted((report / "figures").glob("*.svg"))
    assert {f.stem for f in figures} >= {
        "cumulative", f"agreement-{ENV}", "conflict", "cancelled", f"frequency-{ENV}",
        f"distribution-{ENV}", "stability", f"similarity-{ENV}", f"retained-{ENV}",
        f"removed-{ENV}", f"agreement-difference-{ENV}", f"agreement-scaled-{ENV}",
        f"agreement-strata-{ENV}", f"frequency-control-{ENV}", "group-size-conflict",
        "group-size-cancelled",
    }  # fmt: skip
    for figure in figures:
        ET.fromstring(figure.read_text(encoding="utf-8"))
        assert "nan" not in figure.read_text(encoding="utf-8").lower().split("<style>")[0]
        body = figure.read_text(encoding="utf-8").split("</style>")[1]
        assert "nan" not in body.replace("nfcorpus", "") and "inf" not in body


def test_curves_cover_every_cardinality_with_a_labeled_empty_mask(run: Run) -> None:
    curves = run.summary["curves"]
    assert curves["k"] == list(range(1, 1025))
    assert (
        curves["empty_mask_reference"]["k"] == 0
        and "reference" in (curves["empty_mask_reference"]["note"])
    )
    table = curves["pair_curves"]["similarity"]["all"]["positive"]
    assert len(table["mean"]) == 1024 and table["n"] == TRAIN
    assert np.load(run.directory / ENV / "curve-similarity-positive.npy").shape == (TRAIN, 1024)
    # 18 multi-positive queries: agreement is undefined for a single document.
    assert np.load(run.directory / ENV / "curve-jaccard-positive.npy").shape == (18, 1024)


def test_full_cardinality_returns_to_the_unmasked_pair_and_all_masks_coincide(run: Run) -> None:
    curves = run.summary["curves"]
    assert curves["consistency_at_full"]["similarity"] < 1e-12
    assert curves["consistency_at_full"]["retained_positive_share"] < 1e-12
    assert curves["consistency_at_full"]["removed_negative_share"] < 1e-12
    assert curves["consistency_at_full"]["jaccard"] == 0.0
    similarity = curves["pair_curves"]["similarity"]["all"]["positive"]["mean"]
    assert similarity[-1] == pytest.approx(run.summary["pairs"]["scalar"]["score"]["all"]["mean"])
    agreement = curves["agreement"]["jaccard"]["multi"]
    assert agreement["positive"]["mean"][-1] == 1.0 and agreement["reference"]["mean"][-1] == 1.0
    assert agreement["difference"]["mean"][-1] == 0.0
    # Undefined rather than perfect: every dimension is always selected at k = 1024.
    assert curves["frequency"]["raw"]["positive_vs_magnitude"][-1] is None


def test_reference_groups_are_matched_in_size_and_avoid_each_query_judged_documents(
    run: Run,
) -> None:
    drawn = np.load(run.directory / ENV / "reference-documents.npz")
    judged = {r.query_id: set() for r in _judgments()}
    positives = dict.fromkeys(judged, 0)
    for record in _judgments():
        judged[record.query_id].add(int(record.doc_id[1:]))
        positives[record.query_id] += record.relevance > 0
    ids = sorted(judged)
    assert set(drawn["repeat"].tolist()) == {0, 1, 2}
    for index, qid in enumerate(ids):
        for repeat in range(3):
            rows = drawn["document_row"][
                (drawn["query_index"] == index) & (drawn["repeat"] == repeat)
            ]
            assert rows.size == positives[qid] and rows.size == len(set(rows.tolist()))
            assert not judged[qid] & set(rows.tolist())
    assert run.summary["curves"]["coverage"]["reference_groups_short_of_documents"] == 0


def test_repeated_reference_samples_differ_and_their_band_holds_their_mean(run: Run) -> None:
    drawn = np.load(run.directory / ENV / "reference-documents.npz")
    assert any(
        (
            drawn["document_row"][(drawn["query_index"] == q) & (drawn["repeat"] == 0)]
            != drawn["document_row"][(drawn["query_index"] == q) & (drawn["repeat"] == 1)]
        ).any()
        for q in range(TRAIN)
    )
    curves = run.summary["curves"]
    band = curves["reference_repeats"]["jaccard"]
    mean = curves["agreement"]["jaccard"]["multi"]["reference"]["mean"]
    assert all(
        low - 1e-6 <= value <= high + 1e-6
        for low, value, high in zip(band["min"], mean, band["max"], strict=True)
    )


def test_relevant_and_reference_frequency_share_one_scale(run: Run) -> None:
    arrays = np.load(run.directory / ENV / "frequency-curves.npz")
    k = np.arange(1, 1025)
    for name in ("positive_raw", "reference_raw", "positive_scaled", "reference_scaled"):
        assert arrays[name].sum(axis=1) == pytest.approx(k, rel=1e-4)
        assert arrays[name][-1] == pytest.approx(np.ones(1024), rel=1e-5)
    frequency = run.summary["curves"]["frequency"]
    for variant in ("raw", "scaled"):
        assert all(0.0 <= value <= 1.0 + 1e-9 for value in frequency[variant]["total_variation"])
        assert frequency[variant]["total_variation"][-1] == pytest.approx(0.0, abs=1e-9)


def test_every_control_reports_both_groups_at_a_fixed_document_count(run: Run) -> None:
    rows = {row["size"]: row for row in run.summary["curves"]["group_size"]}
    assert rows[2]["queries"] == 18 and rows[5]["queries"] == 0
    assert rows[2]["sign_conflict_share"]["positive"]["n"] == 18
    assert rows[2]["sign_conflict_share"]["reference"]["n"] == 18
    assert rows[5]["cancelled_share"]["difference"]["n"] == 0
    assert run.summary["curves"]["scalars"]["document_cosine"]["multi"]["reference"]["n"] == 18


def test_report_exposes_full_curves_and_names_cardinality_effects(report: Path) -> None:
    table = (report / "tables" / f"curves-{ENV}.csv").read_text().splitlines()
    assert len(table) == 1 + 1 + 1024  # header, the labeled k = 0 row, every k
    assert table[1].startswith("0,empty-mask reference")
    assert table[-1].startswith("1024,")
    frequency = (report / "tables" / f"frequency-curves-{ENV}.csv").read_text().splitlines()
    assert len(frequency) == 1 + 1024
    text = " ".join((report / "README.md").read_text().split())
    assert "are not evidence of shared relevance structure" in text
    assert "Reference documents are unjudged, not negatives" in text
    assert "307" not in (REPOSITORY / "src/hyperdime/evaluation/representation.py").read_text()


def test_analysis_never_ranks_a_document_or_loads_a_selector() -> None:
    source = (REPOSITORY / "src/hyperdime/evaluation/representation.py").read_text()
    source += (REPOSITORY / "src/hyperdime/evaluation/representation_controls.py").read_text()
    command = (REPOSITORY / "src/hyperdime/cli/analyze_representations.py").read_text()
    for forbidden in ("exact_search", "query_metrics", "load_selector", "evaluate_methods",
                      "load_environment", "load_inputs"):  # fmt: skip
        assert forbidden not in source and forbidden not in command


def _converter() -> Any:
    path = REPOSITORY / "scripts/checks/p3_spike_inputs.py"
    spec = importlib.util.spec_from_file_location("p3_spike_inputs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_spike_conversion_keeps_only_training_judgments_and_labels_its_provenance(
    tmp_path: Path,
) -> None:
    spike = tmp_path / "spike"
    rng = np.random.default_rng(1)
    for directory, ids in (("corpus", [f"d{i}" for i in range(6)]),
                           ("queries", [f"q{i}" for i in range(20)])):  # fmt: skip
        target = spike / ENV / directory
        target.mkdir(parents=True)
        np.save(target / "embeddings.npy", _unit(rng, len(ids)))
        (target / "ids.json").write_text(json.dumps(ids))
        (target / "settings.json").write_text(json.dumps({"model": f"m@{REVISION}"}))
    (spike / ENV / "manifest.json").write_text(json.dumps({"config": {"instruction": "x"}}))
    # Unparsable, so opening either of them breaks the conversion.
    (spike / ENV / "splits.json").write_text(NOT_JSON)
    qrels = spike / "beir" / ENV / "qrels"
    qrels.mkdir(parents=True)
    (qrels / "test.tsv").write_bytes(b"\xff\xfe not text")
    rows = "".join(f"q{i}\td{i % 6}\t1\n" for i in range(20))
    (qrels / "train.tsv").write_text("query-id\tcorpus-id\tscore\n" + rows + "q0\tmissing\t1\n")

    _converter().convert(spike, tmp_path / "out", ENV)

    dataset = latest_artifact(tmp_path / "out", "datasets", ENV)
    assert not (dataset / "qrels-validation.jsonl").exists()
    assert not (dataset / "qrels-test.jsonl").exists()
    queries = [json.loads(line) for line in (dataset / "queries.jsonl").read_text().splitlines()]
    assert sum(q["split"] == "train" for q in queries) == 18  # F2's 10% rule on 20 queries
    assert {q["text"] for q in queries} == {"<withheld>"}
    train = {q["query_id"] for q in queries if q["split"] == "train"}
    judged = [json.loads(line) for line in (dataset / "qrels-train.jsonl").read_text().splitlines()]
    assert {j["query_id"] for j in judged} == train
    manifest = json.loads((dataset / "dataset.json").read_text())
    assert "F0 spike" in manifest["spec"]["provenance"]
    command = [
        "analyze-representations", "--config-dir", str(CONFIGS), "--output",
        str(tmp_path / "out"), "--envs", ENV, *BUDGETS, "--stability-draws", "2",
        "--reference-samples", "2",
    ]  # fmt: skip
    # Spike inputs are never picked up silently.
    assert main(command) == 1
    assert main([*command, "--allow-spike-inputs"]) == 0
    assert main([
        "report", "P3", "--artifacts", str(tmp_path / "out"), "--report-dir",
        str(tmp_path / "report"),
    ]) == 0  # fmt: skip
    text = (tmp_path / "report" / "README.md").read_text()
    assert "**Validation only, canonical results pending (scifact):**" in text
    # ...and their report cannot land where the canonical one lives.
    assert main([
        "report", "P3", "--artifacts", str(tmp_path / "out"), "--report-dir",
        str(tmp_path / "P3"),
    ]) == 1  # fmt: skip
    assert not (tmp_path / "P3" / "README.md").exists()


def _differences() -> Any:
    path = REPOSITORY / "scripts/checks/p3_input_differences.py"
    spec = importlib.util.spec_from_file_location("p3_input_differences", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_input_differences_are_recorded_with_a_manifest_and_reach_the_report(
    tmp_path: Path,
) -> None:
    published, spike = tmp_path / "published", tmp_path / "spike"
    build_inputs(published)
    build_inputs(published, env="nfcorpus")  # no spike counterpart
    extra = QrelsRecord(query_id="q0", doc_id="d39", relevance=1, environment=ENV, split="train")
    build_inputs(spike, [extra])

    assert _differences().main([str(published), str(spike), "--envs", ENV, "nfcorpus"]) == 0

    directory = latest_artifact(published, "evaluation/P3-inputs", "all")
    assert read_manifest(directory).command == "p3-input-differences"
    found = json.loads((directory / "differences.json").read_text())
    assert found["nfcorpus"] == {"spike_inputs": False}
    row = found[ENV]
    assert row["same"]["documents_payload"] is True and row["same"]["model_revision"] is True
    assert row["same"]["train_judgments_file"] is False
    assert row["train_judgments"]["only_spike"] == 1 and row["train_judgments"]["only_f2"] == 0
    assert row["train_queries"] == {"shared": TRAIN, "only_f2": 0, "only_spike": 0}
    assert row["document_vectors"]["rows"] == DOCUMENTS
    assert row["document_vectors"]["min_cosine"] == pytest.approx(1.0)

    assert analyze(published) == 0
    output = tmp_path / "report"
    assert main(["report", "P3", "--artifacts", str(published), "--report-dir", str(output)]) == 0
    table = (output / "tables" / "input-differences.csv").read_text()
    assert "train_judgments_file" in table and "nfcorpus,none" in table


def test_a_payload_that_does_not_match_its_hash_stops_the_run(tmp_path: Path) -> None:
    _, embeddings = build_inputs(tmp_path)
    target = embeddings / "documents.npy"
    target.chmod(0o644)
    vectors = np.load(target)
    vectors[0, 0] = 0.5
    np.save(target, vectors)
    assert analyze(tmp_path) == 1


def test_verification_is_recorded_for_each_environment(run: Run) -> None:
    check = json.loads((run.directory / "inputs.json").read_text())[ENV]["verification"]
    assert check["payload_hashes_match"] is True and check["dim"] == 1024
    assert check["queries_per_split"] == {"test": 4, "train": TRAIN, "validation": 4}
    assert check["instruction"] == "x"
    low, high = check["document_norm_range"]
    assert abs(low - 1.0) < 0.01 and abs(high - 1.0) < 0.01
