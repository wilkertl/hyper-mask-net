"""F12 P3-bench end to end on a tiny CPU corpus (T056), with the leakage (T049), replay (T050),
recorded-negative (T055), and no-test-label (T058) checks.

The fixture is an F2-shaped artifacts root for ``scifact``: a few fit, tune, and evaluation
queries, 1024-dimensional float16 vectors, and qrels in ``qrels-train/validation/test.jsonl``.
The authors' clone at the pinned commit is needed for the official reference; without it the
module is skipped. S1 runs a shortened schedule (the step budget is a module constant), and its
soft-mass guard is disabled because a few steps cannot sharpen the mask.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from hyperdime.baselines import l2s_official
from hyperdime.baselines.l2s_official import DEFAULT_CLONE, L2S_COMMIT, clone_commit
from hyperdime.cli import p3_bench, p3_bench_arms
from hyperdime.cli.__main__ import main
from hyperdime.contracts.hashing import file_sha256
from hyperdime.data import query_split
from hyperdime.evaluation.benchmark import PRIMARY
from hyperdime.evaluation.teachers import contrastive_mask, positive_mask, positive_mean

CORPUS, DATASET_RUN, EMBEDDING_RUN = (
    "scifact",
    "20260101T000000Z-data0000",
    "20260101T000000Z-emb00000",
)
N_DOCS, DIM = 120, 1024
SPLIT_IDS = {
    "fit": ["f0", "f1", "f2", "f3"],
    "tune": ["t0", "t1", "t2"],
    "evaluation": ["e0", "e1", "e2"],
}


def _clone_ready() -> bool:
    try:
        return clone_commit(DEFAULT_CLONE) == L2S_COMMIT
    except (OSError, subprocess.CalledProcessError):
        return False


pytestmark = pytest.mark.skipif(not _clone_ready(), reason="authors' clone at the pinned commit")


def _jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def build_root(root: Path, poison: bool = False) -> None:
    """The fixture corpus. ``poison`` replaces every tune and evaluation judgment (and the test
    split's) with different random labels; fit judgments and all vectors stay the same."""
    rng = np.random.default_rng(5)
    docs = rng.standard_normal((N_DOCS, DIM))
    docs /= np.linalg.norm(docs, axis=1, keepdims=True)
    doc_ids = [f"d{i:03d}" for i in range(N_DOCS)]
    all_ids = [q for ids in SPLIT_IDS.values() for q in ids] + ["x0"]
    queries, qrels = {}, {}
    for qi, qid in enumerate(all_ids):
        # fit queries have one or two positives, the others one: S1's tune pools (every
        # non-positive document, as the corpus is under 1,000 documents) then share a width
        count = 1 + qi % 2 if qid in SPLIT_IDS["fit"] else 1
        positives = [(qi * 7) % N_DOCS, (qi * 7 + 3) % N_DOCS][:count]
        vector = docs[positives].mean(0) + 0.05 * rng.standard_normal(DIM)
        queries[qid] = vector / np.linalg.norm(vector)
        judged = {doc_ids[p]: 1 for p in positives}
        judged[doc_ids[(qi * 11 + 50) % N_DOCS]] = 0  # a known negative
        qrels[qid] = judged
    if poison:
        bad = np.random.default_rng(99)
        for qid in [*SPLIT_IDS["tune"], *SPLIT_IDS["evaluation"], "x0"]:
            qrels[qid] = {doc_ids[int(bad.integers(N_DOCS))]: 1}
    dataset = root / "datasets" / CORPUS / DATASET_RUN
    dataset.mkdir(parents=True)
    _jsonl(
        dataset / "documents.jsonl",
        [{"doc_id": d, "title": "t" * (i % 3), "text": "x" * (5 + i % 4)}
         for i, d in enumerate(doc_ids)],
    )  # fmt: skip
    origin = {"f0": "train", "f1": "train", "f2": "validation", "f3": "train", "t0": "train",
              "t1": "validation", "t2": "train", "e0": "train", "e1": "validation",
              "e2": "train", "x0": "test"}  # fmt: skip
    _jsonl(
        dataset / "queries.jsonl",
        [{"query_id": q, "text": f"query text {q}", "split": origin[q]} for q in all_ids],
    )
    for split in ("train", "validation", "test"):
        _jsonl(
            dataset / f"qrels-{split}.jsonl",
            [
                {"query_id": q, "doc_id": d, "relevance": r}
                for q in all_ids
                if origin[q] == split
                for d, r in qrels[q].items()
            ],
        )
    emb = root / "embeddings" / CORPUS / EMBEDDING_RUN
    emb.mkdir(parents=True)
    (emb / "documents.json").write_text(json.dumps({"ids": doc_ids}))
    np.save(emb / "documents.npy", docs.astype(np.float16))
    (emb / "queries.json").write_text(json.dumps({"ids": all_ids}))
    np.save(emb / "queries.npy", np.stack([queries[q] for q in all_ids]).astype(np.float16))
    split_file = root / query_split.SPLIT_MANIFEST
    split_file.parent.mkdir(parents=True)
    entry = {
        "source_dataset_run": DATASET_RUN,
        "source_queries_sha256": file_sha256(dataset / "queries.jsonl"),
        "excluded_test_overlap_ids": [],
        "counts": {k: len(v) for k, v in SPLIT_IDS.items()},
        "query_ids": SPLIT_IDS,
    }
    split_file.write_text(json.dumps({"seed": 20261112, "corpora": {CORPUS: entry}}))


def _run(*argv: str | Path) -> int:
    if argv[0] == "official":
        # each official run stands for its own process (the adapter seeds once per process)
        l2s_official._RAN = False
    return main(["p3-bench", *map(str, argv)])


def _latest(directory: Path) -> str:
    return sorted(p.name for p in directory.iterdir() if p.is_dir())[-1]


@pytest.fixture(scope="module")
def patched() -> Any:
    patch = pytest.MonkeyPatch()
    patch.setattr(p3_bench_arms, "S1_STEPS", 30)
    patch.setattr(p3_bench_arms, "S1_EVAL_EVERY", 10)
    patch.setattr(p3_bench_arms, "S1_GUARD", 0.0)
    opened: list[str] = []
    original = pathlib.Path.open

    def recording_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        opened.append(self.name)
        return original(self, *args, **kwargs)

    patch.setattr(pathlib.Path, "open", recording_open)
    yield patch, opened
    patch.undo()


def _pin_split(patch: pytest.MonkeyPatch, root: Path) -> None:
    digest = file_sha256(root / query_split.SPLIT_MANIFEST)
    patch.setitem(query_split.load_query_split.__kwdefaults__, "expected_sha256", digest)


def _reference_and_teachers(root: Path, records: Path) -> tuple[str, str]:
    assert _run("official", "--root", root, "--corpus", CORPUS, "--purpose", "reference",
                "--resource-record", records / "resources.json") == 0  # fmt: skip
    official = _latest(root / "selectors" / "l2s-official" / CORPUS)
    assert _run("teachers", "--root", root, "--corpus", CORPUS, "--official-run", official,
                "--resource-record", records / "resources.json") == 0  # fmt: skip
    return official, _latest(root / "evaluation" / "P3-bench" / "teachers" / CORPUS)


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory: pytest.TempPathFactory, patched: Any) -> dict[str, Any]:
    patch, opened = patched
    base = tmp_path_factory.mktemp("p3bench")
    root, records, repo = base / "artifacts", base / "freeze", base / "repo"
    build_root(root)
    opened.clear()  # only reads by the pipeline count, not the fixture's own writes
    _pin_split(patch, root)
    for name in p3_bench.PROTOCOL_FILES:
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(f"frozen {name}\n")
    assert _run("split-provenance", "--root", root, "--corpus", CORPUS) == 0
    assert _run("freeze-protocol", "--repo", repo, "--out", records / "protocol.json") == 0
    assert _run("official", "--root", root, "--corpus", CORPUS, "--purpose", "probe",
                "--limit", 4) == 0  # fmt: skip
    labels = _latest(root / "evaluation" / "P3-bench" / "probe-official" / CORPUS)
    assert _run("probe", "--root", root, "--corpus", CORPUS, "--labels-run", labels,
                "--protocol-record", records / "protocol.json", "--repo", repo) == 0  # fmt: skip
    probe = _latest(root / "evaluation" / "P3-bench" / "probe" / CORPUS)
    assert _run("freeze-resources", "--root", root, "--corpus", CORPUS, "--probe-run", probe,
                "--out", records / "resources.json") == 0  # fmt: skip
    official, teachers = _reference_and_teachers(root, records)
    common = ["--root", root, "--corpus", CORPUS, "--resource-record", records / "resources.json"]
    assert _run("train-arm", *common, "--official-run", official, "--target", "official",
                "--seed", 42) == 0  # fmt: skip
    replay = _latest(root / "selectors" / "p3-teacher-arms" / CORPUS)
    runs: dict[str, str] = {"official": official, "teachers": teachers}
    for teacher in p3_bench_arms.TEACHERS:
        for seed in (42, 0, 1, 2):
            assert _run("train-arm", *common, "--official-run", official, "--teachers-run",
                        teachers, "--target", teacher, "--seed", seed) == 0  # fmt: skip
            runs[f"arm:{teacher}@{seed}"] = _latest(root / "selectors" / "p3-teacher-arms" / CORPUS)
    for arm in ("scratch", "official-init"):
        assert _run("train-s1", *common, "--official-run", official, "--arm", arm) == 0
        runs[f"s1:{arm}"] = _latest(root / "selectors" / "p3-s1" / CORPUS / arm)
    (records / "runs.json").write_text(json.dumps(runs))
    family = records / "family.json"
    family.write_text(json.dumps({"members": [f"{CORPUS}:{t}" for t in p3_bench_arms.TEACHERS]}))
    assert _run("evaluation-lock", *common, "--runs", records / "runs.json", "--family", family,
                "--out", records / "lock.json") == 0  # fmt: skip
    assert _run("evaluate", *common, "--lock-record", records / "lock.json") == 0
    evaluation = _latest(root / "evaluation" / "P3-bench" / "evaluation" / CORPUS)
    (records / "evaluations.json").write_text(json.dumps({CORPUS: evaluation}))
    assert _run("family", "--root", root, "--family", family, "--evaluations",
                records / "evaluations.json") == 0  # fmt: skip
    assert _run("report", *common, "--evaluation-run", evaluation,
                "--lock-record", records / "lock.json") == 0  # fmt: skip
    assert _run("teacher-report", "--root", root, "--corpus", CORPUS, "--teachers-run",
                teachers) == 0  # fmt: skip
    return {"base": base, "root": root, "records": records, "runs": runs, "replay": replay,
            "evaluation": evaluation, "opened": list(opened)}  # fmt: skip


def _dir(p: dict[str, Any], *parts: str) -> Path:
    return Path(p["root"], *parts)


def test_every_stage_writes_its_artifacts(pipeline: dict[str, Any]) -> None:
    evaluation = _dir(
        pipeline, "evaluation", "P3-bench", "evaluation", CORPUS, pipeline["evaluation"]
    )
    for name in ("manifest.json", "summary.json", "per_query.csv", "README.md"):
        assert (evaluation / name).is_file()
    summary = json.loads((evaluation / "summary.json").read_text())
    assert summary["full_vector_parity_at_1024"] is True
    assert set(summary["primary"]) == set(p3_bench_arms.TEACHERS)
    assert "mrl-prefix" in summary["secondary"]
    assert summary["provenance"]["l2s_commit"] == L2S_COMMIT
    family = _dir(pipeline, "evaluation", "P3-bench", "family")
    assert (family / _latest(family) / "family.csv").is_file()
    report = _dir(pipeline, "evaluation", "P3-bench", "report", CORPUS)
    text = (report / _latest(report) / "README.md").read_text()
    assert "## Offline cost" in text and "## Online cost per query" in text
    teacher = _dir(pipeline, "evaluation", "P3-bench", "teacher-report", CORPUS)
    assert "## Demotions" in (teacher / _latest(teacher) / "README.md").read_text()


def test_the_replay_control_reproduces_the_official_weights(pipeline: dict[str, Any]) -> None:
    """T050: fed the reference's own targets, the arm pipeline gives its weights bit for bit."""
    replay = _dir(pipeline, "selectors", "p3-teacher-arms", CORPUS, pipeline["replay"])
    official = _dir(pipeline, "selectors", "l2s-official", CORPUS, pipeline["runs"]["official"])
    assert json.loads((replay / "summary.json").read_text())["replay_equals_official"] is True
    a, b = torch.load(replay / "state_dict.pt"), torch.load(official / "state_dict.pt")
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_contrastive_masks_use_the_recorded_negatives(pipeline: dict[str, Any]) -> None:
    """T055: the teachers rebuild ``n̄`` from the reference's recorded negative rows."""
    official = _dir(pipeline, "selectors", "l2s-official", CORPUS, pipeline["runs"]["official"])
    teachers = _dir(pipeline, "evaluation", "P3-bench", "teachers", CORPUS,
                    pipeline["runs"]["teachers"])  # fmt: skip
    valid = json.loads((official / "valid_query_ids.json").read_text())
    negatives = np.load(official / "negatives_f2_rows.npy")
    inputs = p3_bench.load_fit_inputs(pipeline["root"], CORPUS, 4)
    docs = np.asarray(inputs.documents, dtype=np.float64)
    rows = {d: i for i, d in enumerate(inputs.doc_ids)}
    masks = np.load(teachers / "masks_contrastive.npy")
    positives = np.load(teachers / "masks_positive.npy")
    for qi, qid in enumerate(valid):
        judged = [rows[d] for d, r in inputs.qrels[qid].items() if r > 0]
        q = inputs.queries[inputs.query_ids.index(qid)].astype(np.float64)
        p_bar = positive_mean(docs[judged], np.ones(len(judged)))
        n_bar = docs[negatives[qi]].mean(0)
        for ki, k in enumerate(PRIMARY):
            assert np.array_equal(
                np.flatnonzero(masks[qi, ki]), contrastive_mask(q, p_bar, n_bar, k)
            )
            assert np.array_equal(np.flatnonzero(positives[qi, ki]), positive_mask(q, p_bar, k))


def test_tune_and_evaluation_labels_do_not_reach_teachers_or_teacher_arms(
    pipeline: dict[str, Any], patched: Any
) -> None:
    """T049 / SC-003: with every tune, evaluation, and test judgment replaced, the reference, the
    teacher masks and targets, and a teacher arm's weights are byte-identical."""
    patch, _ = patched
    base, records = pipeline["base"], pipeline["records"]
    root = base / "poisoned"
    build_root(root, poison=True)
    _pin_split(patch, root)  # same split manifest, so the same digest
    official, teachers = _reference_and_teachers(root, records)
    assert _run("train-arm", "--root", root, "--corpus", CORPUS, "--resource-record",
                records / "resources.json", "--official-run", official, "--teachers-run",
                teachers, "--target", "positive", "--seed", 42) == 0  # fmt: skip
    poisoned_arm = _latest(root / "selectors" / "p3-teacher-arms" / CORPUS)
    clean = pipeline["runs"]
    pairs = [
        (f"selectors/l2s-official/{CORPUS}/{clean['official']}",
         f"selectors/l2s-official/{CORPUS}/{official}",
         ("state_dict.pt", "targets.pt", "negatives_f2_rows.npy")),
        (f"evaluation/P3-bench/teachers/{CORPUS}/{clean['teachers']}",
         f"evaluation/P3-bench/teachers/{CORPUS}/{teachers}",
         ("masks_positive.npy", "masks_contrastive.npy", "masks_rank-guided.npy",
          "targets_positive.pt", "global_mask.npy")),
    ]  # fmt: skip
    for clean_dir, poisoned_dir, files in pairs:
        for name in files:
            a = file_sha256(pipeline["root"] / clean_dir / name)
            assert a == file_sha256(root / poisoned_dir / name), name
    a = torch.load(_dir(pipeline, "selectors", "p3-teacher-arms", CORPUS, clean["arm:positive@42"])
                   / "state_dict.pt")  # fmt: skip
    b = torch.load(root / "selectors" / "p3-teacher-arms" / CORPUS / poisoned_arm / "state_dict.pt")
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_no_stage_opens_the_official_test_judgments(pipeline: dict[str, Any]) -> None:
    """T058 / FR-002: across every stage, ``qrels-test.jsonl`` is never opened."""
    assert "qrels-test.jsonl" not in pipeline["opened"]
    assert "qrels-validation.jsonl" in pipeline["opened"]  # the recorder works
