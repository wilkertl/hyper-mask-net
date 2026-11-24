"""The adapter reproduces the authors' Learning2Select run exactly (F12 T010, research B3).

Two isolated CPU processes with seed 42 on one synthetic BEIR folder: the authors' unmodified
``train_and_eval.py`` and our adapter fed the same data in F2 order. Final weights must be equal bit
for bit. The fixture has ties in document length (corpus order), a graded positive, a judged query
without positives, and queries whose pool after removing positives is smaller than 64 (fallback).
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from hyperdime.baselines.l2s_official import DEFAULT_CLONE, L2S_COMMIT, authors_order, clone_commit


def _clone_ready() -> bool:
    try:
        return clone_commit(DEFAULT_CLONE) == L2S_COMMIT
    except (OSError, subprocess.CalledProcessError):
        return False


needs_clone = pytest.mark.skipif(not _clone_ready(), reason="authors' clone at the pinned commit")


def test_authors_order_sorts_by_length_descending_keeping_file_order_among_ties() -> None:
    assert authors_order([3, 5, 3, 7, 5]) == [3, 1, 4, 0, 2]


def _fixture(root: Path) -> dict:
    rng = np.random.default_rng(7)
    n_docs, dim = 70, 32
    doc_ids = [f"d{i}" for i in range(n_docs)]
    titles = ["t" * int(rng.integers(0, 3)) for _ in range(n_docs)]
    texts = ["x" * int(rng.integers(5, 12)) for _ in range(n_docs)]  # many equal lengths
    lengths = [len(t + x) for t, x in zip(titles, texts, strict=True)]
    order = authors_order(lengths)

    def unit(rows: int) -> np.ndarray:
        values = rng.standard_normal((rows, dim))
        values /= np.linalg.norm(values, axis=1, keepdims=True)
        return values.astype(np.float16).astype(np.float32)  # F2 stores float16

    documents = unit(n_docs)
    train_ids = [f"q{i}" for i in range(24)]
    test_ids = [f"s{i}" for i in range(4)]
    queries = {qid: unit(1)[0] for qid in train_ids + test_ids}
    qrels: dict[str, dict[str, int]] = {}
    for i, qid in enumerate(train_ids):
        if i == 3:
            qrels[qid] = {doc_ids[5]: 0}  # judged, no positive: skipped by the authors
        elif i % 6 == 0:
            qrels[qid] = {doc_ids[j]: 1 for j in rng.choice(n_docs, 8, replace=False)}  # fallback
        else:
            picks = rng.choice(n_docs, 2, replace=False)
            qrels[qid] = {doc_ids[picks[0]]: 2, doc_ids[picks[1]]: 1}
    test_qrels = {qid: {doc_ids[int(rng.integers(n_docs))]: 1} for qid in test_ids}

    data = root / "data" / "d"
    (data / "qrels").mkdir(parents=True)
    with (data / "corpus.jsonl").open("w") as handle:
        for i in range(n_docs):
            handle.write(
                json.dumps({"_id": doc_ids[i], "title": titles[i], "text": texts[i]}) + "\n"
            )
    with (data / "queries.jsonl").open("w") as handle:
        for qid in train_ids + test_ids:
            handle.write(json.dumps({"_id": qid, "text": qid}) + "\n")
    for name, table in (("train", qrels), ("test", test_qrels)):
        with (data / "qrels" / f"{name}.tsv").open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["query-id", "corpus-id", "score"])
            for qid, judged in table.items():
                for doc_id, rel in judged.items():
                    writer.writerow([qid, doc_id, rel])
    emb = root / "emb" / "m" / "d"
    (emb / "train").mkdir(parents=True)
    (emb / "test").mkdir(parents=True)
    torch.save(
        torch.from_numpy(np.stack([queries[q] for q in train_ids])),
        emb / "train" / "queries_embeddings.pt",
    )
    torch.save(
        torch.from_numpy(np.stack([queries[q] for q in test_ids])),
        emb / "test" / "queries_embeddings.pt",
    )
    torch.save(torch.from_numpy(documents[order]), emb / "test" / "corpus_embeddings.pt")
    (root / "hyps.json").write_text(json.dumps({"m": {"d": {"epoch": 3, "temp": 0.05}}}))
    np.savez(
        root / "arrays.npz", documents=documents, queries=np.stack([queries[q] for q in train_ids])
    )
    spec = {
        "clone": str(DEFAULT_CLONE),
        "arrays": str(root / "arrays.npz"),
        "doc_ids": doc_ids,
        "lengths": lengths,
        "query_ids": train_ids,
        "qrels": qrels,
        "temperature": 0.05,
        "epochs": 3,
        "device": "cpu",
    }
    (root / "spec.json").write_text(json.dumps(spec))
    return spec


@needs_clone
def test_adapter_matches_the_authors_script_bit_for_bit(tmp_path: Path) -> None:
    _fixture(tmp_path)
    authors = subprocess.run(
        [
            sys.executable,
            str(DEFAULT_CLONE / "train_and_eval.py"),
            "--models",
            "m",
            "--datasets",
            "d",
            "--base-data-path",
            str(tmp_path / "data"),
            "--base-emb-path",
            str(tmp_path / "emb"),
            "--base-result-path",
            str(tmp_path / "results"),
            "--save-dir",
            str(tmp_path / "saved"),
            "--best-hyperparams-file",
            str(tmp_path / "hyps.json"),
            "--device",
            "cpu",
        ],
        cwd=DEFAULT_CLONE,
        capture_output=True,
        text=True,
    )
    assert authors.returncode == 0, authors.stderr[-2000:]
    ours = subprocess.run(
        [
            sys.executable,
            "-m",
            "hyperdime.baselines.l2s_official",
            str(tmp_path / "spec.json"),
            str(tmp_path / "ours"),
        ],
        capture_output=True,
        text=True,
    )
    assert ours.returncode == 0, ours.stderr[-2000:]
    reference = torch.load(tmp_path / "saved" / "m" / "d.pt", weights_only=False)[
        "model_state_dict"
    ]
    adapted = torch.load(tmp_path / "ours" / "state_dict.pt")
    assert reference.keys() == adapted.keys()
    for name in reference:
        assert torch.equal(reference[name], adapted[name]), name
    valid = json.loads((tmp_path / "ours" / "valid.json").read_text())
    fallback = np.load(tmp_path / "ours" / "fallback.npy")
    negatives = np.load(tmp_path / "ours" / "negatives.npy")
    assert "q3" not in valid and len(valid) == 23
    assert fallback.sum() == 4 and negatives.shape == (23, 64)


@needs_clone
def test_recorded_pools_give_the_same_run_without_a_second_corpus_copy(tmp_path: Path) -> None:
    spec = _fixture(tmp_path)
    outputs = {}
    for name, record in (("rebuilt", False), ("recorded", True)):
        path = tmp_path / f"spec-{name}.json"
        path.write_text(json.dumps({**spec, "record_pools": record}))
        run = subprocess.run(
            [
                sys.executable,
                "-m",
                "hyperdime.baselines.l2s_official",
                str(path),
                str(tmp_path / name),
            ],
            capture_output=True,
            text=True,
        )
        assert run.returncode == 0, run.stderr[-2000:]
        outputs[name] = tmp_path / name
    a, b = (torch.load(outputs[n] / "state_dict.pt") for n in ("rebuilt", "recorded"))
    assert all(torch.equal(a[k], b[k]) for k in a)
    for file in ("negatives.npy", "fallback.npy"):
        assert np.array_equal(
            np.load(outputs["rebuilt"] / file), np.load(outputs["recorded"] / file)
        )
