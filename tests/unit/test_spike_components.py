import hashlib
import io
import json
import math
import zipfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import torch

from hyperdime.contracts.hashing import file_sha256, tree_sha256
from hyperdime.data import loaders
from hyperdime.data.loaders import (
    BeirDataset,
    beir_fingerprint,
    download_beir,
    judged_queries,
    read_beir,
)
from hyperdime.embeddings.cache import embed_cached
from hyperdime.evaluation.selector_shift import (
    js_divergence,
    mean_importance,
    overlap_at_k,
    spearman,
)
from hyperdime.evaluation.statistics import holm, paired_bootstrap_ci, paired_randomization_test
from hyperdime.oracle.targets import aggregate_targets
from hyperdime.retrieval.scoring import rank_documents
from hyperdime.spike import pipeline
from hyperdime.spike.environments import EnvSpec, make_splits, sample_corpus
from hyperdime.training.selector_trainer import TrainConfig, train_selector


def test_read_beir_parses_corpus_queries_and_qrels(tmp_path: Path) -> None:
    (tmp_path / "qrels").mkdir()
    (tmp_path / "corpus.jsonl").write_text(
        '{"_id": "d1", "title": "T", "text": "body"}\n{"_id": "d2", "title": "", "text": "only"}\n'
    )
    (tmp_path / "queries.jsonl").write_text(
        '{"_id": "q1", "text": "a"}\n{"_id": "q2", "text": "b"}\n'
    )
    (tmp_path / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t2\nq2\td2\t0\n"
    )
    dataset = read_beir(tmp_path)
    assert dataset.corpus == {"d1": "T body", "d2": "only"}
    assert dataset.qrels == {"test": {"q1": {"d1": 2}, "q2": {"d2": 0}}}
    assert judged_queries(dataset.qrels["test"], dataset.queries) == ["q1"]


def test_aggregate_targets_match_manual_computation() -> None:
    query = torch.tensor([[1.0, 2.0]])
    documents = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    # p = (1*[1,0] + 3*[0,1]) / 4 = [0.25, 0.75]; n = mean([1,1], [0,0]) = [0.5, 0.5]
    # r = q * (p - n) = [-0.25, 0.5]
    target = aggregate_targets(query, documents, [{0: 1, 1: 3}], [[2, 3]], temperature=1.0)
    expected = torch.softmax(torch.tensor([[-0.25, 0.5]]), dim=-1)
    assert torch.allclose(target, expected)


def test_aggregate_targets_ignore_non_positive_judgments_and_require_positives() -> None:
    documents = torch.eye(3)
    with pytest.raises(ValueError):
        aggregate_targets(torch.ones(1, 3), documents, [{0: 0}], [[1]], temperature=1.0)


def test_train_selector_is_deterministic_and_reduces_validation_kl() -> None:
    torch.manual_seed(0)
    mapping = torch.randn(16, 16)
    queries = torch.randn(64, 16)
    targets = torch.softmax(queries @ mapping, dim=-1)
    config = TrainConfig(lr=1e-2, max_epochs=40, patience=40, batch_size=16, seed=3)
    model_a, history = train_selector(
        queries[:48], targets[:48], queries[48:], targets[48:], config
    )
    model_b, _ = train_selector(queries[:48], targets[:48], queries[48:], targets[48:], config)
    assert history[-1]["val_kl"] < history[0]["val_kl"]
    assert torch.equal(model_a.projection.weight, model_b.projection.weight)


def test_holm_matches_manual_adjustment() -> None:
    # sorted: a 0.01*3 = 0.03, c 0.03*2 = 0.06, b max(0.06, 0.04*1) = 0.06
    assert holm({"a": 0.01, "b": 0.04, "c": 0.03}) == pytest.approx(
        {"a": 0.03, "c": 0.06, "b": 0.06}
    )


def test_randomization_test_and_bootstrap_behave_at_the_extremes() -> None:
    same = [0.3, 0.5, 0.7, 0.2]
    assert paired_randomization_test(same, same, n_permutations=999) == 1.0
    a, b = [1.0] * 30, [0.0] * 30
    assert paired_randomization_test(a, b, n_permutations=999) == pytest.approx(1 / 1000)
    assert paired_bootstrap_ci(a, b, n_resamples=500) == (1.0, 1.0)


def test_selector_shift_measures_match_manual_values() -> None:
    mask_a = torch.tensor([[True, True, False, False]])
    mask_b = torch.tensor([[True, False, True, False]])
    assert overlap_at_k(mask_a, mask_b).tolist() == [0.5]
    p, q = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
    assert js_divergence(p, p) == pytest.approx(0.0)
    assert js_divergence(p, q) == pytest.approx(math.log(2))
    ranks = torch.tensor([0.1, 0.4, 0.2, 0.9])
    assert spearman(ranks, ranks * 2) == pytest.approx(1.0)
    assert spearman(ranks, -ranks) == pytest.approx(-1.0)
    assert mean_importance(torch.log_softmax(torch.randn(5, 4), dim=-1)).sum() == pytest.approx(1.0)


def test_rank_documents_skips_the_document_with_the_query_id() -> None:
    documents = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    rankings = rank_documents(
        torch.tensor([[1.0, 0.0]]), documents, ["a"], ["a", "b", "c"], depth=2
    )
    assert rankings == {"a": ["b", "c"]}


def test_embed_cached_resumes_from_saved_chunks(tmp_path: Path) -> None:
    calls: list[int] = []

    def encode(texts: Sequence[str]) -> torch.Tensor:
        calls.append(len(texts))
        return torch.tensor([[float(len(t))] * 3 for t in texts])

    ids, texts = ["a", "b", "c"], ["x", "yy", "zzz"]
    np.save(tmp_path / "part-00000.npy", np.full((2, 3), 7.0, dtype=np.float16))
    loaded_ids, embeddings = embed_cached(ids, texts, encode, tmp_path, chunk_size=2)
    assert calls == [1]
    assert loaded_ids == ids
    assert embeddings[:, 0].tolist() == [7.0, 7.0, 3.0]
    assert not list(tmp_path.glob("part-*.npy"))
    assert embed_cached(ids, texts, encode, tmp_path, chunk_size=2)[1].shape == (3, 3)
    assert calls == [1]
    assert json.loads((tmp_path / "ids.json").read_text()) == ids


def _dataset() -> BeirDataset:
    corpus = {f"d{i}": "text" for i in range(10)}
    queries = {f"q{i}": "query" for i in range(40)}
    train = {f"q{i}": {f"d{i % 10}": 1} for i in range(20)}
    test = {f"q{i}": {f"d{i % 10}": 1} for i in range(20, 40)}
    test["q39"] = {"missing-doc": 1}
    return BeirDataset("toy", corpus, queries, {"train": train, "test": test})


def test_make_splits_keeps_evaluation_labels_out_of_training() -> None:
    splits = make_splits(EnvSpec("toy", "", "train"), _dataset())
    assert len(splits.train) + len(splits.validation) == 20
    assert "q39" not in splits.evaluation  # its only positive is not in the corpus
    assert not set(splits.train_labels) & set(splits.evaluation)
    assert set(splits.eval_labels) == set(splits.evaluation)


def test_make_splits_halves_test_only_datasets_deterministically() -> None:
    spec = EnvSpec("toy", "", None)
    first, second = make_splits(spec, _dataset()), make_splits(spec, _dataset())
    assert first == second
    pool = set(first.train) | set(first.validation)
    assert pool.isdisjoint(first.evaluation)
    assert len(pool) + len(first.evaluation) == 19
    assert set(first.train_labels) == pool


def test_sample_corpus_keeps_judged_documents_and_fills_deterministically() -> None:
    dataset = _dataset()
    corpus = {f"d{i}": "text" for i in range(100)}
    dataset = BeirDataset("toy", corpus, dataset.queries, dataset.qrels)
    splits = make_splits(EnvSpec("toy", "", "train"), dataset)
    sample = sample_corpus(dataset, splits, cap=15)
    assert len(sample) == 15 and len(set(sample)) == 15
    assert {f"d{i}" for i in range(10)} <= set(sample)
    assert sample == sample_corpus(dataset, splits, cap=15)
    assert len(sample_corpus(dataset, splits, cap=0)) == 100


def test_embed_cached_refuses_to_mix_settings(tmp_path: Path) -> None:
    def encode(texts: Sequence[str]) -> torch.Tensor:
        return torch.ones(len(texts), 2)

    embed_cached(["a"], ["x"], encode, tmp_path, settings={"max_length": 256})
    assert embed_cached(["a"], ["x"], encode, tmp_path, settings={"max_length": 256})[0] == ["a"]
    with pytest.raises(ValueError, match="delete it"):
        embed_cached(["a"], ["x"], encode, tmp_path, settings={"max_length": 512})


SHA256_HI = "8f434346648f6b96df89dda901c5176b10a6d83961dd3c1ac88b59b2dc327aa4"  # sha256(b"hi")


def test_file_and_tree_hashes_match_manual_values(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "x.txt").write_bytes(b"hi")
    assert file_sha256(tmp_path / "sub" / "x.txt") == SHA256_HI
    expected = hashlib.sha256(f"sub/x.txt\t{SHA256_HI}\n".encode()).hexdigest()
    assert tree_sha256(tmp_path) == expected


def test_download_beir_records_archive_and_tree_fingerprints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr("toy/corpus.jsonl", '{"_id": "d1", "text": "x"}\n')
        bundle.writestr("toy/queries.jsonl", '{"_id": "q1", "text": "y"}\n')
    archive = buffer.getvalue()
    monkeypatch.setattr(loaders.urllib.request, "urlopen", lambda url: io.BytesIO(archive))
    directory = download_beir("toy", tmp_path)
    record = beir_fingerprint(tmp_path, "toy")
    assert record["zip_sha256"] == hashlib.sha256(archive).hexdigest()
    assert record["tree_sha256"] == tree_sha256(directory)
    assert record["url"].endswith("/toy.zip")

    def offline(url: str) -> io.BytesIO:
        raise AssertionError("an extracted dataset must not be downloaded again")

    monkeypatch.setattr(loaders.urllib.request, "urlopen", offline)
    assert download_beir("toy", tmp_path) == directory
    (tmp_path / "toy.sha256.json").unlink()
    assert beir_fingerprint(tmp_path, "toy")["zip_sha256"] == "unknown"


def test_cache_manifest_fingerprints_the_published_vectors(tmp_path: Path) -> None:
    def encode(texts: Sequence[str]) -> torch.Tensor:
        return torch.ones(len(texts), 2)

    embed_cached(["a", "b"], ["x", "y"], encode, tmp_path, settings={"max_length": 8})
    pipeline._cache_manifest(tmp_path, {"max_length": 8}, 2)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["command"] == "embed"
    assert manifest["config"]["rows"] == 2
    assert manifest["config"]["embeddings_sha256"] == file_sha256(tmp_path / "embeddings.npy")
