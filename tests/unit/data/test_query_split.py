"""F12 query split: frozen hash, disjoint sorted ids, normalization, hash order (T004, T014a)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperdime.contracts.hashing import file_sha256
from hyperdime.contracts.validation import ContractError
from hyperdime.data.query_split import (
    SPLIT_MANIFEST,
    hash_order,
    id_fingerprint,
    load_query_split,
    normalize,
    verify_against_source,
)


def _write(root: Path, ids: dict[str, list[str]], excluded: list[str] | None = None) -> str:
    path = root / SPLIT_MANIFEST
    path.parent.mkdir(parents=True)
    corpus = {
        "source_dataset_run": "run",
        "source_queries_sha256": "a" * 64,
        "excluded_test_overlap_ids": excluded or [],
        "counts": {name: len(values) for name, values in ids.items()},
        "query_ids": ids,
    }
    path.write_text(json.dumps({"seed": 20261112, "corpora": {"c": corpus}}))
    return file_sha256(path)


def test_normalize_applies_nfkc_casefold_and_whitespace() -> None:
    assert normalize("  Ｆｏｏ\tBAR\n") == "foo bar"
    assert normalize("Straße") == normalize("STRASSE")


def test_fingerprint_ignores_order() -> None:
    assert id_fingerprint(["b", "a"]) == id_fingerprint(["a", "b"])


def test_hash_order_is_a_fixed_permutation_whose_prefixes_nest() -> None:
    ids = [f"q{i}" for i in range(50)]
    order = hash_order("c", "fit", ids)
    assert sorted(order) == sorted(ids) and order == hash_order("c", "fit", reversed(ids))
    assert order != hash_order("c", "tune", ids)


def test_load_refuses_a_changed_file(tmp_path: Path) -> None:
    _write(tmp_path, {"fit": ["a"], "tune": ["b"], "evaluation": ["c"]})
    with pytest.raises(ContractError, match="SHA-256"):
        load_query_split(tmp_path, "c")


def test_load_refuses_overlapping_splits(tmp_path: Path) -> None:
    digest = _write(tmp_path, {"fit": ["a"], "tune": ["a"], "evaluation": ["c"]})
    with pytest.raises(ContractError, match="overlap"):
        load_query_split(tmp_path, "c", expected_sha256=digest)


def test_verify_catches_groups_across_splits_and_test_text(tmp_path: Path) -> None:
    digest = _write(tmp_path, {"fit": ["a"], "tune": ["b"], "evaluation": ["c"]}, excluded=["d"])
    split = load_query_split(tmp_path, "c", expected_sha256=digest)
    records = [
        {"query_id": "a", "text": "Same claim", "split": "train"},
        {"query_id": "b", "text": "same  CLAIM", "split": "train"},
        {"query_id": "c", "text": "other", "split": "validation"},
        {"query_id": "d", "text": "held", "split": "train"},
        {"query_id": "t", "text": "HELD", "split": "test"},
    ]
    report = verify_against_source(split, records, "a" * 64)
    assert not report["checks"]["no_group_crosses_splits"]
    assert report["checks"]["dropped_equals_excluded"] and report["checks"]["dropped_match_test"]
    records[1]["text"] = "different"
    assert verify_against_source(split, records, "a" * 64)["passed"]


def test_load_never_exposes_official_test_ids_or_labels(tmp_path: Path) -> None:
    """A manifest carrying poisoned ``test`` entries still loads, and they never appear (FR-002)."""
    path = tmp_path / SPLIT_MANIFEST
    path.parent.mkdir(parents=True)
    ids = {"fit": ["a"], "tune": ["b"], "evaluation": ["c"]}
    corpus = {
        "source_dataset_run": "run",
        "source_queries_sha256": "a" * 64,
        "excluded_test_overlap_ids": [],
        "counts": {**{k: len(v) for k, v in ids.items()}, "test": 1},
        "query_ids": {**ids, "test": ["SECRET-TEST-ID"]},
        "test_qrels": {"SECRET-TEST-ID": {"doc": 1}},
    }
    path.write_text(json.dumps({"seed": 20261112, "corpora": {"c": corpus}}))
    split = load_query_split(path.parent.parent, "c", expected_sha256=file_sha256(path))
    assert set(split.ids) == {"fit", "tune", "evaluation"}
    assert "SECRET-TEST-ID" not in repr(split) and "SECRET-TEST-ID" not in str(split.fingerprints())
