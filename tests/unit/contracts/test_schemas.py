"""Versioned records: round trips, schema versions, and field rules (spec US1, data-model.md)."""

import json
from pathlib import Path
from typing import Any

import pytest

from hyperdime.contracts.schemas import (
    DocumentRecord,
    EmbeddingArtifact,
    EvaluationResult,
    QrelsRecord,
    QueryRecord,
    Record,
    SelectorCheckpoint,
    SpaceDescriptor,
    dump_record,
    dump_records,
    load_record,
    load_records,
)
from hyperdime.contracts.validation import ContractError

SHA = "ab" * 32
NPY = {"path": "payload.npy", "sha256": SHA, "format": "npy"}
SAFETENSORS = {"path": "selector.safetensors", "sha256": SHA, "format": "safetensors"}

VALID: dict[type[Record], dict[str, Any]] = {
    QueryRecord: {
        "query_id": "q1",
        "text": "Does aspirin reduce risk?",
        "instruction": "Given a claim",
        "environment": "scifact",
        "split": "train",
    },
    DocumentRecord: {"doc_id": "d1", "title": "", "text": "Aspirin.", "environment": "scifact"},
    QrelsRecord: {
        "query_id": "q1",
        "doc_id": "d1",
        "relevance": 2,
        "environment": "scifact",
        "split": "test",
    },
    EmbeddingArtifact: {
        "role": "queries",
        "environment": "scifact",
        "ids": ["q1", "q2"],
        "num_rows": 2,
        "dim": 1024,
        "dtype": "float16",
        "normalized": True,
        "pooling": "last_token",
        "model_name": "Qwen/Qwen3-Embedding-0.6B",
        "model_revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "instruction": "Given a claim",
        "payload": NPY,
    },
    SpaceDescriptor: {
        "environment": "scifact",
        "protocol": "P0",
        "feature_names": ["mean", "std"],
        "groups": ["A", "E"],
        "num_documents": 100,
        "num_queries": 0,
        "payload": NPY,
    },
    SelectorCheckpoint: {
        "kind": "per_domain",
        "environment": "scifact",
        "dim": 1024,
        "rank": None,
        "payload": SAFETENSORS,
        "training_manifest_sha256": SHA,
    },
    EvaluationResult: {
        "environment": "scifact",
        "protocol": "in-domain",
        "split": "test",
        "scoring": "exact",
        "selector": "linear",
        "k": 307,
        "metrics": {"ndcg@10": 0.5},
        "per_query": {"q1": {"ndcg@10": 0.4}, "q2": {"ndcg@10": 0.6}},
        "num_queries": 2,
    },
}
RECORD_TYPES = list(VALID)


def _reject(tmp_path: Path, cls: type[Record], data: dict[str, Any], field: str) -> None:
    """Loading ``data`` as ``cls`` fails with a message naming the record type and the field."""
    path = tmp_path / "record.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ContractError) as error:
        load_record(cls, path)
    assert cls.__name__ in str(error.value)
    assert field in str(error.value)


@pytest.mark.parametrize("cls", RECORD_TYPES, ids=lambda c: c.__name__)
def test_record_round_trips_through_json(cls: type[Record], tmp_path: Path) -> None:
    record = cls(**VALID[cls])
    dump_record(record, tmp_path / "record.json")
    loaded = load_record(cls, tmp_path / "record.json")
    assert loaded == record
    assert loaded.schema_version == record.schema_version


@pytest.mark.parametrize("cls", RECORD_TYPES, ids=lambda c: c.__name__)
def test_records_round_trip_through_json_lines(cls: type[Record], tmp_path: Path) -> None:
    records = [cls(**VALID[cls]), cls(**VALID[cls])]
    dump_records(records, tmp_path / "records.jsonl")
    assert load_records(cls, tmp_path / "records.jsonl") == records


@pytest.mark.parametrize("cls", RECORD_TYPES, ids=lambda c: c.__name__)
def test_missing_wrong_and_unknown_fields_are_named(cls: type[Record], tmp_path: Path) -> None:
    first_field = next(iter(VALID[cls]))
    missing = {k: v for k, v in VALID[cls].items() if k != first_field}
    _reject(tmp_path, cls, missing, first_field)
    _reject(tmp_path, cls, {**VALID[cls], "environment": ["not", "a", "string"]}, "environment")
    _reject(tmp_path, cls, {**VALID[cls], "unexpected": 1}, "unexpected")


@pytest.mark.parametrize("cls", RECORD_TYPES, ids=lambda c: c.__name__)
def test_unsupported_schema_version_fails(cls: type[Record], tmp_path: Path) -> None:
    current = cls(**VALID[cls]).schema_version
    _reject(tmp_path, cls, {**VALID[cls], "schema_version": current + 1}, "schema_version")


def test_json_lines_errors_name_the_line(tmp_path: Path) -> None:
    good = json.dumps(VALID[QueryRecord])
    bad = json.dumps({**VALID[QueryRecord], "split": "holdout"})
    path = tmp_path / "queries.jsonl"
    path.write_text(f"{good}\n{good}\n{bad}\n")
    with pytest.raises(ContractError, match="line 3: QueryRecord.split"):
        load_records(QueryRecord, path)


@pytest.mark.parametrize(
    ("cls", "changes", "field"),
    [
        (EmbeddingArtifact, {"dim": 768}, "dim"),
        (EmbeddingArtifact, {"ids": ["q1", "q1"]}, "ids"),
        (EmbeddingArtifact, {"num_rows": 3}, "num_rows"),
        (EmbeddingArtifact, {"role": "documents"}, "instruction"),
        (QrelsRecord, {"relevance": -1}, "relevance"),
        (QueryRecord, {"query_id": ""}, "query_id"),
        (QueryRecord, {"query_id": "q\x07"}, "query_id"),
        (SpaceDescriptor, {"protocol": "in-domain"}, "protocol"),
        (SpaceDescriptor, {"groups": ["A", "B"]}, "groups"),
        (SpaceDescriptor, {"num_queries": 5}, "num_queries"),
        (SpaceDescriptor, {"feature_names": ["mean", "mean"]}, "feature_names"),
        (SelectorCheckpoint, {"kind": "global"}, "environment"),
        (SelectorCheckpoint, {"rank": 8}, "rank"),
        (SelectorCheckpoint, {"payload": NPY}, "payload"),
        (EvaluationResult, {"k": 0}, "k"),
        (EvaluationResult, {"k": 1025}, "k"),
        (EvaluationResult, {"num_queries": 3}, "num_queries"),
        (EvaluationResult, {"per_query": {"q1": {"mrr@10": 0.4}, "q2": {}}}, "per_query"),
        (EvaluationResult, {"scoring": "faiss"}, "scoring"),
    ],
)
def test_field_rules(
    cls: type[Record], changes: dict[str, Any], field: str, tmp_path: Path
) -> None:
    _reject(tmp_path, cls, {**VALID[cls], **changes}, field)


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({**NPY, "path": "/abs/payload.npy"}, "path"),
        ({**NPY, "path": "../payload.npy"}, "path"),
        ({**NPY, "path": "sub\\payload.npy"}, "path"),
        ({**NPY, "sha256": "AB" * 32}, "sha256"),
        ({**NPY, "format": "parquet"}, "format"),
    ],
)
def test_payload_reference_rules(payload: dict[str, str], field: str, tmp_path: Path) -> None:
    _reject(tmp_path, SpaceDescriptor, {**VALID[SpaceDescriptor], "payload": payload}, field)


def test_non_finite_metrics_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    data = {**VALID[EvaluationResult], "metrics": {"ndcg@10": "NaN"}}
    path.write_text(json.dumps(data))
    with pytest.raises(ContractError, match="EvaluationResult.metrics"):
        load_record(EvaluationResult, path)


def test_global_selector_has_no_environment_and_generated_has_a_rank(tmp_path: Path) -> None:
    base = VALID[SelectorCheckpoint]
    assert SelectorCheckpoint(**{**base, "kind": "global", "environment": None}).environment is None
    assert SelectorCheckpoint(**{**base, "kind": "generated", "rank": 8}).rank == 8
