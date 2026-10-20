"""Versioned, serializable records shared by every module (data-model.md, research R1, R10).

Records are immutable, reject unknown fields, and carry a ``schema_version`` that loading
checks. Large numeric payloads are referenced by ``PayloadRef``, never embedded.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, Self, TypeVar

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from hyperdime.contracts.validation import ContractError, format_validation_error
from hyperdime.embeddings.qwen import EMBEDDING_DIM, MODEL_NAME

# The literals below must match the frozen encoder; checked at import time.


def _no_control_characters(value: str) -> str:
    if any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError("must not contain control characters")
    return value


def _relative_posix(value: str) -> str:
    path = PurePosixPath(value)
    if "\\" in value or path.is_absolute() or ".." in path.parts:
        raise ValueError("must be a relative POSIX path without '..'")
    return value


def _finite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("must be finite")
    return value


Id = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
    AfterValidator(_no_control_characters),
]
Split = Literal["train", "validation", "test"]
Protocol = Literal["P0", "P1", "P2", "in-domain", "transfer"]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Dtype = Literal["float16", "bfloat16", "float32"]
Finite = Annotated[float, Field(allow_inf_nan=False), AfterValidator(_finite)]
Probability = Annotated[float, Field(ge=0, le=1)]
Budget = Annotated[int, Field(ge=1, le=EMBEDDING_DIM)]
DescriptorGroup = Literal["A", "B", "C", "D", "E"]

# Descriptor groups each protocol may use (research R11); P0 sees documents only.
PROTOCOL_GROUPS: dict[str, frozenset[str]] = {
    "P0": frozenset({"A", "E"}),
    "P1": frozenset({"A", "B", "C", "D", "E"}),
    "P2": frozenset({"A", "B", "C", "D", "E"}),
}


class PayloadRef(BaseModel):
    """A numeric payload stored next to its record: relative path, SHA-256, and format."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: Annotated[str, StringConstraints(min_length=1), AfterValidator(_relative_posix)]
    sha256: Sha256
    format: Literal["npy", "safetensors"]


class Record(BaseModel):
    """Base of every record; subclasses fix ``schema_version`` with a ``Literal``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int


class QueryRecord(Record):
    schema_version: Literal[1] = 1
    query_id: Id
    text: Annotated[str, StringConstraints(min_length=1)]
    instruction: str | None
    environment: Id
    split: Split


class DocumentRecord(Record):
    schema_version: Literal[1] = 1
    doc_id: Id
    title: str
    text: str
    environment: Id

    @model_validator(mode="after")
    def _has_content(self) -> Self:
        if not (self.title + self.text).strip():
            raise ValueError("text: must be non-empty after joining with title")
        return self


class QrelsRecord(Record):
    schema_version: Literal[1] = 1
    query_id: Id
    doc_id: Id
    relevance: Annotated[int, Field(ge=0)]
    environment: Id
    split: Split


class EmbeddingArtifact(Record):
    schema_version: Literal[1] = 1
    role: Literal["queries", "documents"]
    environment: Id
    ids: list[Id]
    num_rows: Annotated[int, Field(ge=1)]
    dim: Literal[1024] = 1024
    dtype: Dtype
    normalized: bool
    pooling: Literal["last_token"] = "last_token"
    model_name: Literal["Qwen/Qwen3-Embedding-0.6B"] = "Qwen/Qwen3-Embedding-0.6B"
    model_revision: Annotated[str, StringConstraints(min_length=1)]
    instruction: str | None
    payload: PayloadRef

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("ids: must be distinct")
        if len(self.ids) != self.num_rows:
            raise ValueError(f"num_rows: {self.num_rows} != len(ids) = {len(self.ids)}")
        if self.role == "documents" and self.instruction is not None:
            raise ValueError("instruction: must be None for documents")
        return self


class SpaceDescriptor(Record):
    """Per-dimension environment statistics; provisional until the descriptor freezes (M5.5)."""

    schema_version: Literal[0] = 0
    environment: Id
    protocol: Literal["P0", "P1", "P2"]
    feature_names: Annotated[list[str], Field(min_length=1)]
    groups: Annotated[list[DescriptorGroup], Field(min_length=1)]
    num_documents: Annotated[int, Field(ge=1)]
    num_queries: Annotated[int, Field(ge=0)]
    payload: PayloadRef

    @model_validator(mode="after")
    def _protocol_rules(self) -> Self:
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names: must be distinct")
        allowed = PROTOCOL_GROUPS[self.protocol]
        if not set(self.groups) <= allowed:
            raise ValueError(f"groups: {self.protocol} allows only {sorted(allowed)}")
        if self.protocol == "P0" and self.num_queries != 0:
            raise ValueError("num_queries: must be 0 under P0 (documents only)")
        return self


class SelectorCheckpoint(Record):
    """A trained or generated selector; provisional until the low-rank selector (M7)."""

    schema_version: Literal[0] = 0
    kind: Literal["linear", "global", "per_domain", "generated"]
    environment: Id | None
    dim: Literal[1024] = 1024
    rank: Annotated[int, Field(ge=1)] | None = None
    payload: PayloadRef
    training_manifest_sha256: Sha256

    @model_validator(mode="after")
    def _kind_rules(self) -> Self:
        if (self.kind == "global") != (self.environment is None):
            raise ValueError("environment: must be None exactly when kind is 'global'")
        if self.rank is not None and self.kind != "generated":
            raise ValueError("rank: only generated selectors have a rank")
        if self.payload.format != "safetensors":
            raise ValueError("payload: selectors are stored as safetensors")
        return self


class EvaluationResult(Record):
    """Exact-scoring metrics with per-query values for paired tests (constitution V)."""

    schema_version: Literal[1] = 1
    environment: Id
    protocol: Protocol
    split: Split
    scoring: Literal["exact"] = "exact"
    selector: Annotated[str, StringConstraints(min_length=1)]
    k: Annotated[int, Field(ge=1, le=EMBEDDING_DIM)]
    metrics: dict[str, Finite]
    per_query: dict[Id, dict[str, Finite]]
    num_queries: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.num_queries != len(self.per_query):
            raise ValueError(f"num_queries: {self.num_queries} != {len(self.per_query)} queries")
        keys = set(self.metrics)
        for query_id, values in self.per_query.items():
            if set(values) != keys:
                raise ValueError(f"per_query: {query_id} has metrics {sorted(values)}")
        return self


class SelectorProperties(Record):
    """How one seeded selector weights and selects dimensions on the shared query battery (F4)."""

    schema_version: Literal[0] = 0
    environment: Id
    seed: Annotated[int, Field(ge=0)]
    selector_sha256: Sha256
    battery_sha256: Sha256
    k_values: Annotated[list[Budget], Field(min_length=1)]
    mean_importance: PayloadRef
    selection_frequency: PayloadRef


class TransferCell(Record):
    """One ordered (train, test) environment pair on one query set, seeds averaged (F4 D4)."""

    schema_version: Literal[0] = 0
    train_environment: Id
    test_environment: Id
    query_set: Annotated[str, StringConstraints(min_length=1)]
    seeds: Annotated[list[Annotated[int, Field(ge=0)]], Field(min_length=1)]
    k: Budget
    metrics: dict[str, Finite]
    in_domain_ndcg10: Finite
    degradation: Finite
    p_value: Probability | None
    p_holm: Probability | None
    ci_low: Finite | None
    ci_high: Finite | None
    overlap_at_k: Probability
    results: Annotated[list[str], Field(min_length=1)]


class Criterion(BaseModel):
    """One gate criterion with its computed value and whether it holds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Annotated[str, StringConstraints(min_length=1)]
    value: Finite
    threshold: Annotated[str, StringConstraints(min_length=1)]
    met: bool


class GateDecision(Record):
    """A drafted gate (or experiment, e.g. S1) outcome from result files; the researcher decides
    in the ADR."""

    schema_version: Literal[0] = 0
    gate: Literal["G1", "G2", "G3", "G4", "G5", "S1", "S2", "S3", "S4"]
    outcome: Literal["pass", "fail"]
    criteria: Annotated[list[Criterion], Field(min_length=1)]
    context: dict[str, str]
    evidence: Annotated[list[str], Field(min_length=1)]
    adr: Annotated[str, StringConstraints(min_length=1)]
    tag: Annotated[str, StringConstraints(min_length=1)]


class DatasetManifest(Record):
    """What a prepared environment contains and where it came from (F2 data model)."""

    schema_version: Literal[1] = 1
    environment: Id
    spec: dict[str, Any]
    num_documents: Annotated[int, Field(ge=1)]
    dropped_documents: Annotated[int, Field(ge=0)]
    num_queries: dict[str, Annotated[int, Field(ge=0)]]
    num_qrels: dict[str, Annotated[int, Field(ge=0)]]
    dropped_qrels: Annotated[int, Field(ge=0)]
    excluded_queries: dict[str, Annotated[int, Field(ge=0)]]
    split_seed: Annotated[int, Field(ge=0)]
    validation_fraction: Annotated[float, Field(gt=0, lt=1)]
    files: dict[str, Sha256]
    fingerprint: Sha256


class NegativePool(Record):
    """Sampled hard negatives of one training or validation query (ADR-0002)."""

    schema_version: Literal[1] = 1
    query_id: Id
    environment: Id
    split: Literal["train", "validation"]
    doc_ids: list[Id]
    depth: Annotated[int, Field(ge=1)]
    pool_size: Annotated[int, Field(ge=1)]
    seed: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def _pool_size(self) -> Self:
        if len(self.doc_ids) != self.pool_size:
            raise ValueError(f"doc_ids: {len(self.doc_ids)} entries != pool_size {self.pool_size}")
        if self.query_id in self.doc_ids:
            raise ValueError("doc_ids: must not contain the query's own id")
        return self


if EMBEDDING_DIM != 1024 or MODEL_NAME != "Qwen/Qwen3-Embedding-0.6B":
    raise ImportError("record literals no longer match hyperdime.embeddings.qwen")

R = TypeVar("R", bound=Record)


def dump_record(record: Record, path: Path) -> None:
    """Write one record as JSON (UTF-8, trailing newline)."""
    path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_record(cls: type[R], path: Path) -> R:
    """Read one record; any invalid content raises ``ContractError`` naming the field."""
    try:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as error:
        raise ContractError(format_validation_error(cls.__name__, error)) from error


def dump_records(records: Iterable[Record], path: Path) -> None:
    """Write records as JSON Lines."""
    lines = [record.model_dump_json() + "\n" for record in records]
    path.write_text("".join(lines), encoding="utf-8")


def load_records(cls: type[R], path: Path) -> list[R]:
    """Read JSON Lines; errors name the line number as well as the field."""
    records: list[R] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(cls.model_validate_json(line))
        except ValidationError as error:
            raise ContractError(format_validation_error(cls.__name__, error, number)) from error
    return records
