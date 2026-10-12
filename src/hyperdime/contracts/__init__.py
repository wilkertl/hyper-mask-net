"""M0: serializable schemas, run manifests, and validation shared by every module."""

from hyperdime.contracts.hashing import (
    NO_DATASET,
    canonical_json,
    config_hash,
    file_sha256,
    tree_sha256,
)
from hyperdime.contracts.schemas import (
    DocumentRecord,
    EmbeddingArtifact,
    EvaluationResult,
    PayloadRef,
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
from hyperdime.contracts.validation import ConfigError, ContractError

__all__ = [
    "NO_DATASET",
    "ConfigError",
    "ContractError",
    "DocumentRecord",
    "EmbeddingArtifact",
    "EvaluationResult",
    "PayloadRef",
    "QrelsRecord",
    "QueryRecord",
    "Record",
    "SelectorCheckpoint",
    "SpaceDescriptor",
    "canonical_json",
    "config_hash",
    "dump_record",
    "dump_records",
    "file_sha256",
    "load_record",
    "load_records",
    "tree_sha256",
]
