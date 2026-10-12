"""Publishing and loading embedding artifacts, and the settings guard (F2 US2)."""

from pathlib import Path

import numpy as np
import pytest

from hyperdime.contracts.config import ModelConfig
from hyperdime.contracts.hashing import file_sha256
from hyperdime.contracts.manifests import ArtifactWriter, start_manifest
from hyperdime.contracts.validation import ContractError
from hyperdime.embeddings.cache import (
    check_settings,
    embed_sharded,
    load_embeddings,
    publish_embeddings,
)
from hyperdime.embeddings.encoders import encoder_settings

from conftest import StubEncoder

MODEL = ModelConfig(
    name="Qwen/Qwen3-Embedding-0.6B",
    revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    embedding_dim=1024,
    frozen=True,
    normalize=True,
    max_length=512,
    dtype="bfloat16",
)
IDS = [f"q{i}" for i in range(5)]


def _publish(root: Path) -> Path:
    settings = encoder_settings(MODEL, "local", "queries", "Find it", compute_dtype="float32")
    parts = embed_sharded(IDS, iter(IDS), StubEncoder().encode, root / "work", settings, 2)
    with ArtifactWriter(root, "embeddings/toy", start_manifest("embed", {}, seed=0)) as writer:
        publish_embeddings(writer, "queries", IDS, parts, "queries", "toy", "Find it", MODEL)
        return writer.commit()


def test_published_vectors_and_record(tmp_path: Path) -> None:
    directory = _publish(tmp_path)
    record, vectors = load_embeddings(directory, "queries")
    assert record.ids == IDS and record.num_rows == 5 and record.dim == 1024
    assert record.role == "queries" and record.instruction == "Find it"
    assert record.model_revision == MODEL.revision and record.dtype == "float16"
    assert record.payload.sha256 == file_sha256(directory / "queries.npy")
    assert isinstance(vectors, np.memmap) and vectors.shape == (5, 1024)
    assert np.allclose(np.linalg.norm(vectors.astype(np.float32), axis=1), 1.0, atol=1e-2)


def test_settings_guard_names_the_differing_keys() -> None:
    stored = encoder_settings(MODEL, "local", "queries", "Find it", compute_dtype="bfloat16")
    check_settings(stored, dict(stored))
    other = encoder_settings(MODEL, "vllm", "queries", "Other", compute_dtype="bfloat16")
    with pytest.raises(ContractError, match="backend.*instruction|instruction.*backend"):
        check_settings(stored, other)


def test_document_settings_carry_no_instruction() -> None:
    documents = encoder_settings(MODEL, "local", "documents", None, compute_dtype="bfloat16")
    queries = encoder_settings(MODEL, "local", "queries", "Find it", compute_dtype="bfloat16")
    assert "instruction" not in documents and queries["instruction"] == "Find it"
    assert documents["revision"] == MODEL.revision and documents["max_length"] == 512
    assert documents["storage_dtype"] == "float16" and documents["backend"] == "local"
