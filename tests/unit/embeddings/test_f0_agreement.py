"""The SC-004 check script, on tiny caches built here."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from hyperdime.contracts.config import ModelConfig
from hyperdime.contracts.manifests import ArtifactWriter, start_manifest
from hyperdime.embeddings.cache import embed_cached, embed_sharded, publish_embeddings

from conftest import StubEncoder

SCRIPT = Path(__file__).parents[3] / "scripts" / "checks" / "f0_agreement.py"
MODEL = ModelConfig(
    name="Qwen/Qwen3-Embedding-0.6B",
    revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    embedding_dim=1024,
    frozen=True,
    normalize=True,
    max_length=512,
    dtype="bfloat16",
)


def _script():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("f0_agreement", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def caches(tmp_path: Path) -> tuple[Path, Path]:
    """A spike-format cache and an F2 artifact holding the same vectors."""
    ids = ["a", "b", "c"]
    encoder = StubEncoder()
    for name in ("corpus", "queries"):
        embed_cached(ids, ids, encoder.encode, tmp_path / "spike" / name)
    parts = embed_sharded(ids, iter(ids), encoder.encode, tmp_path / "work", {}, 2)
    with ArtifactWriter(tmp_path, "embeddings/toy", start_manifest("embed", {}, seed=0)) as writer:
        publish_embeddings(writer, "documents", ids, parts, "documents", "toy", None, MODEL)
        publish_embeddings(writer, "queries", ids, parts, "queries", "toy", "x", MODEL)
        return tmp_path / "spike", writer.commit()


def test_identical_vectors_pass(caches: tuple[Path, Path]) -> None:
    assert _script().main([str(caches[0]), str(caches[1])]) == 0


def test_different_vectors_fail(caches: tuple[Path, Path]) -> None:
    corpus = caches[0] / "corpus" / "embeddings.npy"
    np.save(corpus, -np.load(corpus))
    assert _script().main([str(caches[0]), str(caches[1])]) == 1
