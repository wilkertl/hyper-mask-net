"""The SC-004 check script, on tiny caches built here."""

import importlib.util
from pathlib import Path

import numpy as np

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


def test_identical_vectors_pass_and_different_ones_fail(tmp_path: Path) -> None:
    ids = ["a", "b", "c"]
    encoder = StubEncoder()
    for name in ("corpus", "queries"):
        embed_cached(ids, ids, encoder.encode, tmp_path / "spike" / name)
    parts = embed_sharded(ids, iter(ids), encoder.encode, tmp_path / "work", {}, 2)
    with ArtifactWriter(tmp_path, "embeddings/toy", start_manifest("embed", {}, seed=0)) as writer:
        publish_embeddings(writer, "documents", ids, parts, "documents", "toy", None, MODEL)
        publish_embeddings(writer, "queries", ids, parts, "queries", "toy", "x", MODEL)
        published = writer.commit()
    script = _script()
    assert script.main([str(tmp_path / "spike"), str(published)]) == 0
    np.save(
        tmp_path / "spike" / "corpus" / "embeddings.npy",
        -np.load(tmp_path / "spike" / "corpus" / "embeddings.npy"),
    )
    assert script.main([str(tmp_path / "spike"), str(published)]) == 1
