"""Streamed exact search equals full exact search (F2 research D6)."""

import numpy as np
import pytest
import torch

from hyperdime.retrieval.exact import exact_search, exact_search_streamed


@pytest.mark.parametrize(("chunk_size", "query_batch"), [(300, 64), (64, 7), (5, 3), (1, 50)])
def test_streamed_matches_full_search(chunk_size: int, query_batch: int) -> None:
    generator = torch.Generator().manual_seed(0)
    documents = torch.randn(300, 16, generator=generator)
    queries = torch.randn(20, 16, generator=generator)
    _, expected = exact_search(queries, documents, 25)
    found = exact_search_streamed(
        queries, documents.numpy(), 25, chunk_size=chunk_size, query_batch=query_batch
    )
    assert torch.equal(found, expected)


def test_streamed_search_reads_a_memory_map(tmp_path) -> None:  # type: ignore[no-untyped-def]
    raw = np.random.default_rng(1).standard_normal((50, 8))
    documents = (raw / np.linalg.norm(raw, axis=1, keepdims=True)).astype(np.float16)
    np.save(tmp_path / "d.npy", documents)
    mapped = np.load(tmp_path / "d.npy", mmap_mode="r")
    queries = torch.from_numpy(documents[:3].astype(np.float32))
    found = exact_search_streamed(queries, mapped, 1, chunk_size=7)
    assert found[:, 0].tolist() == [0, 1, 2]  # each unit vector is its own best match


def test_depth_is_validated() -> None:
    with pytest.raises(ValueError, match="depth"):
        exact_search_streamed(torch.zeros(1, 4), np.zeros((3, 4), dtype=np.float16), 4)
