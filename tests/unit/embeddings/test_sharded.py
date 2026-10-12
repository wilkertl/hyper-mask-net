"""Sharded, resumable embedding (F2 US2, research D5)."""

from pathlib import Path

import numpy as np
import pytest

from hyperdime.contracts.validation import ContractError
from hyperdime.embeddings.cache import embed_sharded

from conftest import StubEncoder

SETTINGS = {"revision": "r1", "max_length": 512}
IDS = [f"d{i}" for i in range(10)]
TEXTS = [f"text {i}" for i in range(10)]


def _stack(parts: list[Path]) -> np.ndarray:
    return np.concatenate([np.load(p) for p in parts])


def test_shards_are_written_in_order(tmp_path: Path) -> None:
    encoder = StubEncoder()
    parts = embed_sharded(IDS, iter(TEXTS), encoder.encode, tmp_path, SETTINGS, shard_size=4)
    assert [p.name for p in parts] == ["part-00000.npy", "part-00001.npy", "part-00002.npy"]
    vectors = _stack(parts)
    assert vectors.dtype == np.float16 and vectors.shape == (10, 1024)
    assert np.allclose(vectors[3], encoder.encode([TEXTS[3]]).numpy(), atol=1e-3)


def test_an_interrupted_run_resumes_after_its_last_shard(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="interrupted"):
        embed_sharded(
            IDS, iter(TEXTS), StubEncoder(fail_after=2).encode, tmp_path / "a", SETTINGS, 4
        )
    assert sorted(p.name for p in (tmp_path / "a").glob("part-*.npy")) == [
        "part-00000.npy",
        "part-00001.npy",
    ]
    resumed = StubEncoder()
    parts = embed_sharded(IDS, iter(TEXTS), resumed.encode, tmp_path / "a", SETTINGS, 4)
    assert resumed.encoded == TEXTS[8:]  # only the missing shard is encoded
    clean = embed_sharded(IDS, iter(TEXTS), StubEncoder().encode, tmp_path / "b", SETTINGS, 4)
    assert np.array_equal(_stack(parts), _stack(clean))


def test_a_work_directory_with_other_settings_is_refused(tmp_path: Path) -> None:
    embed_sharded(IDS, iter(TEXTS), StubEncoder().encode, tmp_path, SETTINGS, 4)
    with pytest.raises(ContractError, match="max_length"):
        embed_sharded(
            IDS, iter(TEXTS), StubEncoder().encode, tmp_path, {**SETTINGS, "max_length": 256}, 4
        )


def test_a_work_directory_for_other_ids_is_refused(tmp_path: Path) -> None:
    embed_sharded(IDS, iter(TEXTS), StubEncoder().encode, tmp_path, SETTINGS, 4)
    with pytest.raises(ContractError, match="ids"):
        embed_sharded(IDS[::-1], iter(TEXTS), StubEncoder().encode, tmp_path, SETTINGS, 4)


def test_texts_must_match_the_ids(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="texts"):
        embed_sharded(IDS, iter(TEXTS[:7]), StubEncoder().encode, tmp_path, SETTINGS, 4)
