"""Streamed group-A statistics: moments, tails, and quantiles by hand (F10 research D3)."""

import numpy as np
import pytest
import torch

from hyperdime.space.marginal import marginal_stats
from hyperdime.space.streaming import (
    document_moments,
    document_quantiles,
    quantile_rows,
    tail_share,
)

# Column 0: four 0s and a 5 (mean 1, m2 4, m3 12, m4 52); column 1: -2..2 (m2 2, m4 6.8);
# column 2: constant 3.
DOCS = np.array(
    [[0, -2, 3], [0, -1, 3], [0, 0, 3], [0, 1, 3], [5, 2, 3]], dtype=np.float16
)  # fmt: skip


def test_moments_by_hand_streamed_in_chunks_of_two() -> None:
    found = document_moments(DOCS, chunk_size=2)
    assert found.count == 5
    assert found.mean.tolist() == pytest.approx([1.0, 0.0, 3.0])
    assert found.std.tolist() == pytest.approx([2.0, 2**0.5, 0.0])
    assert found.mean_abs.tolist() == pytest.approx([1.0, 1.2, 3.0])
    assert found.rms.tolist() == pytest.approx([5**0.5, 2**0.5, 3.0])
    assert found.skew.tolist() == pytest.approx([12 / 8, 0.0, 0.0], abs=1e-12)
    assert found.kurt.tolist() == pytest.approx([52 / 16 - 3, 6.8 / 4 - 3, 0.0], abs=1e-12)


def test_chunked_equals_one_pass_and_row_order_does_not_matter() -> None:
    whole = document_moments(DOCS, chunk_size=5)
    for found in (document_moments(DOCS, chunk_size=2), document_moments(DOCS[::-1], 3)):
        for name in ("mean", "std", "mean_abs", "rms", "skew", "kurt"):
            assert np.allclose(getattr(found, name), getattr(whole, name), atol=1e-12)


def test_tail_share_counts_one_three_sigma_outlier_in_twenty() -> None:
    column = np.zeros((20, 2), dtype=np.float16)
    column[7, 0] = 4.0  # deviation 3.8 against 3 sigma = 2.6
    moments = document_moments(column, chunk_size=6)
    assert tail_share(column, moments, chunk_size=6).tolist() == [0.05, 0.0]


def test_a_constant_column_has_no_tail_and_no_nan() -> None:
    moments = document_moments(DOCS, chunk_size=2)
    tails = tail_share(DOCS, moments, chunk_size=2)
    assert tails[2] == 0.0 and np.isfinite(tails).all()
    assert np.isfinite(moments.skew).all() and np.isfinite(moments.kurt).all()


def test_on_a_small_corpus_the_nine_marginals_equal_marginal_stats() -> None:
    rng = np.random.default_rng(0)
    docs = rng.normal(size=(50, 4)).astype(np.float16)
    expected = marginal_stats(torch.from_numpy(docs)).numpy()
    rows = quantile_rows([f"d{i}" for i in range(50)])
    assert rows.tolist() == list(range(50))  # every document under the cap
    quantiles = document_quantiles(docs, rows).astype(np.float32)
    assert np.array_equal(quantiles, expected[:, 4:])
    moments = document_moments(docs, chunk_size=7)
    found = np.stack([moments.mean, moments.std, moments.mean_abs, moments.rms], axis=1)
    assert np.allclose(found.astype(np.float32), expected[:, :4], rtol=1e-6, atol=1e-7)


def test_above_the_cap_rows_are_a_seeded_hash_sample_of_document_ids() -> None:
    ids = [f"doc-{i}" for i in range(10)]
    rows = quantile_rows(ids, size=4)
    assert len(rows) == 4 and rows.tolist() == sorted(rows.tolist())
    assert np.array_equal(rows, quantile_rows(ids, size=4))
    shuffled = ids[::-1]
    assert {shuffled[r] for r in quantile_rows(shuffled, size=4)} == {ids[r] for r in rows}
