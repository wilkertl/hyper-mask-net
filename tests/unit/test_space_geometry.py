"""Streamed corpus geometry: variance, covariance spectrum, effective dimensionality (P1)."""

import numpy as np
import pytest

from hyperdime.space.geometry import corpus_geometry

# Columns are independent: +-2, +-1, +-1 in a balanced design, and a constant 3.
SIGNS = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.float64)
DOCS = np.hstack([SIGNS * [2.0, 1.0, 1.0], np.full((4, 1), 3.0)]).astype(np.float16)


def test_variances_and_participation_ratio_by_hand() -> None:
    found = corpus_geometry(DOCS, chunk_size=4)
    assert found.variance.tolist() == pytest.approx([4.0, 1.0, 1.0, 0.0])
    assert found.mean.tolist() == pytest.approx([0.0, 0.0, 0.0, 3.0])
    assert sorted(found.eigenvalues.tolist(), reverse=True) == pytest.approx([4.0, 1.0, 1.0, 0.0])
    assert found.participation_ratio == pytest.approx(36 / 18)


def test_energy_in_the_mean_by_hand() -> None:
    # E||x||^2 = 4 + 1 + 1 + 9 = 15; ||mu||^2 = 9.
    assert corpus_geometry(DOCS, chunk_size=4).mean_energy == pytest.approx(9 / 15)


def test_the_chunk_size_does_not_change_the_result() -> None:
    whole, chunked = corpus_geometry(DOCS, chunk_size=4), corpus_geometry(DOCS, chunk_size=3)
    assert np.allclose(whole.variance, chunked.variance)
    assert np.allclose(whole.eigenvalues, chunked.eigenvalues)
