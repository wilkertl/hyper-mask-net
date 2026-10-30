"""Group E, redundancy, from a covariance with known correlations (F10 research D4)."""

import numpy as np
import pytest

from hyperdime.space.redundancy import REDUNDANCY_FEATURES, redundancy_features

# Standard deviations 1, 2, 3; correlations r01 = 0.5, r02 = 0, r12 = -0.25.
STD = np.array([1.0, 2.0, 3.0])
CORR = np.array([[1.0, 0.5, 0.0], [0.5, 1.0, -0.25], [0.0, -0.25, 1.0]])
COV = CORR * np.outer(STD, STD)


def test_correlation_features_by_hand() -> None:
    found = redundancy_features(COV)
    assert list(found) == list(REDUNDANCY_FEATURES)
    assert found["corr_mean_abs"].tolist() == pytest.approx([0.25, 0.375, 0.125])
    assert found["corr_max_abs"].tolist() == pytest.approx([0.5, 0.5, 0.25])
    # min(8, D - 1) = 2 partners: the mean of both squared correlations.
    assert found["corr_top8"].tolist() == pytest.approx([0.125, 0.15625, 0.03125])
    # r capped at D = 3: the whole space holds every dimension's variance.
    for r in (8, 32, 128):
        assert found[f"pc_share_{r}"].tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_a_diagonal_covariance_puts_the_top_dimensions_in_the_top_components() -> None:
    variances = np.random.default_rng(0).permutation(np.arange(1.0, 41.0))  # distinct
    found = redundancy_features(np.diag(variances))
    for r in (8, 32):
        top = variances >= 41 - r
        assert found[f"pc_share_{r}"][top].tolist() == pytest.approx([1.0] * r)
        assert found[f"pc_share_{r}"][~top].tolist() == pytest.approx([0.0] * (40 - r), abs=1e-12)
    assert found["corr_max_abs"].tolist() == pytest.approx([0.0] * 40)


def test_a_zero_variance_dimension_gets_zero_correlations_and_share() -> None:
    cov = np.zeros((3, 3))
    cov[:2, :2] = [[1.0, 0.8], [0.8, 1.0]]
    found = redundancy_features(cov)
    for values in found.values():
        assert np.isfinite(values).all() and values[2] == 0.0
    assert found["corr_max_abs"][:2].tolist() == pytest.approx([0.8, 0.8])


def test_permuting_dimensions_permutes_the_rows() -> None:
    order = [2, 0, 1]
    found = redundancy_features(COV)
    permuted = redundancy_features(COV[np.ix_(order, order)])
    for name in REDUNDANCY_FEATURES:
        assert np.allclose(permuted[name], found[name][order])
