"""Descriptor group A: per-dimension marginal statistics of a corpus (protocol P0)."""

import math

import pytest
import torch

from hyperdime.space.marginal import MARGINAL_FEATURES, marginal_stats


def test_marginal_stats_match_hand_computed_columns() -> None:
    documents = torch.tensor(
        [
            [1.0, -2.0, 0.0],
            [3.0, 2.0, 0.0],
            [5.0, -2.0, 0.0],
            [7.0, 2.0, 4.0],
        ]
    )
    stats = marginal_stats(documents)
    assert MARGINAL_FEATURES == (
        "mean",
        "std",
        "mean_abs",
        "rms",
        "q05",
        "q25",
        "q50",
        "q75",
        "q95",
    )
    assert stats.shape == (3, 9)
    # Population std; quantiles interpolate linearly between order statistics.
    expected = [
        [4.0, math.sqrt(5.0), 4.0, math.sqrt(21.0), 1.3, 2.5, 4.0, 5.5, 6.7],
        [0.0, 2.0, 2.0, 2.0, -2.0, -2.0, 0.0, 2.0, 2.0],
        [1.0, math.sqrt(3.0), 1.0, 2.0, 0.0, 0.0, 0.0, 1.0, 3.4],
    ]
    for row, values in zip(stats.tolist(), expected, strict=True):
        assert row == pytest.approx(values, abs=1e-6)


def test_marginal_stats_ignore_document_order_and_follow_dimension_order() -> None:
    documents = torch.randn(10, 1024, generator=torch.Generator().manual_seed(0))
    stats = marginal_stats(documents)
    assert stats.shape == (1024, len(MARGINAL_FEATURES))
    shuffled = documents[torch.randperm(10, generator=torch.Generator().manual_seed(1))]
    assert torch.allclose(marginal_stats(shuffled), stats, atol=1e-6)
    permutation = torch.randperm(1024, generator=torch.Generator().manual_seed(2))
    assert torch.allclose(marginal_stats(documents[:, permutation]), stats[permutation], atol=1e-6)
