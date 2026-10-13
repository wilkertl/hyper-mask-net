"""Selector fit as a ranking (F4 research D2), with a hand-computed example."""

import torch

from hyperdime.evaluation.transfer import fit_gain, fits


def _scores(*subsets: set[int], dim: int = 6) -> torch.Tensor:
    """Per-query importance whose top-2 is exactly each given subset."""
    rows = torch.zeros(len(subsets), dim)
    for row, subset in zip(rows, subsets, strict=True):
        row[sorted(subset)] = torch.tensor([2.0, 1.0])
    return rows


def test_fit_gain_by_hand() -> None:
    selector = _scores({0, 1}, {2, 3})
    global_ = _scores({0, 5}, {0, 5})
    oracle = _scores({0, 1}, {2, 4})
    # selector vs oracle: 2/2 and 1/2; global vs oracle: 1/2 and 0/2.
    assert fit_gain([selector], global_, oracle, k=2).tolist() == [0.5, 0.5]


def test_seed_overlaps_are_averaged_per_query() -> None:
    oracle = _scores({0, 1}, {2, 4})
    global_ = _scores({0, 5}, {0, 5})
    seeds = [_scores({0, 1}, {2, 3}), _scores({4, 5}, {2, 4})]
    # seed overlaps (1, 0.5) and (0, 1) average to (0.5, 0.75); minus global (0.5, 0).
    assert fit_gain(seeds, global_, oracle, k=2).tolist() == [0.0, 0.75]


def test_fit_needs_a_positive_gain_and_a_significant_holm_p() -> None:
    assert fits(0.1, 0.05)
    assert not fits(0.1, 0.06)
    assert not fits(0.0, 0.001)
    assert not fits(-0.2, 0.001)
