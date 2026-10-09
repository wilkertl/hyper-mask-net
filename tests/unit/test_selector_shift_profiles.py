"""Oracle top-k selection profiles and their reliability and permutation tests (research R20 D1)."""

import pytest
import torch

from hyperdime.evaluation.selector_shift import (
    half_split_profiles,
    profile_permutation_test,
    selection_frequency,
    spearman,
    split_half_spearman,
)


def test_selection_frequency_counts_top_k_masks_by_hand() -> None:
    scores = torch.tensor(
        [
            [4.0, 3.0, 1.0, 0.0],  # top-2: dims 0, 1
            [0.0, 5.0, 6.0, 1.0],  # top-2: dims 1, 2
            [9.0, 0.0, 0.0, 8.0],  # top-2: dims 0, 3
        ]
    )
    frequency = selection_frequency(scores, k=2)
    assert frequency.tolist() == pytest.approx([2 / 3, 2 / 3, 1 / 3, 1 / 3])
    assert float(frequency.sum()) == pytest.approx(2.0)


def test_spearman_gives_tied_values_their_average_rank() -> None:
    # Average ranks: a -> [0.5, 0.5, 2.5, 2.5], b -> [2.5, 0.5, 0.5, 2.5]; centered dot = 0.
    # Breaking ties by position would give 0.4 instead.
    assert spearman(torch.tensor([0.0, 0.0, 1.0, 1.0]), torch.tensor([1.0, 0.0, 0.0, 1.0])) == (
        pytest.approx(0.0)
    )
    assert spearman(torch.tensor([0.0, 0.0, 1.0]), torch.tensor([0.0, 0.0, 1.0])) == pytest.approx(
        1.0
    )


def test_split_half_profiles_are_seeded_and_disjoint_halves() -> None:
    scores = torch.randn(11, 16, generator=torch.Generator().manual_seed(0))
    first_a, first_b = half_split_profiles(scores, k=4, draws=5, seed=3)
    again_a, again_b = half_split_profiles(scores, k=4, draws=5, seed=3)
    assert first_a.shape == first_b.shape == (5, 16)
    assert torch.equal(first_a, again_a) and torch.equal(first_b, again_b)
    # Each half has 5 queries, so each profile sums to k and is a multiple of 1/5.
    assert first_a.sum(dim=-1).tolist() == pytest.approx([4.0] * 5)
    assert torch.allclose(first_a * 5, (first_a * 5).round())


def test_split_half_spearman_is_one_when_every_query_selects_the_same_dimensions() -> None:
    row = torch.arange(8, dtype=torch.float32)
    scores = row.expand(10, 8) + 0.01 * torch.randn(
        10, 8, generator=torch.Generator().manual_seed(1)
    )
    assert split_half_spearman(scores, k=3, draws=20, seed=0) == pytest.approx(1.0)
    assert split_half_spearman(scores, k=3, draws=20, seed=0) == split_half_spearman(
        scores, k=3, draws=20, seed=0
    )


def test_profile_permutation_test_extremes() -> None:
    generator = torch.Generator().manual_seed(2)
    noise = 0.01 * torch.randn(8, 8, generator=generator)
    same = torch.arange(8, dtype=torch.float32).expand(8, 8) + noise
    # Identical environments: every shuffle is as extreme as the observed split.
    assert profile_permutation_test(same, same.clone(), k=2, n=200, seed=0) == pytest.approx(1.0)
    # Disjoint supports: environment a always selects dims {0, 1}, b always {2, 3}.
    a = torch.zeros(8, 8)
    a[:, :2] = 1.0
    b = torch.zeros(8, 8)
    b[:, 2:4] = 1.0
    a, b = a + noise, b + noise
    assert profile_permutation_test(a, b, k=2, n=200, seed=0) <= 2 / 201
