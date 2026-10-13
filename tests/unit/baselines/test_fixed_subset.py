"""The transferred fixed dimension subset (F4 research D6)."""

import torch

from hyperdime.baselines.methods import fixed_subset, rank_importance
from hyperdime.selection.topk import top_k_mask


def test_fixed_subset_breaks_ties_by_the_lower_index() -> None:
    frequencies = torch.tensor([0.9, 0.5, 0.1, 0.5])
    assert fixed_subset([frequencies], k=2).tolist() == [0, 1]


def test_frequencies_are_averaged_over_seeds_before_ranking() -> None:
    seeds = [torch.tensor([1.0, 0.0, 0.6, 0.4]), torch.tensor([0.0, 1.0, 0.6, 0.4])]
    # means 0.5, 0.5, 0.6, 0.4: dimension 2 first, then 0 before 1 by index.
    assert fixed_subset(seeds, k=2).tolist() == [2, 0]


def test_rank_importance_selects_exactly_the_subset_for_every_query() -> None:
    subset = torch.tensor([7, 3, 1000])
    importance = rank_importance(subset, torch.randn(4, 1024))
    expected = torch.zeros(4, 1024, dtype=torch.bool)
    expected[:, [3, 7, 1000]] = True
    assert torch.equal(top_k_mask(importance, 3), expected)
