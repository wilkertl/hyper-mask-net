"""Top-k overlap between the retrieval selector and L2S, with seed noise floors (F5 D8)."""

import pytest
import torch

from hyperdime.cli.compare_selectors import overlaps
from hyperdime.cli.transfer_matrix import BUDGETS
from hyperdime.evaluation.selector_shift import overlap_at_k
from hyperdime.selection.topk import top_k_mask

GEN = torch.Generator().manual_seed(0)
SCORES = {s: torch.randn(200, 1024, generator=GEN) for s in (0, 1, 2)}


def test_identical_selectors_overlap_completely_at_every_budget() -> None:
    found = overlaps(SCORES, SCORES)
    assert list(found) == [str(k) for k in BUDGETS]
    assert all(v["retrieval_l2s"] == pytest.approx(1.0) for v in found.values())


def test_unrelated_selectors_overlap_by_chance() -> None:
    other = {s: torch.randn(200, 1024, generator=GEN) for s in (0, 1, 2)}
    for k, values in overlaps(SCORES, other).items():
        assert values["retrieval_l2s"] == pytest.approx(int(k) / 1024, abs=0.03)


def test_the_noise_floor_is_the_mean_over_the_three_seed_pairs() -> None:
    same = {0: SCORES[0], 1: SCORES[0], 2: SCORES[1]}  # pairs overlap 1, chance, chance
    floor = overlaps(same, SCORES)["512"]["l2s_seeds"]
    chance = float(overlap_at_k(top_k_mask(SCORES[0], 512), top_k_mask(SCORES[1], 512)).mean())
    assert floor == pytest.approx((1.0 + 2 * chance) / 3)
