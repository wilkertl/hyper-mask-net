"""The common query battery and pairwise behavior measures (F4 research D5, D7)."""

import hashlib
import math

import pytest
import torch

from hyperdime.evaluation.selector_shift import kl_divergence, pair_measures
from hyperdime.evaluation.transfer import sample_battery

IDS = {
    "small": ["s2", "s0", "s1"],
    "large": [f"l{i:02d}" for i in range(20)],
}


def test_the_battery_takes_the_smallest_count_from_every_environment() -> None:
    battery = sample_battery(IDS)
    assert battery.n == 3
    assert battery.ids["small"] == ["s0", "s1", "s2"]
    assert len(battery.ids["large"]) == 3 and set(battery.ids["large"]) <= set(IDS["large"])


def test_the_battery_is_seeded_and_ignores_input_order() -> None:
    reordered = {"small": IDS["small"], "large": list(reversed(IDS["large"]))}
    assert sample_battery(IDS) == sample_battery(reordered)
    assert sample_battery(IDS, seed=1).ids["large"] != sample_battery(IDS, seed=0).ids["large"]


def test_the_battery_hash_is_over_its_sorted_ids() -> None:
    battery = sample_battery(IDS)
    ids = sorted(f"{env}/{q}" for env, qids in battery.ids.items() for q in qids)
    assert battery.sha256 == hashlib.sha256("\n".join(ids).encode()).hexdigest()


def test_kl_divergence_by_hand() -> None:
    expected = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
    assert kl_divergence(torch.tensor([0.5, 0.5]), torch.tensor([0.25, 0.75])) == pytest.approx(
        expected
    )


def test_pair_measures_of_a_selector_with_itself() -> None:
    scores = torch.log_softmax(torch.randn(6, 1024, generator=torch.Generator().manual_seed(0)), -1)
    measures = pair_measures(scores, scores, k=307)
    assert set(measures) == {
        "overlap",
        "js",
        "kl_ab",
        "kl_ba",
        "spearman_importance",
        "spearman_frequency",
    }
    assert measures["overlap"] == pytest.approx(1.0)
    assert measures["js"] == pytest.approx(0.0, abs=1e-12)
    assert measures["kl_ab"] == pytest.approx(0.0, abs=1e-12)
    assert measures["spearman_importance"] == pytest.approx(1.0)
