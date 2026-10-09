"""Leave-one-environment-out descriptor probe and its exact permutation test (research R20 D3)."""

import pytest
import torch

from hyperdime.spike.diagnostics import loeo_probe, probe_permutation_test, recommendation

ENVS = ["a", "b", "c", "d", "e"]


def _features(rows: int = 200, columns: int = 3) -> dict[str, torch.Tensor]:
    return {
        env: torch.randn(rows, columns, generator=torch.Generator().manual_seed(i))
        for i, env in enumerate(ENVS)
    }


def test_probe_recovers_a_planted_linear_signal() -> None:
    features = _features()
    weights = torch.tensor([1.0, -2.0, 0.5])
    noise = torch.Generator().manual_seed(99)
    targets = {
        env: x @ weights + 0.1 * torch.randn(x.shape[0], generator=noise)
        for env, x in features.items()
    }
    result = loeo_probe(features, targets)
    assert set(result["folds"]) == set(ENVS)
    assert result["mean_gain"] > 0.5
    for fold in result["folds"].values():
        assert fold["gain"] == pytest.approx(fold["score"] - fold["baseline"])
    p, assignments = probe_permutation_test(features, targets)
    assert assignments == 120
    assert p == pytest.approx(1 / 120)


def test_probe_finds_no_gain_without_a_signal() -> None:
    features = _features()
    noise = torch.Generator().manual_seed(7)
    targets = {env: torch.randn(200, generator=noise) for env in ENVS}
    assert abs(loeo_probe(features, targets)["mean_gain"]) < 0.2


def test_probe_normalizes_with_the_training_environments_only() -> None:
    features = _features()
    targets = {env: x[:, 0] for env, x in features.items()}
    before = loeo_probe(features, targets)["folds"]["a"]["zscore"]
    scaled = {**features, "a": features["a"] * 1000}
    after = loeo_probe(scaled, targets)["folds"]["a"]["zscore"]
    assert after == before


@pytest.mark.parametrize(
    ("o1", "t", "expected"),
    [
        (False, False, "stop_or_pivot"),
        (False, True, "stop_or_pivot"),
        (True, True, "continue_strengthen_selector"),
        (True, False, "continue"),
    ],
)
def test_recommendation_follows_the_research_r20_table(o1: bool, t: bool, expected: str) -> None:
    assert recommendation(o1, t) == expected
