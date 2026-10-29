"""E2 probe helpers: ranks, the descriptor ladder, and the shift check (F10 research D0, D7)."""

import numpy as np
import pytest

from hyperdime.evaluation import descriptor_probe as probe


def test_average_ranks_share_ties_and_scale_to_the_unit_interval() -> None:
    # 0-based average ranks [3, 0.5, 0.5, 2], divided by n - 1 = 3.
    assert probe.average_ranks(np.array([3.0, 1.0, 1.0, 2.0])) == pytest.approx(
        [1.0, 1 / 6, 1 / 6, 2 / 3]
    )
    assert probe.average_ranks(np.array([5.0])) == pytest.approx([0.0])


def test_spearman_is_one_for_a_monotone_transform_and_minus_one_reversed() -> None:
    values = np.array([0.3, -1.0, 2.0, 0.0, 5.0])
    assert probe.spearman(values, np.exp(values)) == pytest.approx(1.0)
    assert probe.spearman(values, -(values**3)) == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        probe.spearman(values, values[:3])


def test_spearman_brown_steps_a_correlation_up_to_more_replicates() -> None:
    assert probe.spearman_brown(0.5, 3) == pytest.approx(0.75)  # 3 * 0.5 / (1 + 2 * 0.5)
    assert probe.spearman_brown(1.0, 3) == pytest.approx(1.0)
    assert probe.spearman_brown(0.0, 3) == pytest.approx(0.0)


def test_the_ladder_is_six_nested_sets_with_their_protocols() -> None:
    names = [s.name for s in probe.LADDER]
    assert names == ["variance", "A", "A+E", "A+E+tails", "+B+C", "+D"]
    assert [s.protocol for s in probe.LADDER] == ["P0"] * 4 + ["P1"] * 2
    assert probe.LADDER[0].columns == ("doc_std",)
    for smaller, larger in zip(probe.LADDER[1:], probe.LADDER[2:], strict=False):
        assert set(smaller.columns) < set(larger.columns)
    assert [len(s.columns) for s in probe.LADDER] == [1, 9, 15, 18, 40, 43]


def _environments(
    own: dict[str, float], dim: int = 256, noise: float = 0.1
) -> dict[str, dict[int, np.ndarray]]:
    """Three environments x three seeds: one shared profile, an optional environment-specific
    component of the given strength, and independent seed noise."""
    generator = np.random.default_rng(0)
    shared = generator.normal(size=dim)
    found = {}
    for env in ("a", "b", "c"):
        profile = shared + own.get(env, 0.0) * generator.normal(size=dim)
        found[env] = {s: profile + noise * generator.normal(size=dim) for s in (0, 1, 2)}
    return found


def test_the_shift_check_finds_no_room_when_environments_share_one_profile() -> None:
    check = probe.shift_check(_environments({}))
    assert check["no_shift"] is True
    for fold in check["folds"].values():
        assert fold["gap"] < probe.NO_SHIFT_GAP
        assert fold["ceiling"] == pytest.approx(probe.spearman_brown(fold["seed_spearman"], 3))


def test_the_shift_check_reports_the_gap_of_an_environment_specific_component() -> None:
    check = probe.shift_check(_environments({"a": 1.0}))
    gaps = {env: fold["gap"] for env, fold in check["folds"].items()}
    assert gaps["a"] > probe.NO_SHIFT_GAP
    # The odd environment also enters the other folds' B1, but its own gap is the largest.
    assert gaps["a"] == max(gaps.values())
    assert check["no_shift"] is False
    small = probe.shift_check(_environments({"a": 0.02}))
    assert small["no_shift"] is True


def test_b1_is_the_mean_of_the_other_environments_seed_averaged_profiles() -> None:
    envs = _environments({"a": 1.0})
    check = probe.shift_check(envs)
    target = np.mean([envs["a"][s] for s in (0, 1, 2)], axis=0)
    shared = np.mean([np.mean([envs[e][s] for s in (0, 1, 2)], axis=0) for e in "bc"], axis=0)
    assert check["folds"]["a"]["b1"] == pytest.approx(probe.spearman(shared, target))


def test_the_shift_check_needs_three_environments_and_two_seeds() -> None:
    envs = _environments({})
    with pytest.raises(ValueError, match="three environments"):
        probe.shift_check({e: envs[e] for e in "ab"})
    with pytest.raises(ValueError, match="two seeds"):
        probe.shift_check({e: {0: envs[e][0]} for e in "abc"})
