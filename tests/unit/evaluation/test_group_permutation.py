"""G2 criterion 2's exact test over selector-to-environment assignments (F4 research D7)."""

import pytest
import torch

from hyperdime.evaluation.selector_shift import group_permutation_test, pair_measures

GROUPS = [0, 0, 0, 1, 1, 1, 2, 2, 2]


def _block_overlaps(groups: list[int]) -> torch.Tensor:
    """Overlap 1 between selectors of the same environment, 0 across."""
    labels = torch.tensor(groups)
    return (labels[:, None] == labels[None, :]).float()


def test_three_groups_of_three_have_1680_assignments_and_p_6_over_1680() -> None:
    statistic, p, assignments = group_permutation_test(_block_overlaps(GROUPS), GROUPS)
    assert assignments == 1680
    assert statistic == pytest.approx(1.0)
    assert p == pytest.approx(6 / 1680)


def test_a_constant_overlap_matrix_gives_p_1() -> None:
    statistic, p, _ = group_permutation_test(torch.full((9, 9), 0.4), GROUPS)
    assert statistic == pytest.approx(0.0) and p == 1.0


def test_the_order_of_selectors_does_not_matter() -> None:
    overlaps = torch.rand(9, 9, generator=torch.Generator().manual_seed(0))
    overlaps = (overlaps + overlaps.T) / 2
    order = torch.randperm(9, generator=torch.Generator().manual_seed(1))
    shuffled_groups = [GROUPS[i] for i in order.tolist()]
    expected = group_permutation_test(overlaps, GROUPS)
    shuffled = group_permutation_test(overlaps[order][:, order], shuffled_groups)
    assert shuffled == pytest.approx(expected)


def _criterion_2(environment_of_seed: list[int]) -> tuple[float, float, int]:
    """Nine synthetic selectors: each scores the battery with its environment's planted
    preference plus seed noise; criterion 2 runs on their measured overlaps."""
    battery = torch.randn(40, 1024, generator=torch.Generator().manual_seed(0))
    planted = torch.randn(3, 1024, generator=torch.Generator().manual_seed(1)) * 3
    scores = []
    for index, environment in enumerate(environment_of_seed):
        noise = torch.randn(1024, generator=torch.Generator().manual_seed(100 + index)) * 0.5
        scores.append(torch.log_softmax(battery * 0.3 + planted[environment] + noise, dim=-1))
    matrix = torch.eye(9, dtype=torch.float64)
    for i in range(9):
        for j in range(i + 1, 9):
            matrix[i, j] = matrix[j, i] = pair_measures(scores[i], scores[j], k=307)["overlap"]
    return group_permutation_test(matrix, GROUPS)


def test_planted_environment_preferences_satisfy_criterion_2() -> None:
    statistic, p, _ = _criterion_2(GROUPS)
    assert statistic > 0 and p <= 0.05


def test_selectors_of_one_environment_under_three_labels_do_not() -> None:
    statistic, p, _ = _criterion_2([0] * 9)
    assert not (statistic > 0 and p <= 0.05)
