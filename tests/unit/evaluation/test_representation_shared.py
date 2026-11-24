"""The shared-mask kernels on hand-computable cases (ADR-0015)."""

from itertools import combinations
from typing import Any

import numpy as np
import pytest
import torch

from hyperdime.evaluation.representation_shared import (
    TINY_REGRET,
    cell_sizes,
    fixed_count,
    group_summary,
    masses,
    optimum_curves,
    overlap_curves,
    partitions,
    regret_shares,
    scores_under,
    shared_order,
    strata_of,
    transfer_cell,
    uniform_mask_regret,
    within_query_correlation,
)


def _t(values: Any) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values, dtype=np.float64))


def _regret(values: torch.Tensor, weights: list[float] | None = None) -> torch.Tensor:
    count = values.shape[0]
    weight = _t(weights if weights is not None else [1.0 / count] * count)[None]
    order = shared_order(values[None], weight)
    optimum, _ = optimum_curves(values)
    return optimum - scores_under(values[None], order)[0]


def test_conflicting_positives_pay_a_hand_computable_regret() -> None:
    # Mean (0.5, 0.5, 2.0): top 1 is dimension 2 for both. Own bests are 3 and 3; under it 1 and 3.
    values = _t([[3.0, -1.0, 1.0], [-2.0, 2.0, 3.0]])
    raw = _regret(values)
    assert raw[:, 0].tolist() == [2.0, 0.0]


def test_equal_contributions_have_no_regret() -> None:
    assert (_regret(_t([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])) == 0).all()


def test_tied_dimensions_have_no_regret_whichever_mask_is_chosen() -> None:
    assert (_regret(_t([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])) == 0).all()


def test_regret_is_never_negative_and_vanishes_at_full_cardinality() -> None:
    values = _t(np.random.default_rng(0).normal(size=(6, 64)))
    raw = _regret(values)
    assert raw.min().item() > -1e-12
    assert raw[:, -1].abs().max().item() < 1e-12


def test_mean_top_k_is_the_weighted_optimum_at_each_k_by_brute_force() -> None:
    rng = np.random.default_rng(1)
    values = rng.normal(size=(3, 6))
    weights = np.array([0.5, 0.25, 0.25])
    order = shared_order(_t(values)[None], _t(weights)[None])
    scores = scores_under(_t(values)[None], order)[0].numpy()
    for k in range(1, 7):
        best = max(
            (weights @ values[:, list(chosen)].sum(axis=1)) for chosen in combinations(range(6), k)
        )
        assert weights @ scores[:, k - 1] == pytest.approx(best)


def test_a_tie_in_the_mean_is_exact_and_goes_to_the_lower_dimension() -> None:
    # Every dimension sums to 6. Weights of 1/3 would round; whole-number gains cannot.
    values = _t([[3.0, 1.0, 2.0], [1.0, 3.0, 2.0], [2.0, 2.0, 2.0]])
    assert shared_order(values[None], _t([[1.0, 1.0, 1.0]])).tolist() == [[0, 1, 2]]


def test_weights_decide_whose_optimum_the_mask_serves() -> None:
    values = _t([[2.0, 0.0], [0.0, 3.0]])
    assert _regret(values, [0.9, 0.1])[0, 0].item() == 0.0
    assert _regret(values, [0.1, 0.9])[1, 0].item() == 0.0


def test_a_document_without_mass_has_undefined_ratios() -> None:
    found = masses(_t([[0.0, 0.0], [1.0, -2.0]]))
    assert found[0].isnan() and found[1].item() == 3.0


def test_uniform_mask_regret_is_exact_on_a_hand_case() -> None:
    values = _t([[3.0, 1.0]])  # keeps k/2 of the score 4 in expectation: 2, then 4
    optimum, _ = optimum_curves(values)
    assert uniform_mask_regret(optimum, values)[0].tolist() == [1.0, 0.0]


def test_overlap_is_one_when_a_document_agrees_with_the_mask() -> None:
    values = _t([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
    _, ranks = optimum_curves(values)
    order = torch.tensor([[0, 1, 2]])
    found = overlap_curves(ranks[None], order)[0]
    assert found[0].tolist() == [1.0, 1.0, 1.0]
    assert found[1].tolist() == pytest.approx([0.0, 1 / 3, 1.0])


def test_group_summary_aggregates_within_one_query() -> None:
    raw = _t([[0.0, 2.0], [0.0, 4.0], [0.0, 6.0]])
    found = group_summary(raw, raw / 10, _t([[1.0, 0.5]] * 3))
    assert found["mean_norm"].tolist() == pytest.approx([0.0, 0.4])
    assert found["max_norm"].tolist() == pytest.approx([0.0, 0.6])
    assert found["max_raw"].tolist() == [0.0, 6.0]
    assert found["p90_norm"].tolist() == pytest.approx([0.0, 0.56])
    assert found["std_norm"][1].item() == pytest.approx(np.std([0.2, 0.4, 0.6]))
    assert found["worst_share"][0].isnan()
    assert found["worst_share"][1].item() == pytest.approx(0.5)


def test_regret_share_is_undefined_without_regret() -> None:
    worst, fifth = regret_shares(_t([[TINY_REGRET / 10], [0.0]]))
    assert worst.isnan().all() and fifth.isnan().all()


def test_within_query_correlation_is_undefined_for_a_flat_positive_group() -> None:
    found = within_query_correlation(_t([[0.1], [0.1]]), _t([[0.2], [0.4]]))
    assert found.isnan().all()
    assert within_query_correlation(_t([[0.1], [0.2]]), _t([[0.4], [0.2]])).item() == -1.0


def test_two_positives_use_both_directed_transfers() -> None:
    found = partitions(2, 1, 1, np.random.default_rng(0), 8)
    assert [(c.tolist(), h.tolist()) for c, h in found] == [([0], [1]), ([1], [0])]


def test_partitions_are_disjoint_deterministic_and_bounded() -> None:
    first = partitions(20, 5, 5, np.random.default_rng(3), 8)
    second = partitions(20, 5, 5, np.random.default_rng(3), 8)
    assert len(first) == 8
    for (a, b), (c, d) in zip(first, second, strict=True):
        assert a.tolist() == c.tolist() and b.tolist() == d.tolist()
        assert not set(a.tolist()) & set(b.tolist())


def test_partitions_are_exhaustive_when_few_exist() -> None:
    assert len(partitions(3, 1, 1, np.random.default_rng(0), 8)) == 6
    with pytest.raises(ValueError, match="asked of"):
        partitions(2, 2, 1, np.random.default_rng(0), 8)


def test_cells_admit_only_queries_with_enough_positives() -> None:
    assert cell_sizes(3, 2, 2) is None
    assert cell_sizes(4, 2, 2) == (2, 2)
    assert cell_sizes(5, None, None) == (2, 3)


def _cell(values: torch.Tensor, grades: np.ndarray) -> Any:
    optimum, ranks = optimum_curves(values)
    mass = masses(values)
    privileged = _t(np.zeros(values.shape))
    splits = [(np.array([0]), np.array([1])), (np.array([1]), np.array([0]))]
    reference = _t([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]])
    return transfer_cell(values, optimum, mass, ranks, grades, splits, privileged, reference)


def test_transfer_scores_a_held_out_document_against_its_own_optimum() -> None:
    values = _t([[3.0, 1.0, 0.0], [1.0, 3.0, 0.0]])
    result = _cell(values, np.array([1, 1]))
    # k = 1: each mask picks its own best, 3, and the other document then scores 1: regret 2 of 4.
    assert result.measures["held_mean_norm"][0].item() == pytest.approx(0.5)
    assert result.measures["construct_mean_norm"][0].item() == 0.0
    assert result.per_partition.shape[0] == result.partitions == 2


def test_held_out_values_cannot_affect_the_mask() -> None:
    values = _t(np.random.default_rng(5).normal(size=(4, 16)))
    changed = values.clone()
    changed[2:] = -100 * changed[2:]
    weights = _t([[0.5, 0.5]])
    order = shared_order(values[None][:, [0, 1]], weights)
    assert torch.equal(order, shared_order(changed[None][:, [0, 1]], weights))


def test_held_out_grades_cannot_affect_the_mask() -> None:
    values = _t(np.random.default_rng(5).normal(size=(4, 16)))
    splits = [(np.array([0, 1]), np.array([2, 3]))]
    optimum, ranks = optimum_curves(values)
    reference = _t(np.random.default_rng(6).normal(size=(1, 2, 16)))
    zero = _t(np.zeros(values.shape))
    low, high = (
        transfer_cell(values, optimum, masses(values), ranks, np.array(grades), splits, zero,
                      reference)
        for grades in ([1, 1, 1, 1], [1, 1, 3, 3])
    )  # fmt: skip
    assert torch.equal(low.measures["held_mean_norm"], high.measures["held_mean_norm"])


def test_fixed_count_takes_the_worst_of_the_group_it_was_given() -> None:
    values = _t([[3.0, -1.0, 1.0], [-2.0, 2.0, 3.0], [9.0, 9.0, 9.0]])
    optimum, _ = optimum_curves(values)
    found = fixed_count(values, optimum, masses(values), np.array([1, 1, 1]), np.array([[0, 1]]))
    assert found["max_raw"][0].item() == 2.0
    assert found["mean_raw"][0].item() == 1.0
    assert found["worst_share"][0].item() == 1.0


def test_strata_of_counts_queries_not_pairs() -> None:
    found = strata_of(np.array([2, 5, 6, 40]))
    assert found["multi"].sum() == 4
    assert found["positives 2-5"].sum() == 2 and found["positives 21+"].sum() == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gpu_gives_the_cpu_regret() -> None:
    rng = np.random.default_rng(7)
    values = _t(rng.normal(size=(32, 1024)).astype(np.float16))
    gains = _t(np.ones((1, 32)))
    order = shared_order(values[None], gains)
    assert torch.equal(order, shared_order(values[None].to("cuda"), gains.to("cuda")).cpu())
    optimum, _ = optimum_curves(values)
    raw = optimum - scores_under(values[None], order)[0]
    gpu_optimum, _ = optimum_curves(values.to("cuda"))
    gpu_raw = gpu_optimum - scores_under(values[None].to("cuda"), order.to("cuda"))[0]
    assert (raw - gpu_raw.cpu()).abs().max().item() < 1e-12
