"""P3's batched kernels on hand-computable cases, and against a GPU when there is one."""

import subprocess
import sys
from typing import Any

import numpy as np
import pytest
import torch

from hyperdime.evaluation.representation import (
    budget_masks,
    contribution_order,
    contributions,
    cumulative_scores,
)
from hyperdime.evaluation.representation_controls import (
    floor_curve,
)
from hyperdime.evaluation.representation_engine import (
    Clock,
    Profiles,
    finite_mean,
    gather,
    group_jaccard,
    group_measures,
    intersection_curves,
    jaccard_curves,
    pair_curves,
    ranks_of,
    stable_order,
)


def _t(values: Any) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values))


def _order(values: Any) -> torch.Tensor:
    return stable_order(_t(np.asarray(values, dtype=np.float64)))


def _ranks(values: Any) -> torch.Tensor:
    return ranks_of(_order(values))


def _measures(values: Any, groups: list[int]) -> tuple[list[float], list[float]]:
    conflict, cancelled = group_measures(
        _t(np.asarray(values, dtype=np.float64)),
        _t(np.arange(len(groups))),
        _t(np.array(groups)),
        max(groups) + 1,
    )
    return conflict.tolist(), cancelled.tolist()


def test_ranks_invert_an_ordering() -> None:
    assert ranks_of(_t([[2, 0, 1]])).tolist() == [[1, 2, 0]]


def test_ties_go_to_the_lower_dimension_whatever_the_sign_of_zero() -> None:
    assert _order([[0.0, -0.0, 1.0, 0.0, -0.0]]).tolist() == [[2, 0, 1, 3, 4]]
    assert _order([[-0.0, 0.0, -1.0]]).tolist() == [[0, 1, 2]]


def test_ordering_matches_the_numpy_definition() -> None:
    values = np.random.default_rng(5).normal(size=(8, 64)).round(1)  # rounding makes real ties
    assert _order(values).tolist() == contribution_order(values).tolist()


def test_pair_curves_keep_three_quantities_apart_on_a_hand_case() -> None:
    values = _t([[2.0, -2.0, 1.0]])  # order (0, 2, 1): P = 3, N = 2, full score 1
    curves = pair_curves(values, stable_order(values))
    assert curves["similarity"].tolist() == [[2.0, 3.0, 1.0]]
    assert curves["retained_positive_share"][0].tolist() == pytest.approx([2 / 3, 1.0, 1.0])
    assert curves["removed_negative_share"][0].tolist() == pytest.approx([1.0, 1.0, 0.0])


def test_pair_curves_end_at_the_full_score_and_full_masses() -> None:
    rng = np.random.default_rng(0)
    q, docs = rng.normal(size=64), rng.normal(size=(5, 64))
    values = contributions(q, docs)
    order = contribution_order(values)
    curves = pair_curves(_t(values), _t(order))
    assert curves["similarity"][:, -1].numpy() == pytest.approx(docs @ q)
    assert curves["similarity"].numpy() == pytest.approx(cumulative_scores(values, order))
    assert curves["retained_positive_share"][:, -1].numpy() == pytest.approx(np.ones(5))
    assert curves["removed_negative_share"][:, -1].numpy() == pytest.approx(np.zeros(5))


def test_a_pair_without_negative_mass_has_an_undefined_removed_share() -> None:
    values = _t([[1.0, 2.0]])
    curves = pair_curves(values, stable_order(values))
    assert curves["removed_negative_share"].isnan().all()
    assert curves["retained_positive_share"][0].tolist() == pytest.approx([2 / 3, 1.0])


def test_intersection_on_a_hand_case() -> None:
    # Orders (0, 1, 2, 3) and (1, 0, 3, 2): masks share 0, 2, 2, 4 dimensions.
    left = _ranks([[4.0, 3.0, 2.0, 1.0]])
    right = _ranks([[3.0, 4.0, 1.0, 2.0]])
    assert intersection_curves(left, right).tolist() == [[0, 2, 2, 4]]
    assert jaccard_curves(left, right)[0].tolist() == pytest.approx([0.0, 1.0, 0.5, 1.0])


def test_incremental_agreement_matches_explicit_masks_at_every_k() -> None:
    rng = np.random.default_rng(1)
    values = rng.normal(size=(6, 128))
    order = contribution_order(values)
    ranks = ranks_of(_t(order))
    curves = jaccard_curves(ranks[[0, 2, 4]], ranks[[1, 3, 5]]).numpy()
    for k in range(1, 129):
        masks = budget_masks(order, k)
        shared = (masks[[0, 2, 4]] & masks[[1, 3, 5]]).sum(axis=1)
        assert curves[:, k - 1] == pytest.approx(shared / (2 * k - shared))


def test_group_jaccard_averages_within_each_group_only() -> None:
    ranks = _ranks([[4.0, 3.0, 2.0, 1.0], [3.0, 4.0, 1.0, 2.0], [4.0, 3.0, 2.0, 1.0]])
    found = group_jaccard(ranks, _t([0, 0]), _t([1, 2]), _t([0, 1]), 2)
    assert found[0].tolist() == pytest.approx([0.0, 1.0, 0.5, 1.0])
    assert found[1].tolist() == [1.0, 1.0, 1.0, 1.0]


def test_masks_coincide_at_full_cardinality_whatever_the_documents() -> None:
    rng = np.random.default_rng(2)
    ranks = _ranks(rng.normal(size=(4, 32)))
    first, second = np.triu_indices(4, 1)
    found = group_jaccard(ranks, _t(first), _t(second), _t(np.zeros(6, dtype=np.int64)), 1)
    assert found[0, -1].item() == 1.0


def test_finite_mean_ignores_undefined_values_and_marks_empty_slots() -> None:
    values = _t([[1.0, np.nan], [3.0, np.nan], [np.nan, 5.0]])
    found = finite_mean(values, _t([0, 0, 1]), 3)
    assert found[0, 0].item() == 2.0 and found[0, 1].isnan()
    assert found[1, 1].item() == 5.0 and found[2].isnan().all()


def test_gather_returns_rows_in_the_order_asked_for() -> None:
    matrix = np.arange(8.0, dtype=np.float16).reshape(4, 2)
    assert gather(matrix, np.array([3, 0, 3])).tolist() == [[6, 7], [0, 1], [6, 7]]


def test_clock_adds_time_per_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = iter([1.0, 3.0, 10.0, 15.0])
    monkeypatch.setattr("hyperdime.evaluation.representation_engine.time.perf_counter",
                        lambda: next(ticks))  # fmt: skip
    clock = Clock("cpu")
    for _ in range(2):
        with clock.span("read"):
            pass
    assert clock.seconds == {"read": 7.0}


def test_two_real_subsets_never_fall_below_the_floor() -> None:
    rng = np.random.default_rng(3)
    ranks = _ranks(rng.normal(size=(2, 64)))
    assert (jaccard_curves(ranks[:1], ranks[1:])[0].numpy() >= floor_curve(64) - 1e-12).all()


def test_group_measures_on_a_hand_case() -> None:
    conflict, cancelled = _measures([[1.0, -1.0, 1.0], [-1.0, -1.0, 1.0]], [0, 0])
    assert conflict[0] == pytest.approx(1 / 3)  # only dimension 0 changes sign
    assert cancelled[0] == pytest.approx(1 - 2 / 3)  # mean (0, -1, 1): mass 2 of 3


def test_group_measures_keep_groups_apart() -> None:
    conflict, cancelled = _measures([[1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]], [0, 0, 1])
    assert conflict == [1.0, 0.0]
    assert cancelled == pytest.approx([1.0, 0.0])


def test_cancellation_can_fall_when_a_document_joins_but_conflict_cannot() -> None:
    # Two opposite documents: the mean is (0, 0), so all mass cancels.
    # A third like the first: mean (1/3, -1/3), mass 2/3 of 2, so 2/3 cancels. Conflict stays full.
    pair = [[1.0, -1.0], [-1.0, 1.0]]
    conflict_two, cancelled_two = _measures(pair, [0, 0])
    conflict_three, cancelled_three = _measures([*pair, [1.0, -1.0]], [0, 0, 0])
    assert cancelled_two[0] == pytest.approx(1.0)
    assert cancelled_three[0] == pytest.approx(2 / 3)
    assert conflict_two[0] == conflict_three[0] == 1.0


def test_cancellation_can_also_rise_when_a_document_joins() -> None:
    _, one = _measures([[1.0, -1.0]], [0])
    _, two = _measures([[1.0, -1.0], [-1.0, 1.0]], [0, 0])
    assert one[0] == pytest.approx(0.0) and two[0] == pytest.approx(1.0)


def _profiles(labels: np.ndarray, sizes: list[int]) -> Profiles:
    return Profiles(3, labels, np.array(sizes), "cpu")


def _add(profiles: Profiles, query: int, ranks: torch.Tensor, weight: float) -> None:
    rows = ranks.shape[0]
    profiles.add(
        _t(np.full(rows, query)),
        _t(np.zeros(rows, dtype=np.int64)),
        ranks,
        _t(np.full(rows, weight)),
    )


def test_a_query_weighs_one_however_many_reference_samples_it_has() -> None:
    labels = np.array([[0, 1]])
    single, repeated = _profiles(labels, [2]), _profiles(labels, [2])
    ranks = _ranks([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
    other = _ranks([[2.0, 3.0, 1.0]])
    _add(single, 0, ranks, 1 / 2)
    _add(single, 1, other, 1.0)
    for _ in range(5):
        _add(repeated, 0, ranks, 1 / 10)
        _add(repeated, 1, other, 1 / 5)
    assert repeated.frequency(2) == pytest.approx(single.frequency(2))
    assert repeated.frequency(2).sum(axis=1) == pytest.approx([1.0, 2.0, 3.0])
    assert repeated.half_frequency() == pytest.approx(single.half_frequency())
    assert repeated.stratum_frequency(0) == pytest.approx(single.stratum_frequency(0))


def test_profiles_give_query_balanced_frequency_at_every_k_without_masks() -> None:
    profiles = _profiles(np.array([[0, 1]]), [2])
    _add(profiles, 0, _ranks([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]]), 1 / 2)  # two documents
    _add(profiles, 1, _ranks([[3.0, 2.0, 1.0]]), 1.0)
    frequency = profiles.frequency(2)
    # k = 1: query 0 picks dimensions 0 and 2 half of the time, query 1 picks dimension 0.
    assert frequency[0].tolist() == pytest.approx([0.75, 0.0, 0.25])
    assert frequency.sum(axis=1).tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert frequency[-1].tolist() == pytest.approx([1.0, 1.0, 1.0])
    halves = profiles.half_frequency()
    assert halves[0, 1, 0].tolist() == pytest.approx([1.0, 0.0, 0.0])
    assert halves[0, 0, 0].tolist() == pytest.approx([0.5, 0.0, 0.5])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gpu_gives_the_cpu_ordering_and_curves() -> None:
    rng = np.random.default_rng(7)
    q = rng.normal(size=1024).astype(np.float16).astype(np.float64)
    docs = rng.normal(size=(512, 1024)).astype(np.float16).astype(np.float64)
    docs[0, :8] = 0.0  # zeros of both signs, a real source of ties
    values = q * docs
    cpu = _t(values)
    gpu = cpu.to("cuda")
    order = stable_order(cpu)
    assert torch.equal(order, stable_order(gpu).cpu())
    assert torch.equal(ranks_of(order), ranks_of(order.to("cuda")).cpu())
    found_cpu = pair_curves(cpu, order)
    found_gpu = pair_curves(gpu, order.to("cuda"))
    for name, curve in found_cpu.items():
        gap = (curve - found_gpu[name].cpu()).abs()
        assert gap[gap.isfinite()].max().item() < 1e-12


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_peak_memory_resets_on_a_device_not_used_yet() -> None:
    # Failed only in a process that had not touched CUDA: that needs a fresh interpreter.
    code = (
        "from hyperdime.evaluation.representation_engine import device_memory, reset_device_memory;"
        "reset_device_memory('cuda:0'); print(device_memory('cuda:0')['name'])"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
