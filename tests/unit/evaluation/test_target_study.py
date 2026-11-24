"""P3 target-encoding study (ADR-0019): hand-computable math and the trainer's exactness."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from hyperdime.baselines.l2s_official import (
    DEFAULT_CLONE,
    L2S_COMMIT,
    clone_commit,
    import_official,
)
from hyperdime.evaluation.benchmark import positive_ranks
from hyperdime.evaluation.target_study import (
    Selector,
    jaccard,
    kl_rows,
    loss_budget_input,
    loss_kl,
    loss_membership,
    margins,
    mask_average,
    nesting,
    ranks_from_scores,
    top_k_mask,
    train,
)


def test_top_k_mask_breaks_ties_to_the_lower_index() -> None:
    mask = top_k_mask(torch.tensor([1.0, 3.0, 3.0, 2.0, 3.0]), 2)
    assert mask.tolist() == [False, True, True, False, False]


def test_jaccard_by_hand() -> None:
    a = torch.tensor([True, True, False, False])
    b = torch.tensor([True, False, True, False])
    assert jaccard(a, b).item() == pytest.approx(1 / 3)
    assert jaccard(torch.zeros(3, dtype=torch.bool), torch.zeros(3, dtype=torch.bool)).item() == 1.0


def test_mask_average_is_the_mean_of_uniform_distributions() -> None:
    masks = torch.tensor([[[True, False, False, False], [True, False, True, False]]])
    assert torch.allclose(
        mask_average(masks), torch.tensor([[0.75, 0.0, 0.25, 0.0]], dtype=torch.float64)
    )


def test_kl_rows_by_hand_with_zero_target_entries() -> None:
    target = torch.tensor([[0.5, 0.5, 0.0]])
    log_probs = torch.log(torch.tensor([[0.25, 0.25, 0.5]]))
    assert kl_rows(target, log_probs).item() == pytest.approx(math.log(2))


def test_ranks_from_scores_equal_the_evaluator_ranks() -> None:
    scores = torch.tensor(
        [[0.5, 0.1], [0.9, 0.1], [0.5, 0.3], [0.1, 0.3], [0.5, 0.0]], dtype=torch.float64
    )
    pos = torch.tensor([2, 0, 4])
    got = ranks_from_scores(scores, pos)
    for column in range(2):
        assert torch.equal(got[:, column], positive_ranks(scores[:, column], pos))


def test_margins_by_hand() -> None:
    scores = torch.tensor([[0.2], [0.9], [0.5]], dtype=torch.float64)
    assert margins(scores, torch.tensor([0, 2])).item() == pytest.approx(0.5 - 0.9)


def test_nesting_by_hand() -> None:
    masks = torch.tensor([[[True, True, False, False], [True, False, True, True]]])
    assert nesting(masks).tolist() == [[0.5]]


def test_per_budget_cross_entropy_without_budget_input_equals_the_average_target() -> None:
    torch.manual_seed(0)
    masks = torch.rand(4, 3, 16) > 0.5
    masks[..., 0] = True
    log_p = F.log_softmax(torch.randn(4, 16, dtype=torch.float64), -1)
    per_budget = (
        sum(
            -(masks[:, b].double() / masks[:, b].sum(-1, keepdim=True) * log_p).sum(-1)
            for b in range(3)
        )
        / 3
    )
    averaged = -(mask_average(masks) * log_p).sum(-1)
    assert torch.allclose(per_budget, averaged)


def test_membership_loss_is_low_when_the_scores_match_the_masks() -> None:
    model = Selector(8)
    with torch.no_grad():
        model.net[0].weight.copy_(torch.eye(8))
        model.net[0].bias.zero_()
    x = torch.tensor([[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
    good = torch.stack([top_k_mask(x, 2), top_k_mask(x, 4)], 1)
    bad = torch.stack([top_k_mask(-x, 2), top_k_mask(-x, 4)], 1)
    assert loss_membership(model, x, good, 0.1, [2, 4]) < loss_membership(
        model, x, bad, 0.1, [2, 4]
    )


def test_budget_input_offsets_start_at_zero_and_select_per_budget() -> None:
    model = Selector(4, budgets=2)
    x = torch.randn(3, 4)
    assert torch.equal(model.logits(x, 0), model.logits(x, 1))
    targets = torch.full((3, 2, 4), 0.25)
    assert loss_budget_input(model, x, targets).item() == pytest.approx(
        F.kl_div(model(x), targets[:, 0], reduction="batchmean").item(), rel=1e-6
    )


def _clone_ready() -> bool:
    try:
        return clone_commit(DEFAULT_CLONE) == L2S_COMMIT
    except Exception:
        return False


@pytest.mark.skipif(not _clone_ready(), reason="authors' clone at the pinned commit")
def test_trainer_with_the_kl_loss_equals_the_authors_train_predictor() -> None:
    module = import_official(DEFAULT_CLONE)
    rng = np.random.default_rng(1)
    x = torch.from_numpy(rng.standard_normal((70, 16)).astype(np.float32))
    target = torch.softmax(torch.from_numpy(rng.standard_normal((70, 16)).astype(np.float32)), -1)
    module.set_random_seed(42)
    loader = module.DataLoader(module.PredictorDataset(x, target), batch_size=32, shuffle=True)
    reference = module.DimensionPredictor(input_dim=16)
    reference, _ = module.train_predictor(
        model=reference, train_loader=loader, epochs=4, device="cpu", lr=1e-4
    )
    ours = train(
        x, target, lambda m, xb, tb, _beta: loss_kl(m, xb, tb), epochs=4, seed=42, device="cpu",
        set_seed=module.set_random_seed,
    )  # fmt: skip
    for name, value in reference.state_dict().items():
        assert torch.equal(value, ours.model.state_dict()[name]), name


def test_pairwise_seed_agreement_and_tune_summary_by_hand() -> None:
    from hyperdime.cli.p3_target import _pairwise, _score_summary

    a = torch.tensor([True, True, False, False])
    b = torch.tensor([True, False, True, False])
    assert _pairwise([a, a, b]) == pytest.approx((1 + 1 / 3 + 1 / 3) / 3)
    rows = [
        {"half": h, "query": q, "k": k, "arm": arm, "ndcg10": v}
        for q, h in (("q1", "A"), ("q2", "B"), ("q3", "B"))
        for k in (102, 204)
        for arm, v in (("E0@42", 0.5), ("E1@42", 0.4))
    ]
    summary = _score_summary(rows, ["q1", "q2", "q3"], {"q1": "A", "q2": "B", "q3": "B"})
    assert summary["queries"] == {"A": 1, "B": 2}
    e1 = summary["arms"]["E1@42"]
    assert e1["tune_B_mean"] == pytest.approx(0.4)
    assert e1["tune_B_minus_E0_same_seed"]["D"] == pytest.approx(-0.1)
    assert "tune_B_minus_E0_same_seed" not in summary["arms"]["E0@42"]


def test_arm_payloads_have_the_declared_shapes_and_budget_inputs() -> None:
    from hyperdime.cli.p3_target import _arm_payload

    masks = torch.zeros(5, 9, 1024, dtype=torch.bool)
    for b, k in enumerate((102, 204, 307, 409, 512, 614, 716, 819, 921)):
        masks[:, b, :k] = True
    data = {
        "masks": masks,
        "official": torch.full((5, 1024), 1 / 1024),
        "average": mask_average(masks).float(),
    }
    for arm, shape, budgets in (
        ("E0", (5, 1024), 0),
        ("E1", (5, 1024), 0),
        ("E2", (5, 9, 1024), 0),
        ("E3", (5, 9, 1024), 9),
        ("E3c", (5, 9, 1024), 9),
    ):
        payload, _, got_budgets = _arm_payload(arm, data, None)
        assert tuple(payload.shape) == shape and got_budgets == budgets
    payload, _, _ = _arm_payload("E4", data, 0.3)
    assert tuple(payload.shape) == (5, 2, 1024)
