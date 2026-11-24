"""ADR-0020 objective study: encodings, losses, ranking, statistics, scoring (CPU)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from hyperdime.cli.p3_objective import column_ranks, measures
from hyperdime.evaluation.benchmark import positive_ranks, query_measures
from hyperdime.evaluation.objective_study import (
    TRAINED,
    ArmData,
    arm_payload,
    contrast,
    graded_labels,
    membership_loss,
    parameters,
    ranking_loss,
    reading,
    relaxed_membership,
    select_lr,
)
from hyperdime.evaluation.target_study import (
    Selector,
    kl_rows,
    loss_membership,
    mask_average,
    top_k_mask,
    train,
)
from hyperdime.training.examples import retrieval_examples
from hyperdime.training.losses import masked_retrieval_loss


def test_relaxed_membership_is_half_exactly_at_the_hard_top_k() -> None:
    scores = torch.tensor([[3.0, 1.0, 2.0, 0.0]])
    soft = relaxed_membership(scores, (2,), beta=0.5)[:, 0]
    # standardized: mean 1.5, std sqrt(1.25); threshold midway between 2 and 1 → raw 1.5 → z = 0
    z = (scores - 1.5) / (math.sqrt(1.25) + 1e-6)
    assert torch.allclose(soft, torch.sigmoid(z / 0.5))
    assert (soft > 0.5).tolist() == [[True, False, True, False]]


def test_membership_loss_with_binary_labels_equals_the_adr0019_e2_loss() -> None:
    torch.manual_seed(0)
    model, x = Selector(8), torch.randn(5, 8)
    sizes = (2, 4, 6)
    masks = torch.stack([top_k_mask(torch.randn(5, 8), k) for k in sizes], 1)
    expected = loss_membership(model, x, masks, 0.3, sizes)
    assert torch.allclose(membership_loss(model, x, masks, 0.3, sizes), expected)


def test_graded_labels_tend_to_the_binary_masks_as_beta_shrinks() -> None:
    torch.manual_seed(1)
    log_target = torch.log_softmax(torch.randn(4, 16), -1)
    sizes = (3, 8)
    labels = graded_labels(log_target, sizes, beta=1e-4)
    binary = torch.stack([top_k_mask(log_target, k) for k in sizes], 1)
    assert torch.equal(labels > 0.5, binary)
    assert torch.allclose(labels, binary.float(), atol=1e-6)
    assert not labels.requires_grad


def test_nested_masks_are_recovered_exactly_from_their_average() -> None:
    torch.manual_seed(2)
    source = torch.randn(6, 20)
    sizes = (2, 5, 9, 14)
    masks = torch.stack([top_k_mask(source, k) for k in sizes], 1)
    average = mask_average(masks)
    for b, k in enumerate(sizes):
        assert torch.equal(top_k_mask(average, k), masks[:, b])


def test_kl_and_cross_entropy_differ_only_by_the_target_entropy() -> None:
    target = torch.tensor([[0.5, 0.25, 0.25]], dtype=torch.float64)
    entropy = -(target * target.log()).sum()
    for logits in (torch.tensor([[0.0, 1.0, 2.0]]), torch.tensor([[3.0, -1.0, 0.5]])):
        log_p = torch.log_softmax(logits.double(), -1)
        cross_entropy = -(target * log_p).sum()
        assert torch.allclose(kl_rows(target, log_p).sum(), cross_entropy - entropy)


def _examples(n: int = 6, dim: int = 8) -> tuple[torch.Tensor, object]:
    rng = np.random.default_rng(3)
    queries = rng.standard_normal((n, dim)).astype(np.float32)
    documents = rng.standard_normal((12, dim)).astype(np.float32)
    ids = [f"q{i}" for i in range(n)]
    doc_ids = [f"d{j}" for j in range(12)]
    judgments = {q: {f"d{i}": 1, f"d{(i + 1) % 12}": 2} for i, q in enumerate(ids)}
    pools = {q: [f"d{(i + 5) % 12}", f"d{(i + 7) % 12}"] for i, q in enumerate(ids)}
    examples = retrieval_examples(
        ids,
        queries,
        {q: i for i, q in enumerate(ids)},
        documents,
        {d: j for j, d in enumerate(doc_ids)},
        judgments,
        pools,
    )
    return torch.from_numpy(queries), examples


def test_ranking_loss_at_one_budget_is_the_s1_masked_retrieval_loss() -> None:
    from hyperdime.training.retrieval_trainer import _gather

    torch.manual_seed(4)
    x, examples = _examples()
    model = Selector(8)
    rows = torch.tensor([0, 2, 3])
    found = ranking_loss(model, rows, examples, (3,), beta=0.4)  # type: ignore[arg-type]
    queries, documents, positives, weights, negatives = _gather(
        examples,  # type: ignore[arg-type]
        rows,
        torch.device("cpu"),
    )
    soft = relaxed_membership(model.logits(queries), (3,), 0.4)[:, 0]
    expected = masked_retrieval_loss(queries, soft, documents, positives, weights, negatives, 0.05)
    assert torch.allclose(found, expected)
    assert torch.equal(queries, x[rows])


def _arm_data() -> ArmData:
    x, examples = _examples()
    sizes = (2, 4, 6)
    source = torch.randn(6, 8, generator=torch.Generator().manual_seed(5))
    masks = torch.stack([top_k_mask(source, k) for k in sizes], 1)
    official = torch.softmax(source / 0.5, -1)
    return ArmData(x, official, masks, examples, sizes)  # type: ignore[arg-type]


@pytest.mark.parametrize("arm", TRAINED)
def test_every_arm_trains_one_epoch_with_a_finite_loss(arm: str) -> None:
    data = _arm_data()
    payload, loss, budgets = arm_payload(arm, data)
    assert payload.shape[0] == data.x.shape[0]
    assert budgets == (3 if arm in ("KBk", "KBa") else 0)
    result = train(data.x, payload, loss, epochs=1, seed=0, device="cpu", budgets=budgets)
    assert math.isfinite(result.final_loss)
    assert result.steps == 1


def test_a_start_state_is_kept_at_zero_learning_rate() -> None:
    data = _arm_data()
    start = {"net.0.weight": torch.eye(8), "net.0.bias": torch.zeros(8)}
    payload, loss, budgets = arm_payload("RI", data)
    result = train(data.x, payload, loss, epochs=2, seed=0, device="cpu", lr=0.0, start=start)
    assert torch.equal(result.model.net[0].weight, torch.eye(8))
    with pytest.raises(ValueError):
        train(data.x, payload, loss, epochs=1, seed=0, device="cpu", start={"bad": torch.ones(1)})


def test_contrast_pairs_seeds_and_queries() -> None:
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.zeros((2, 2))
    rng = np.random.Generator(np.random.PCG64(0))
    result = contrast(a, b, rng, family=10, draws=1000)
    assert result["D"] == pytest.approx(2.5)  # seed-mean diffs [2, 3]
    assert result["spread"] == pytest.approx(2.0)  # per-seed means 1.5 and 3.5
    assert result["per_seed"] == [1.5, 3.5]
    assert result["bonferroni_low"] <= result["low"] <= 2.5 <= result["high"]
    with pytest.raises(ValueError):
        contrast(a, np.zeros((3, 2)), rng, family=10, draws=10)


def _r(d: float, spread: float, low: float, high: float) -> dict[str, float]:
    return {"D": d, "spread": spread, "bonferroni_low": low, "bonferroni_high": high}


def test_reading_rule() -> None:
    supported = {"a": _r(0.02, 0.01, 0.005, 0.03), "b": _r(0.01, 0.005, -0.01, 0.03)}
    assert reading(supported) == "supported"
    assert reading({"a": _r(0.02, 0.01, 0.005, 0.03), "b": _r(-0.01, 0.0, -0.02, 0.0)}) == "mixed"
    small = {"a": _r(0.02, 0.03, 0.005, 0.03), "b": _r(0.01, 0.005, 0.001, 0.03)}
    assert reading(small) == "not separated"
    wide = {"a": _r(0.02, 0.01, -0.005, 0.03), "b": _r(0.01, 0.005, -0.01, 0.03)}
    assert reading(wide) == "not separated"


def test_select_lr_breaks_ties_to_the_smaller_rate() -> None:
    assert select_lr({1e-4: 0.5, 3e-4: 0.6, 1e-3: 0.6}) == 3e-4
    assert select_lr({3e-5: 0.4, 1e-4: 0.3}) == 3e-5


def test_parameter_counts() -> None:
    assert parameters("KG") == 1_049_600
    assert parameters("KBk") == 1_049_600 + 9 * 1024


def test_column_ranks_and_measures_match_the_benchmark_helpers() -> None:
    torch.manual_seed(6)
    scores = torch.randint(0, 5, (40, 7)).to(torch.float64)  # many ties
    positives = torch.tensor([3, 17, 30])
    gains = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float64)
    ranks = column_ranks(scores, positives, chunk=3)
    from hyperdime.evaluation.benchmark import ideal_dcg

    found = measures(ranks, gains, ideal_dcg(gains.tolist()))
    for m in range(scores.shape[1]):
        expected_ranks = positive_ranks(scores[:, m], positives)
        assert torch.equal(ranks[:, m], expected_ranks)
        expected = query_measures(expected_ranks, gains)
        assert found["ndcg10"][m].item() == pytest.approx(expected["ndcg@10"])
        assert found["mrr10"][m].item() == pytest.approx(expected["mrr@10"])
        assert found["recall100"][m].item() == pytest.approx(expected["recall@100"])


def test_corpus_results_and_render_from_synthetic_rows() -> None:
    from hyperdime.cli.p3_objective import _render, corpus_results
    from hyperdime.evaluation.benchmark import DECILES

    rng = np.random.default_rng(7)
    arms = ["REF", *TRAINED]
    selection = {"arms": {a: {"selected_lr": 0.0003} for a in TRAINED}}
    rows = []
    for half in "AB":
        for q in range(6):
            for arm in arms:
                lr = "0.0001" if arm == "REF" else "0.0003"
                for seed in ("42", "0", "1"):
                    for k in DECILES:
                        value = 0.5 if k == 1024 else float(rng.uniform())
                        rows.append(
                            {
                                "half": half,
                                "query": f"{half}{q}",
                                "k": str(k),
                                "arm": arm,
                                "lr": lr,
                                "seed": seed,
                                "ndcg10": str(value),
                                "mrr10": str(value),
                                "recall100": str(value),
                            }
                        )
    result = corpus_results(rows, selection)
    assert result["queries"] == 6
    assert result["parity_1024_max_gap"] == 0.0
    assert set(result["contrasts"]) == {
        "C1",
        "C2",
        "C3",
        "C4",
        "C6",
        "C7",
        "C8",
        "C9",
        "C10",
        "C11",
    }
    assert all("holm_p" in c for c in result["contrasts"].values())
    diag = {
        f"{a}|{'0.0001' if a == 'REF' else '0.0003'}|42": {"jaccard": [0.5] * 9, "kl_t_off": 0.1}
        for a in arms
    }
    costs = {a: {"runs": 1, "seconds": 2.0, "devices": ["h GPU-x"]} for a in TRAINED}
    text = _render({"scifact": result}, {"scifact": costs}, {"scifact": diag}, {"scifact": {}})
    assert "| C3 | MS − KS |" in text
    assert "Parity at k = 1024" in text
