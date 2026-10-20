"""The retrieval grid and its validation selection (F5 research D6; F9 research D3–D6)."""

import hashlib
from typing import cast

import numpy as np
import pytest
import torch

from hyperdime.baselines.learning_to_select import GeometricSelector, LinearSelector
from hyperdime.cli.train_selector import RECIPES, SEEDED_RECIPES, TRACK_INITS
from hyperdime.training.examples import RetrievalExamples
from hyperdime.training.retrieval_selection import (
    RetrievalCell,
    TrainedRetrieval,
    best_retrieval,
    candidate_queries,
    choose_candidates,
    family_bests,
    retrieval_cells,
    retrieval_summary,
    train_retrieval_cells,
)
from hyperdime.training.retrieval_trainer import BUDGETS, HEADLINE_K, soft_mass
from hyperdime.training.selection import Supervision


def test_the_grid_has_32_cells_in_product_order_with_distinct_keys() -> None:
    cells = retrieval_cells()
    assert len({c.key for c in cells}) == 32
    assert cells[0].key == "fixed-307-scratch-lr0.0001-ts0.02-b0.1"
    assert cells[-1].key == "multi-budget-l2s-lr0.001-ts0.05-b0.02"


@pytest.mark.parametrize(
    ("variant", "budgets"), [("fixed-307", (HEADLINE_K,)), ("multi-budget", BUDGETS)]
)
def test_a_cell_trains_on_its_variants_budgets(variant: str, budgets: tuple[int, ...]) -> None:
    cell = RetrievalCell(variant, "scratch", 1e-3, 0.05, 0.1)
    assert cell.config(seed=2, device="cpu").budgets == budgets


def _trained(variant: str, init: str, mean: float, mass: float = 1.0) -> TrainedRetrieval:
    cell = RetrievalCell(variant, init, 1e-3, 0.05, 0.1)
    return TrainedRetrieval(cell, LinearSelector(4), [{"soft_mass": mass}], {102: mean, 204: mean})


def test_selection_keeps_the_best_cell_whose_soft_mask_passes_the_guard() -> None:
    cells = [
        _trained("fixed-307", "scratch", 0.5),
        _trained("fixed-307", "l2s", 0.9, mass=0.5),
        _trained("multi-budget", "scratch", 0.6),
        _trained("multi-budget", "l2s", 0.6),
    ]
    assert best_retrieval(cells) is cells[2]
    assert {f: t.mean for f, t in family_bests(cells).items()} == {
        "fixed-307-scratch": 0.5,
        "multi-budget-scratch": 0.6,
        "multi-budget-l2s": 0.6,
    }


def test_selection_refuses_when_no_cell_passes_the_guard() -> None:
    with pytest.raises(ValueError, match="research D3"):
        best_retrieval([_trained("fixed-307", "scratch", 0.5, mass=0.2)])


def test_the_s2_grid_has_16_cells_that_all_start_from_the_prior() -> None:
    cells = retrieval_cells(inits=("prior",))
    assert len(cells) == 16 and {c.init for c in cells} == {"prior"}


def test_a_prior_cell_is_refused_without_a_prior() -> None:
    cell = RetrievalCell("fixed-307", "prior", 1e-3, 0.05, 0.1, steps=1)
    with pytest.raises(ValueError, match="geometric prior"):
        train_retrieval_cells([cell], cast(RetrievalExamples, None), cast(RetrievalExamples, None))


def test_the_s3_grid_has_16_weighted_cells() -> None:
    cells = retrieval_cells(inits=("weighted",))
    assert len(cells) == 16 and {c.init for c in cells} == {"weighted"}


def test_a_weighted_cell_is_refused_without_a_prior() -> None:
    cell = RetrievalCell("fixed-307", "weighted", 1e-3, 0.05, 0.1, steps=1)
    with pytest.raises(ValueError, match="geometric prior"):
        train_retrieval_cells([cell], cast(RetrievalExamples, None), cast(RetrievalExamples, None))


@pytest.mark.parametrize("init", ["budget", "weighted-aligned"])
def test_each_s4_track_has_8_multi_budget_cells_under_the_aligned_rules(init: str) -> None:
    cells = retrieval_cells(inits=(init,))
    assert len(cells) == 8 and {(c.variant, c.init) for c in cells} == {("multi-budget", init)}
    config = cells[0].config(seed=0, device="cpu")
    assert (config.checkpoint_rule, config.nonnegative_prior) == ("validation", True)
    assert config.budgets == BUDGETS


def test_earlier_tracks_keep_the_rerank_rule() -> None:
    for init in ("scratch", "l2s", "prior", "weighted"):
        config = RetrievalCell("multi-budget", init, 1e-3, 0.05, 0.1).config(seed=0, device="cpu")
        assert (config.checkpoint_rule, config.nonnegative_prior) == ("rerank", False)


def test_the_candidate_subset_is_the_smallest_hashes_whatever_the_row_order() -> None:
    ids = [f"q{i}" for i in range(50)]
    rows = candidate_queries(ids, size=10)
    expected = sorted(ids, key=lambda q: hashlib.sha256(f"0:{q}".encode()).hexdigest())[:10]
    assert [ids[r] for r in rows] == expected
    shuffled = ids[::-1]
    assert [shuffled[r] for r in candidate_queries(shuffled, size=10)] == expected
    assert sorted(candidate_queries(ids[:7], size=10)) == list(range(7))
    assert len(candidate_queries([f"q{i}" for i in range(1500)])) == 1000


# On the full corpus, keeping dimension 0 ranks each query's relevant document first and keeping
# dimension 1 ranks a distractor first; a bias of 1 on a dimension decides which is kept at k = 1.
DIM = 1024
QIDS = ["q0", "q1", "q2"]
QUERIES = torch.zeros(3, DIM)
QUERIES[:, :2] = 1.0
DOCS = np.zeros((2, DIM), dtype=np.float32)
DOCS[0, 0], DOCS[1, 1] = 1.0, 2.0
VALIDATION_ONLY = Supervision(
    (torch.empty(0), torch.empty(0), torch.empty(0)),
    (QUERIES, torch.empty(0), torch.empty(0)),
    QIDS,
    {q: {"rel": 1} for q in QIDS},
)
GOOD, BAD = 0, 1


def _state(keep: int, weight: float = 1.0) -> dict[str, torch.Tensor]:
    model = GeometricSelector(torch.ones(DIM), learn_weight=True)
    with torch.no_grad():
        model.projection.bias[keep] = 1.0
        model.prior_weight.fill_(weight)
    return {k: v.clone() for k, v in model.state_dict().items()}


def _aligned(steps_keep: list[tuple[int, int]], rerank_best: int) -> TrainedRetrieval:
    cell = RetrievalCell("multi-budget", "weighted-aligned", 1e-3, 0.05, 0.1)
    history = [
        {"step": float(step), "val_rerank_ndcg10": 1.0 if step == rerank_best else 0.5,
         "soft_mass": 0.123, "projected_steps": 2.0}
        for step, _ in steps_keep
    ]  # fmt: skip
    candidates = [(step, _state(keep)) for step, keep in steps_keep]
    model = GeometricSelector(torch.ones(DIM), learn_weight=True)
    return TrainedRetrieval(cell, model, history, candidates=candidates)


def _choose(trained: list[TrainedRetrieval]) -> list[str]:
    return choose_candidates(
        trained, VALIDATION_ONLY, DOCS, ["rel", "bad"], environment="toy", budgets=(1,)
    )


def test_the_aligned_rule_keeps_the_full_corpus_best_not_the_rerank_best() -> None:
    trained = _aligned([(0, BAD), (10, GOOD), (20, BAD)], rerank_best=20)
    assert sorted(_choose([trained])) == QIDS
    assert trained.kept_step == 10 and trained.candidates == []
    assert trained.candidate_scores[10] == pytest.approx(1.0)
    assert trained.candidate_scores[20] == pytest.approx(1 / np.log2(3))
    assert torch.equal(trained.model.projection.bias, _state(GOOD)["projection.bias"])


@pytest.mark.parametrize(
    ("steps_keep", "kept"), [([(0, GOOD), (10, BAD)], 0), ([(0, GOOD), (10, GOOD)], 0)]
)
def test_step_0_is_kept_when_best_and_ties_go_to_the_earlier_step(
    steps_keep: list[tuple[int, int]], kept: int
) -> None:
    trained = _aligned(steps_keep, rerank_best=10)
    _choose([trained])
    assert trained.kept_step == kept


def test_the_guard_reads_the_kept_states_soft_mass_at_the_final_temperature() -> None:
    trained = _aligned([(0, BAD), (10, GOOD)], rerank_best=0)
    with pytest.raises(ValueError, match="choose_candidates"):
        _ = trained.soft_mass
    _choose([trained])
    expected = soft_mass(trained.model, QUERIES, (1,), beta=0.1)
    assert trained.soft_mass == pytest.approx(expected) and trained.soft_mass != 0.123
    assert _trained("multi-budget", "weighted", 0.5, mass=0.4).soft_mass == 0.4


def test_the_selection_record_carries_the_aligned_rules_additions() -> None:
    trained = _aligned([(0, BAD), (10, GOOD)], rerank_best=0)
    _choose([trained])
    trained.scores = {1: 1.0}
    entry = retrieval_summary([trained])[trained.cell.key]
    assert entry["checkpoint_rule"] == "validation" and entry["kept_step"] == 10
    assert entry["candidate_scores"]["0"]["val_rerank_ndcg10"] == 1.0
    assert entry["candidate_scores"]["10"]["ndcg@10"] == pytest.approx(1.0)
    assert entry["prior_weights"] == [1.0] and entry["projected_steps"] == 2
    unaligned = _trained("multi-budget", "weighted", 0.5)
    unaligned.history = [{"step": 0.0, "val_rerank_ndcg10": 0.5, "soft_mass": 1.0}]
    assert "kept_step" not in retrieval_summary([unaligned])[unaligned.cell.key]


def test_the_s4_recipes_resolve_to_their_tracks() -> None:
    assert {"budget-grid", "budget-seeded", "weighted-aligned-grid",
            "weighted-aligned-seeded"} <= set(RECIPES)  # fmt: skip
    assert {"budget-seeded", "weighted-aligned-seeded"} <= set(SEEDED_RECIPES)
    assert TRACK_INITS["budget"] == ("budget",)
    assert TRACK_INITS["weighted-aligned"] == ("weighted-aligned",)
    assert "weighted-aligned-grid".removesuffix("-grid") in TRACK_INITS
