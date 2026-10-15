"""The retrieval grid and its validation selection (F5 research D6)."""

from typing import cast

import pytest

from hyperdime.baselines.learning_to_select import LinearSelector
from hyperdime.training.examples import RetrievalExamples
from hyperdime.training.retrieval_selection import (
    RetrievalCell,
    TrainedRetrieval,
    best_retrieval,
    family_bests,
    retrieval_cells,
    train_retrieval_cells,
)
from hyperdime.training.retrieval_trainer import BUDGETS, HEADLINE_K


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
