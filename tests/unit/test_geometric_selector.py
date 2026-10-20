"""The selector with a geometric prior: fixed (S2), weighted (S3), weighted per budget (S4)."""

from pathlib import Path

import pytest
import torch

from hyperdime.baselines.learning_to_select import (
    BUDGETS,
    BudgetGeometricSelector,
    GeometricSelector,
    LinearSelector,
)
from hyperdime.baselines.methods import variance_importance
from hyperdime.contracts.manifests import ArtifactWriter, start_manifest
from hyperdime.selection.topk import top_k_mask
from hyperdime.training.selection import load_selector, save_selector

GEN = torch.Generator().manual_seed(0)
VARIANCE = torch.rand(1024, generator=GEN) * 1e-3
QUERIES = torch.nn.functional.normalize(torch.randn(6, 1024, generator=GEN), dim=-1)


def test_an_untrained_geometric_selector_is_the_variance_method() -> None:
    model = GeometricSelector(VARIANCE)
    for k in (102, 307, 921):
        assert torch.equal(
            top_k_mask(model(QUERIES), k), top_k_mask(variance_importance(QUERIES, VARIANCE), k)
        )


def test_its_only_learned_parameters_are_the_linear_map() -> None:
    names = [name for name, _ in GeometricSelector(VARIANCE).named_parameters()]
    assert names == ["projection.weight", "projection.bias"]


def _save(model: LinearSelector, root: Path) -> Path:
    with ArtifactWriter(root, "selectors/toy", start_manifest("train-selector", {}, seed=0)) as w:
        save_selector(w, "selector", model, kind="per_domain", environment="toy",
                      training_manifest_sha256="ab" * 32)  # fmt: skip
        return w.commit()


def test_a_saved_geometric_selector_reloads_with_its_prior(tmp_path: Path) -> None:
    model = GeometricSelector(VARIANCE)
    torch.nn.init.normal_(model.projection.weight, generator=GEN)
    _, loaded = load_selector(_save(model, tmp_path), "selector")
    assert isinstance(loaded, GeometricSelector)
    assert torch.equal(loaded(QUERIES), model.eval()(QUERIES))


def test_an_l2s_checkpoint_still_loads_as_a_linear_selector(tmp_path: Path) -> None:
    _, loaded = load_selector(_save(LinearSelector(), tmp_path), "selector")
    assert type(loaded) is LinearSelector


def test_an_untrained_weighted_selector_is_the_variance_method_with_one_more_parameter() -> None:
    model = GeometricSelector(VARIANCE, learn_weight=True)
    assert torch.equal(
        top_k_mask(model(QUERIES), 307), top_k_mask(variance_importance(QUERIES, VARIANCE), 307)
    )
    names = [name for name, _ in model.named_parameters()]
    assert names == ["prior_weight", "projection.weight", "projection.bias"]
    assert model.prior_weight.item() == 1.0


def test_a_zero_prior_weight_leaves_the_linear_map_alone() -> None:
    model = GeometricSelector(VARIANCE, learn_weight=True)
    torch.nn.init.normal_(model.projection.weight, generator=GEN)
    with torch.no_grad():
        model.prior_weight.zero_()
    linear = LinearSelector()
    linear.load_state_dict(
        {k: v for k, v in model.state_dict().items() if k.startswith("projection.")}
    )
    assert torch.allclose(model(QUERIES), linear(QUERIES), atol=1e-6)


def test_a_saved_weighted_selector_reloads_with_its_weight(tmp_path: Path) -> None:
    model = GeometricSelector(VARIANCE, learn_weight=True)
    with torch.no_grad():
        model.prior_weight.fill_(0.25)
    _, loaded = load_selector(_save(model, tmp_path), "selector")
    assert isinstance(loaded, GeometricSelector) and loaded.prior_weight.item() == 0.25
    assert torch.equal(loaded(QUERIES), model.eval()(QUERIES))


def test_an_unweighted_geometric_selector_has_no_prior_weight(tmp_path: Path) -> None:
    _, loaded = load_selector(_save(GeometricSelector(VARIANCE), tmp_path), "selector")
    assert type(loaded) is GeometricSelector and "prior_weight" not in loaded.state_dict()


def test_an_untrained_budget_selector_is_the_variance_method_at_every_budget() -> None:
    model = BudgetGeometricSelector(VARIANCE)
    reference = variance_importance(QUERIES, VARIANCE)
    for k in BUDGETS:
        assert torch.equal(top_k_mask(model(QUERIES, k), k), top_k_mask(reference, k))


def test_a_budget_selector_adds_one_weight_per_budget_and_describes_its_budgets() -> None:
    model = BudgetGeometricSelector(VARIANCE)
    shapes = {name: tuple(p.shape) for name, p in model.named_parameters()}
    assert shapes == {
        "prior_weights": (9,), "projection.weight": (1024, 1024), "projection.bias": (1024,)
    }  # fmt: skip
    assert model.prior_weights.tolist() == [1.0] * 9
    assert model.budgets.dtype == torch.int64 and tuple(model.budgets.tolist()) == BUDGETS
    assert set(dict(model.named_buffers())) == {"variance", "budgets"}


def test_with_equal_weights_it_scores_as_the_weighted_selector() -> None:
    model = BudgetGeometricSelector(VARIANCE)
    weighted = GeometricSelector(VARIANCE, learn_weight=True)
    torch.nn.init.normal_(weighted.projection.weight, std=0.01, generator=GEN)
    with torch.no_grad():
        weighted.prior_weight.fill_(0.5)
        model.prior_weights.fill_(0.5)
    model.load_state_dict(
        {**model.state_dict(), **{k: v for k, v in weighted.state_dict().items() if "proj" in k}}
    )
    for k in BUDGETS:
        assert torch.equal(model(QUERIES, k), weighted(QUERIES))


def test_a_budgets_weight_changes_only_that_budgets_mask() -> None:
    # On D = 4 with variance 1 and no correction, a weight scales log(q²) and keeps its order;
    # with a correction b = (0, 0, 0, 3), a weight 0 lets the correction alone pick dim 3.
    model = BudgetGeometricSelector(torch.ones(4), budgets=(1, 2))
    queries = torch.tensor([[0.9, 0.5, 0.3, 0.1]])
    with torch.no_grad():
        model.projection.bias.copy_(torch.tensor([0.0, 0.0, 0.0, 3.0]))
    before = {k: top_k_mask(model(queries, k), k) for k in (1, 2)}
    assert before[1].nonzero()[:, 1].tolist() == [0]  # log 0.81 + 0 beats log 0.01 + 3
    with torch.no_grad():
        model.prior_weights[0] = 0.0
    assert top_k_mask(model(queries, 1), 1).nonzero()[:, 1].tolist() == [3]
    assert torch.equal(top_k_mask(model(queries, 2), 2), before[2])


@pytest.mark.parametrize("k", [None, 0, 103, 1024])
def test_a_budget_outside_its_grid_is_refused(k: int | None) -> None:
    with pytest.raises(ValueError, match="is not one of"):
        BudgetGeometricSelector(VARIANCE)(QUERIES, k)


@pytest.mark.parametrize("budgets", [(), (204, 102), (102, 102)])
def test_budgets_must_be_strictly_increasing(budgets: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        BudgetGeometricSelector(VARIANCE, budgets)


def test_a_saved_budget_selector_reloads_with_its_weights_and_budgets(tmp_path: Path) -> None:
    model = BudgetGeometricSelector(VARIANCE)
    with torch.no_grad():
        model.prior_weights.copy_(torch.linspace(0.1, 0.9, 9))
    _, loaded = load_selector(_save(model, tmp_path), "selector")
    assert isinstance(loaded, BudgetGeometricSelector)
    assert torch.equal(loaded.prior_weights, model.prior_weights)
    for k in BUDGETS:
        assert torch.equal(loaded(QUERIES, k), model.eval()(QUERIES, k))
