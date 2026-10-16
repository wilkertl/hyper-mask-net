"""The selector with a fixed geometric prior (S2)."""

from pathlib import Path

import torch

from hyperdime.baselines.learning_to_select import GeometricSelector, LinearSelector
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
    assert "prior_weight" not in loaded.state_dict()
