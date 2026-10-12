"""E0 importance methods (F3 US3)."""

import torch

from hyperdime.baselines.learning_to_select import LinearSelector
from hyperdime.baselines.methods import (
    global_importance,
    oracle_scores,
    prefix_importance,
    random_importance,
    selector_importance,
)
from hyperdime.selection.topk import top_k_mask

QUERIES = torch.randn(4, 1024, generator=torch.Generator().manual_seed(0))


def test_every_method_selects_exactly_k() -> None:
    mean = torch.softmax(torch.randn(1024), 0)
    for scores in (
        prefix_importance(QUERIES),
        random_importance(QUERIES),
        global_importance(QUERIES, mean),
        selector_importance(QUERIES, LinearSelector()),
        oracle_scores(QUERIES, torch.randn(4, 1024), torch.randn(4, 1024)),
    ):
        assert top_k_mask(scores, 307).sum(1).tolist() == [307] * 4


def test_prefix_keeps_the_first_k_dimensions() -> None:
    assert top_k_mask(prefix_importance(QUERIES), 3)[0].nonzero().flatten().tolist() == [0, 1, 2]


def test_random_is_reproducible_by_seed() -> None:
    assert torch.equal(random_importance(QUERIES, 1), random_importance(QUERIES, 1))
    assert not torch.equal(random_importance(QUERIES, 1), random_importance(QUERIES, 2))


def test_global_importance_is_shared_by_all_queries() -> None:
    mean = torch.softmax(torch.randn(1024), 0)
    masks = top_k_mask(global_importance(QUERIES, mean), 10)
    assert all(torch.equal(masks[0], row) for row in masks)
