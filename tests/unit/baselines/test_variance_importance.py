"""The label-free geometric importance q_d^2 * Var(x_d) (P1)."""

import torch

from hyperdime.baselines.methods import variance_importance
from hyperdime.selection.topk import apply_mask, top_k_mask


def test_importance_is_the_squared_query_times_the_corpus_variance() -> None:
    queries = torch.tensor([[1.0, -2.0, 3.0]])
    found = variance_importance(queries, torch.tensor([0.5, 1.0, 0.0]))
    assert found.tolist() == [[0.5, 4.0, 0.0]]


def test_a_constant_dimension_is_dropped_and_the_ranking_kept() -> None:
    documents = torch.tensor([[5.0, 0.1], [5.0, 0.9], [5.0, 0.4]])  # dim 0 is a constant offset
    query = torch.tensor([[0.9, 0.3]])
    mask = top_k_mask(variance_importance(query, documents.var(0, correction=0)), 1)
    assert mask.tolist() == [[False, True]]
    masked = apply_mask(query, mask) @ documents.T
    assert (
        masked.argsort(descending=True).tolist()
        == (query @ documents.T).argsort(descending=True).tolist()
    )
