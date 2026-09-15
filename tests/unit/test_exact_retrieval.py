import pytest
import torch

from hyperdime.retrieval.exact import exact_search


def test_exact_search_matches_brute_force_across_batches() -> None:
    torch.manual_seed(0)
    queries, documents = torch.randn(5, 8), torch.randn(12, 8)
    scores, indices = exact_search(queries, documents, top_n=3, batch_size=2)
    expected_scores, expected_indices = (queries @ documents.T).topk(3, dim=-1)
    assert torch.allclose(scores, expected_scores)
    assert torch.equal(indices, expected_indices)


def test_exact_search_validates_top_n() -> None:
    with pytest.raises(ValueError):
        exact_search(torch.randn(2, 4), torch.randn(3, 4), top_n=4)
