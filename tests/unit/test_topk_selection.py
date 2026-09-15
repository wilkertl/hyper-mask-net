import pytest
import torch

from hyperdime.selection.topk import apply_mask, resolve_k, top_k_mask


def test_resolve_k_accepts_counts_and_floors_ratios() -> None:
    assert resolve_k(3, 10) == 3
    assert resolve_k(0.29, 100) == 29
    assert resolve_k(0.1, 1024) == 102
    assert resolve_k(1.0, 1024) == 1024


@pytest.mark.parametrize("k", [0, 11, 0.05])
def test_resolve_k_rejects_out_of_range(k: int | float) -> None:
    with pytest.raises(ValueError):
        resolve_k(k, 10)


def test_resolve_k_rejects_bool() -> None:
    with pytest.raises(TypeError):
        resolve_k(True, 10)


def test_top_k_mask_keeps_highest_importance_coordinates() -> None:
    mask = top_k_mask(torch.tensor([[0.1, 0.8, 0.2, 0.9]]), 2)
    assert mask.tolist() == [[False, True, False, True]]
    masked = apply_mask(torch.tensor([[1.0, 2.0, 3.0, 4.0]]), mask)
    assert torch.equal(masked, torch.tensor([[0.0, 2.0, 0.0, 4.0]]))


def test_full_k_is_equivalent_to_unmasked_scoring() -> None:
    torch.manual_seed(0)
    queries, documents = torch.randn(3, 16), torch.randn(7, 16)
    mask = top_k_mask(torch.rand(3, 16), 16)
    assert torch.allclose(apply_mask(queries, mask) @ documents.T, queries @ documents.T)


def test_masked_scores_use_only_selected_coordinates_and_leave_documents_unchanged() -> None:
    torch.manual_seed(0)
    queries, documents = torch.randn(4, 16), torch.randn(10, 16)
    documents_before = documents.clone()
    mask = top_k_mask(torch.rand(4, 16), 5)
    scores = apply_mask(queries, mask) @ documents.T
    for row in range(4):
        selected = mask[row]
        assert torch.allclose(scores[row], documents[:, selected] @ queries[row, selected])
    assert torch.equal(documents, documents_before)
