import pytest
import torch

from hyperdime.oracle.importance import oracle_importance, oracle_scores

QUERY = torch.tensor([[1.0, -2.0, 0.5]])
POSITIVE = torch.tensor([[0.5, 0.0, 1.0]])
NEGATIVE = torch.tensor([[0.0, 1.0, 1.0]])


def test_oracle_scores_match_manual_computation() -> None:
    expected = torch.tensor([[0.5, 2.0, 0.0]])
    assert torch.allclose(oracle_scores(QUERY, POSITIVE, NEGATIVE), expected)


def test_oracle_importance_is_a_tempered_distribution() -> None:
    target = oracle_importance(QUERY, POSITIVE, NEGATIVE, temperature=0.5)
    assert torch.allclose(target, torch.softmax(torch.tensor([[1.0, 4.0, 0.0]]), dim=-1))
    assert torch.allclose(target.sum(dim=-1), torch.ones(1))
    assert not target.isnan().any()


def test_oracle_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        oracle_importance(QUERY, POSITIVE, NEGATIVE, temperature=0.0)
    with pytest.raises(ValueError):
        oracle_scores(QUERY, POSITIVE[:, :2], NEGATIVE)
