import pytest
import torch

from hyperdime.embeddings.qwen import format_query, last_token_pool


def test_format_query_uses_qwen_instruction_template() -> None:
    assert format_query("claim", "task") == "Instruct: task\nQuery:claim"
    assert format_query("claim", None) == "claim"


def test_last_token_pool_handles_right_and_left_padding() -> None:
    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    right_padded = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    left_padded = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    assert torch.equal(last_token_pool(hidden, right_padded), hidden[[0, 1], [1, 3]])
    assert torch.equal(last_token_pool(hidden, left_padded), hidden[[0, 1], [3, 3]])


def test_last_token_pool_rejects_misaligned_mask() -> None:
    with pytest.raises(ValueError):
        last_token_pool(torch.zeros(2, 4, 3), torch.ones(2, 3))
