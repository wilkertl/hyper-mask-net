"""QwenEncoder with an injected stub model and tokenizer (no weights, no network)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor

from hyperdime.embeddings.qwen import MODEL_NAME, QwenEncoder


class StubBatch(dict[str, Tensor]):
    def to(self, device: str) -> StubBatch:
        return self


class StubTokenizer:
    """One token per text, whose id is the text's length."""

    def __call__(self, texts: list[str], **kwargs: Any) -> StubBatch:
        ids = torch.tensor([[len(text)] for text in texts])
        return StubBatch(input_ids=ids, attention_mask=torch.ones_like(ids))


class StubModel:
    """Hidden state of weight 3 on coordinate ``token id % 1024``."""

    config = SimpleNamespace(_commit_hash="abc123")

    def to(self, device: str) -> StubModel:
        return self

    def eval(self) -> StubModel:
        return self

    def requires_grad_(self, flag: bool) -> StubModel:
        return self

    def __call__(self, input_ids: Tensor, attention_mask: Tensor) -> SimpleNamespace:
        hidden = torch.zeros(input_ids.shape[0], input_ids.shape[1], 1024)
        hidden[torch.arange(input_ids.shape[0]), -1, input_ids[:, -1] % 1024] = 3.0
        return SimpleNamespace(last_hidden_state=hidden)


def encoder() -> QwenEncoder:
    return QwenEncoder(model=StubModel(), tokenizer=StubTokenizer(), batch_size=2)


def test_encode_keeps_input_order_across_length_sorted_batches_and_normalizes() -> None:
    vectors = encoder().encode(["ccc", "a", "bb"])
    assert vectors.shape == (3, 1024) and vectors.dtype == torch.float32
    assert torch.allclose(vectors.norm(dim=-1), torch.ones(3))
    assert (vectors[0, 3], vectors[1, 1], vectors[2, 2]) == (1.0, 1.0, 1.0)


def test_queries_are_encoded_with_the_qwen_instruction_format() -> None:
    formatted = "Instruct: I\nQuery:q"
    assert encoder().encode_queries(["q"], "I")[0, len(formatted)] == 1.0


def test_settings_identify_the_backend_model_revision_and_limits() -> None:
    assert encoder().settings == {
        "backend": "local",
        "model": f"{MODEL_NAME}@abc123",
        "max_length": 512,
        "dtype": "torch.float32",
    }
