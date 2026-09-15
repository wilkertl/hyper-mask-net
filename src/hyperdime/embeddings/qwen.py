"""Qwen3-Embedding query-formatting and pooling conventions."""

from __future__ import annotations

import torch
from torch import Tensor

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_DIM = 1024


def format_query(query: str, instruction: str | None) -> str:
    """Prefix a query with its task instruction; documents are encoded without formatting."""
    return f"Instruct: {instruction}\nQuery:{query}" if instruction else query


def last_token_pool(hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Select each sequence's last attended token, for left- or right-padded batches."""
    if hidden_states.ndim != 3 or attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError("expected hidden_states [B, T, H] and attention_mask [B, T].")
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    last = (attention_mask.long() * positions).argmax(dim=1)
    rows = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[rows, last]
