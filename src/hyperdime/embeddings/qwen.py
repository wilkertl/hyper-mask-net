"""Qwen3-Embedding query-formatting and pooling conventions, and a frozen encoder."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

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


class QwenEncoder:
    """Frozen Qwen3-Embedding encoder producing L2-normalized ``(N, 1024)`` float32 vectors."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        revision: str | None = None,
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 16,
        dtype: torch.dtype = torch.float32,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        """Load the model and a left-padding tokenizer, unless both are injected (tests)."""
        self.device, self.max_length, self.batch_size = device, max_length, batch_size
        self.dtype = dtype
        if model is None or tokenizer is None:
            from transformers import AutoModel, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                model_name, revision=revision, padding_side="left"
            )
            model = AutoModel.from_pretrained(model_name, revision=revision, dtype=dtype)
        self.tokenizer: Any = tokenizer
        self.model: Any = model.to(device)
        self.model.eval().requires_grad_(False)
        self.revision = str(getattr(self.model.config, "_commit_hash", None) or revision)

    @property
    def settings(self) -> dict[str, Any]:
        return {
            "backend": "local",
            "model": f"{MODEL_NAME}@{self.revision}",
            "max_length": self.max_length,
            "dtype": str(self.dtype),
        }

    def encode_queries(self, texts: Sequence[str], instruction: str | None) -> Tensor:
        return self.encode([format_query(text, instruction) for text in texts])

    def encode(self, texts: Sequence[str], progress: str = "") -> Tensor:
        """Encode in length-sorted batches to limit padding; rows keep the input order."""
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        output = torch.empty(len(texts), EMBEDDING_DIM)
        with torch.inference_mode():
            for start in range(0, len(order), self.batch_size):
                rows = order[start : start + self.batch_size]
                batch = self.tokenizer(
                    [texts[i] for i in rows],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**batch).last_hidden_state
                pooled = last_token_pool(hidden, batch["attention_mask"])
                output[rows] = F.normalize(pooled.float(), dim=-1).cpu()
                if progress and (start // self.batch_size) % 50 == 0:
                    print(f"{progress}: {start + len(rows)}/{len(texts)}", file=sys.stderr)
        return output
