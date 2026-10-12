"""Encoder settings and construction for F2 (research D4).

The settings recorded with every embedding are what decide whether cached vectors can be reused:
model, pinned revision, width, pooling, normalization, input limit, compute and storage dtypes,
backend, role, and, for queries, the instruction.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Protocol

import torch
from torch import Tensor

from hyperdime.contracts.config import ModelConfig
from hyperdime.contracts.validation import ContractError

Backend = Literal["local", "vllm"]
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


class Encoder(Protocol):
    def encode(self, texts: Sequence[str], progress: str = "") -> Tensor: ...

    def encode_queries(self, texts: Sequence[str], instruction: str | None) -> Tensor: ...


def compute_dtype(model: ModelConfig, device: str) -> str:
    """The configured dtype on a GPU; float32 on CPU, where bfloat16 matmuls are slow."""
    return model.dtype if device.startswith("cuda") else "float32"


def encoder_settings(
    model: ModelConfig,
    backend: str,
    role: str,
    instruction: str | None,
    *,
    compute_dtype: str,
) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "model": model.name,
        "revision": model.revision,
        "embedding_dim": model.embedding_dim,
        "pooling": "last_token",
        "normalize": model.normalize,
        "max_length": model.max_length,
        "compute_dtype": compute_dtype,
        "storage_dtype": "float16",
        "backend": backend,
        "role": role,
    }
    if role == "queries":
        settings["instruction"] = instruction
    return settings


def build_encoder(
    model: ModelConfig,
    backend: str,
    device: str,
    batch_size: int,
    base_url: str | None = None,
) -> Encoder:
    """The frozen encoder on the local device, or a client of the shared vLLM server."""
    if backend == "local":
        from hyperdime.embeddings.qwen import QwenEncoder

        return QwenEncoder(
            model_name=model.name,
            revision=model.revision,
            device=device,
            max_length=model.max_length,
            batch_size=batch_size,
            dtype=DTYPES[compute_dtype(model, device)],
        )
    if backend == "vllm":
        if not base_url:
            raise ContractError("the vllm backend needs --base-url or V4_EMBEDDING_BASE_URL")
        from hyperdime.embeddings.remote import VllmEncoder

        return VllmEncoder(base_url, max_length=model.max_length, batch_size=batch_size)
    raise ContractError(f"unknown embedding backend {backend!r}; use 'local' or 'vllm'")
