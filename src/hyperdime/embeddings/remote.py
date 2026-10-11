"""Qwen3-Embedding-0.6B served by vLLM (OpenAI-compatible ``POST {base_url}/embeddings``).

Same service contract as the ``beir`` project's v4 adapter, so one server, tunnel, and set of
environment variables serve both projects:

* ``V4_EMBEDDING_BASE_URL`` is the endpoint, e.g. ``http://127.0.0.1:8000/v1``;
  ``V4_EMBEDDING_API_KEY`` is sent as a bearer token only if set.
* The endpoint must be ``https://`` with certificate verification, a loopback address (the local
  end of an ``ssh -L`` tunnel), or a host declared in ``V4_EMBEDDING_TUNNEL_HOSTS``.
* A response is used only with exactly one result per input, indices ``0..N-1``, and 1,024
  finite values with nonzero norm per vector; vectors are then L2-normalized to float32.
* Timeouts, connection failures, 408, 429, and 5xx are retried within a bound; anything else
  fails at once. Errors never carry request text, response bodies, or credentials.

Texts are cut to ``max_length`` tokens on the client with the Qwen tokenizer, so results do not
depend on the server's context length or truncation side.
"""

from __future__ import annotations

import ipaddress
import json
import math
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import torch
from torch import Tensor

from hyperdime.embeddings.qwen import EMBEDDING_DIM, MODEL_NAME, format_query

BASE_URL_ENV = "V4_EMBEDDING_BASE_URL"
API_KEY_ENV = "V4_EMBEDDING_API_KEY"
TUNNEL_HOSTS_ENV = "V4_EMBEDDING_TUNNEL_HOSTS"

_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_LOOPBACK_NAMES = frozenset({"localhost", "localhost.localdomain"})

Truncate = Callable[[Sequence[str]], list[str]]


class EndpointConfigurationError(ValueError):
    """The configured endpoint would send text over an unprotected channel."""


class EmbeddingServiceError(RuntimeError):
    """The service failed or answered with something unusable."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


def validate_endpoint(base_url: str, *, verify_tls: bool = True) -> str:
    """Return ``base_url`` if it is https, loopback, or a declared tunnel host; else raise."""
    parts = urlsplit(base_url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise EndpointConfigurationError(
            "embedding endpoint must be an http(s) URL with a host, such as "
            f"http://127.0.0.1:8000/v1 (got scheme {parts.scheme or 'none'!r})"
        )
    if parts.username or parts.password:
        raise EndpointConfigurationError(
            f"embedding endpoint must not embed credentials; set {API_KEY_ENV} instead"
        )
    if parts.scheme == "https":
        if not verify_tls:
            raise EndpointConfigurationError("certificate verification must stay enabled")
        return base_url
    host = parts.hostname
    if not _is_loopback(host) and host.lower() not in _declared_tunnel_hosts():
        raise EndpointConfigurationError(
            f"plaintext http to {host} would send text unencrypted; use https, a loopback "
            "tunnel (ssh -L 8000:localhost:8000 host, then http://127.0.0.1:8000/v1), "
            f"or declare {host} in {TUNNEL_HOSTS_ENV}"
        )
    return base_url


def qwen_truncator(
    max_length: int, model_name: str = MODEL_NAME, tokenizer: Any | None = None
) -> Truncate:
    """Cut texts to ``max_length`` Qwen tokens (one is left for the end-of-text token).

    ``tokenizer`` defaults to the model's Hugging Face tokenizer; tests pass a stub.
    """
    if max_length < 2:
        raise ValueError("max_length must leave room for at least one token.")
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
    qwen: Any = tokenizer
    limit = max_length - 1

    def truncate(texts: Sequence[str]) -> list[str]:
        encoded = qwen(list(texts), add_special_tokens=False)["input_ids"]
        return [
            text if len(ids) <= limit else str(qwen.decode(ids[:limit]))
            for text, ids in zip(texts, encoded, strict=True)
        ]

    return truncate


class VllmEncoder:
    """Frozen Qwen3-Embedding vectors from a vLLM server, as ``(N, 1024)`` float32 tensors."""

    def __init__(
        self,
        base_url: str,
        *,
        max_length: int = 512,
        batch_size: int = 64,
        timeout: float = 120.0,
        max_attempts: int = 4,
        truncate: Truncate | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if batch_size <= 0 or max_attempts <= 0:
            raise ValueError("batch_size and max_attempts must be positive.")
        self.base_url = validate_endpoint(base_url).rstrip("/")
        self.max_length, self.batch_size = max_length, batch_size
        self.timeout, self.max_attempts = timeout, max_attempts
        self._truncate = truncate if truncate is not None else qwen_truncator(max_length)
        self._sleep = sleep
        self._api_key = os.environ.get(API_KEY_ENV) or None
        self._ssl = ssl.create_default_context() if base_url.startswith("https") else None
        self.revision = self._served_model()

    def __repr__(self) -> str:
        return f"VllmEncoder({self.base_url!r})"

    @property
    def settings(self) -> dict[str, Any]:
        return {"backend": "vllm", "model": self.revision, "max_length": self.max_length}

    def encode_queries(self, texts: Sequence[str], instruction: str | None) -> Tensor:
        return self.encode([format_query(text, instruction) for text in texts])

    def encode(self, texts: Sequence[str], progress: str = "") -> Tensor:
        output = np.empty((len(texts), EMBEDDING_DIM), dtype=np.float32)
        for batch_index, start in enumerate(range(0, len(texts), self.batch_size)):
            batch = self._truncate(texts[start : start + self.batch_size])
            payload = self._request("/embeddings", {"model": MODEL_NAME, "input": batch})
            output[start : start + len(batch)] = _validated(payload, len(batch), self.base_url)
            if progress and batch_index % 50 == 0:
                print(f"{progress}: {start + len(batch)}/{len(texts)}", file=sys.stderr)
        return torch.from_numpy(output)

    def _served_model(self) -> str:
        """Fail before sending any text if the server does not serve Qwen3-Embedding-0.6B."""
        payload = self._request("/models", None)
        models = payload.get("data") if isinstance(payload, dict) else None
        for model in models if isinstance(models, list) else []:
            if isinstance(model, dict) and model.get("id") == MODEL_NAME:
                root = model.get("root")
                return f"{MODEL_NAME}@{root}" if root and root != MODEL_NAME else MODEL_NAME
        raise EmbeddingServiceError(
            f"{self.base_url}/models does not list {MODEL_NAME}", retryable=False
        )

    def _request(self, path: str, body: dict[str, Any] | None) -> Any:
        url = self.base_url + path
        headers = {"Accept": "application/json"}
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        failure: EmbeddingServiceError | None = None
        for attempt in range(self.max_attempts):
            if attempt:
                self._sleep(min(2.0**attempt, 30.0))
            request = urllib.request.Request(url, data=data, headers=headers)
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout, context=self._ssl
                ) as response:
                    raw = response.read()
            except urllib.error.HTTPError as error:
                # The body is dropped on purpose: servers echo the input.
                retryable = error.code in _RETRYABLE_STATUS
                failure = EmbeddingServiceError(
                    f"embedding service {url} answered HTTP {error.code}", retryable=retryable
                )
                if not retryable:
                    raise failure from None
                continue
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
                reason = getattr(error, "reason", error)
                failure = EmbeddingServiceError(
                    f"embedding service {url} unreachable: {type(reason).__name__}",
                    retryable=True,
                )
                continue
            try:
                return json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise EmbeddingServiceError(
                    f"embedding service {url} returned a body that is not JSON", retryable=False
                ) from None
        assert failure is not None
        raise EmbeddingServiceError(
            f"{failure} (after {self.max_attempts} attempts)", retryable=True
        )


def _validated(payload: Any, expected: int, url: str) -> np.ndarray:
    """Check one response against the contract and return unit vectors."""

    def reject(reason: str) -> EmbeddingServiceError:
        return EmbeddingServiceError(f"embedding service {url}: {reason}", retryable=False)

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise reject("response has no data list")
    if len(data) != expected:
        raise reject(f"{len(data)} results for {expected} inputs")
    by_index: dict[int, Any] = {}
    for item in data:
        index = item.get("index") if isinstance(item, dict) else None
        if isinstance(index, bool) or not isinstance(index, int) or index in by_index:
            raise reject(f"result index {index!r} is invalid or repeated")
        by_index[index] = item
    if sorted(by_index) != list(range(expected)):
        raise reject(f"result indices are not exactly 0..{expected - 1}")
    vectors = np.empty((expected, EMBEDDING_DIM), dtype=np.float32)
    for index in range(expected):
        embedding = by_index[index].get("embedding")
        if not isinstance(embedding, list) or len(embedding) != EMBEDDING_DIM:
            raise reject(f"result {index} is not a list of {EMBEDDING_DIM} values")
        if any(isinstance(v, bool) or not isinstance(v, int | float) for v in embedding):
            raise reject(f"result {index} holds a non-numeric value")
        values = np.asarray(embedding, dtype=np.float64)
        norm = float(np.linalg.norm(values))
        if not np.all(np.isfinite(values)) or not math.isfinite(norm) or norm == 0.0:
            raise reject(f"result {index} is not finite or is a zero vector")
        vectors[index] = values / norm
    return vectors


def _is_loopback(host: str) -> bool:
    if host.lower() in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _declared_tunnel_hosts() -> frozenset[str]:
    raw = os.environ.get(TUNNEL_HOSTS_ENV, "")
    return frozenset(host.strip().lower() for host in raw.split(",") if host.strip())
