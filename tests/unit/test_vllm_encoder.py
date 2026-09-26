"""VllmEncoder against a fake OpenAI-compatible vLLM server on loopback HTTP."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
import torch

from hyperdime.embeddings.qwen import MODEL_NAME
from hyperdime.embeddings.remote import (
    API_KEY_ENV,
    TUNNEL_HOSTS_ENV,
    EmbeddingServiceError,
    EndpointConfigurationError,
    VllmEncoder,
    qwen_truncator,
    validate_endpoint,
)

Reply = Callable[[list[str]], tuple[int, Any]]


def vector(text: str) -> list[float]:
    """Deterministic 1024-d vector: coordinate ``len(text) % 1024`` gets weight 2, others 0."""
    values = [0.0] * 1024
    values[len(text) % 1024] = 2.0
    return values


def ok(inputs: list[str]) -> tuple[int, Any]:
    data = [
        {"object": "embedding", "index": i, "embedding": vector(t)} for i, t in enumerate(inputs)
    ]
    return 200, {"object": "list", "model": MODEL_NAME, "data": data}


@dataclass
class FakeVllm:
    url: str
    requests: list[dict[str, Any]] = field(default_factory=list)
    headers: list[dict[str, str]] = field(default_factory=list)
    replies: list[Reply] = field(default_factory=list)
    models: list[str] = field(default_factory=lambda: [MODEL_NAME])


@pytest.fixture
def fake() -> Iterator[FakeVllm]:
    state = FakeVllm(url="")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

        def _send(self, status: int, body: Any) -> None:
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            self._send(200, {"data": [{"id": m, "root": m} for m in state.models]})

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.requests.append(body)
            state.headers.append(dict(self.headers))
            reply = state.replies.pop(0) if state.replies else ok
            self._send(*reply(body["input"]))

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    yield state
    server.shutdown()


def encoder(fake: FakeVllm, **kwargs: Any) -> VllmEncoder:
    def truncate(texts: Sequence[str]) -> list[str]:
        return [t[:12] for t in texts]

    return VllmEncoder(fake.url, truncate=truncate, sleep=lambda _: None, **kwargs)


def test_validate_endpoint_allows_only_protected_channels(monkeypatch: pytest.MonkeyPatch) -> None:
    assert validate_endpoint("https://embed.example/v1")
    assert validate_endpoint("http://127.0.0.1:8000/v1")
    assert validate_endpoint("http://localhost:8000/v1")
    for bad in ("http://10.0.0.5:8000/v1", "ftp://127.0.0.1/v1", "https://u:p@embed.example/v1"):
        with pytest.raises(EndpointConfigurationError):
            validate_endpoint(bad)
    monkeypatch.setenv(TUNNEL_HOSTS_ENV, "10.0.0.5")
    assert validate_endpoint("http://10.0.0.5:8000/v1")


def test_encode_batches_truncates_and_normalizes_in_input_order(fake: FakeVllm) -> None:
    texts = ["a", "bb", "ccc", "a very long document text", "e"]
    vectors = encoder(fake, batch_size=2).encode(texts)
    assert [len(r["input"]) for r in fake.requests] == [2, 2, 1]
    assert fake.requests[1]["input"][1] == "a very long "
    assert all(r["model"] == MODEL_NAME for r in fake.requests)
    assert vectors.dtype == torch.float32 and vectors.shape == (5, 1024)
    assert torch.allclose(vectors.norm(dim=-1), torch.ones(5))
    assert vectors[2, 3] == 1.0 and vectors[3, 12] == 1.0


def test_queries_carry_the_qwen_instruction(fake: FakeVllm) -> None:
    encoder(fake).encode_queries(["q"], "Find it")
    sent = fake.requests[0]["input"][0]
    assert sent == "Instruct: Fi"  # the fake truncator keeps 12 characters of the formatted query


def test_results_are_reordered_by_index(fake: FakeVllm) -> None:
    def shuffled(inputs: list[str]) -> tuple[int, Any]:
        status, body = ok(inputs)
        body["data"].reverse()
        return status, body

    fake.replies.append(shuffled)
    vectors = encoder(fake).encode(["x", "yy"])
    assert vectors[0, 1] == 1.0 and vectors[1, 2] == 1.0


def test_transient_failures_are_retried_and_permanent_ones_are_not(fake: FakeVllm) -> None:
    fake.replies += [lambda _: (503, {}), lambda _: (429, {})]
    assert encoder(fake).encode(["x"]).shape == (1, 1024)
    assert len(fake.requests) == 3
    fake.replies.append(lambda _: (400, {"error": "bad"}))
    with pytest.raises(EmbeddingServiceError) as error:
        encoder(fake).encode(["x"])
    assert not error.value.retryable and "x" not in str(error.value)


@pytest.mark.parametrize(
    "reply",
    [
        lambda inputs: (200, {"data": []}),
        lambda inputs: (200, {"data": [{"index": 0, "embedding": [1.0] * 10}]}),
        lambda inputs: (200, {"data": [{"index": 0, "embedding": [0.0] * 1024}]}),
        lambda inputs: (200, {"data": [{"index": 1, "embedding": [1.0] * 1024}]}),
    ],
)
def test_malformed_responses_are_rejected(fake: FakeVllm, reply: Reply) -> None:
    fake.replies.append(reply)
    with pytest.raises(EmbeddingServiceError):
        encoder(fake).encode(["x"])


def test_server_must_serve_the_qwen_model(fake: FakeVllm) -> None:
    fake.models = ["some/other-model"]
    with pytest.raises(EmbeddingServiceError, match="does not list"):
        encoder(fake)


def test_api_key_is_sent_only_from_the_environment(
    fake: FakeVllm, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoder(fake).encode(["x"])
    assert "Authorization" not in fake.headers[-1]
    monkeypatch.setenv(API_KEY_ENV, "secret-token")
    enc = encoder(fake)
    enc.encode(["x"])
    assert fake.headers[-1]["Authorization"] == "Bearer secret-token"
    assert "secret-token" not in repr(enc)
    assert enc.settings == {"backend": "vllm", "model": MODEL_NAME, "max_length": 512}


class WordTokenizer:
    """Stub Qwen tokenizer: one token per whitespace word."""

    def __init__(self) -> None:
        self.vocabulary: list[str] = []

    def __call__(self, texts: list[str], add_special_tokens: bool) -> dict[str, list[list[int]]]:
        ids = []
        for text in texts:
            row = []
            for word in text.split():
                if word not in self.vocabulary:
                    self.vocabulary.append(word)
                row.append(self.vocabulary.index(word))
            ids.append(row)
        return {"input_ids": ids}

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.vocabulary[i] for i in ids)


def test_qwen_truncator_keeps_max_length_minus_one_tokens() -> None:
    truncate = qwen_truncator(4, tokenizer=WordTokenizer())
    assert truncate(["a b c d e", "x y", "p q r"]) == ["a b c", "x y", "p q r"]
    with pytest.raises(ValueError):
        qwen_truncator(1, tokenizer=WordTokenizer())
