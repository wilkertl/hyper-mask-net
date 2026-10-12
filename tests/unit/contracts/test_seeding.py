"""One seed manager for every randomness source (research R7)."""

import random
from collections.abc import Iterator

import numpy as np
import pytest
import torch

from hyperdime.contracts.seeding import seed_everything


@pytest.fixture(autouse=True)
def _restore_torch_determinism() -> Iterator[None]:
    enabled = torch.are_deterministic_algorithms_enabled()
    benchmark = torch.backends.cudnn.benchmark
    yield
    torch.use_deterministic_algorithms(enabled)
    torch.backends.cudnn.benchmark = benchmark


def _draws() -> tuple[float, list[float], list[float]]:
    return random.random(), np.random.rand(3).tolist(), torch.rand(3).tolist()


def test_same_seed_gives_identical_draws_from_every_source() -> None:
    seed_everything(7)
    first = _draws()
    seed_everything(7)
    assert _draws() == first
    seed_everything(8)
    assert _draws() != first


def test_seed_manager_records_what_it_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    record = seed_everything(7)
    assert record == {
        "seed": "7",
        "deterministic": True,
        "cudnn_benchmark": False,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "0",
    }
    assert torch.are_deterministic_algorithms_enabled()


def test_non_deterministic_mode_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    record = seed_everything(3, deterministic=False)
    assert record["deterministic"] is False
    assert record["PYTHONHASHSEED"] is None
    assert not torch.are_deterministic_algorithms_enabled()
