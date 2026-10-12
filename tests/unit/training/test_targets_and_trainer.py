"""Oracle targets from gathered rows, trainer extensions, and selector checkpoints (F3 US2)."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from hyperdime.contracts.manifests import ArtifactWriter, start_manifest
from hyperdime.oracle.targets import aggregate_targets, oracle_pairs, targets_at, uniform_kl
from hyperdime.training.selection import load_selector, save_selector
from hyperdime.training.selector_trainer import TrainConfig, train_selector

GEN = torch.Generator().manual_seed(0)
DOCS = F.normalize(torch.randn(20, 1024, generator=GEN), dim=-1).numpy().astype(np.float16)
QUERIES = F.normalize(torch.randn(3, 1024, generator=GEN), dim=-1).numpy()
DOC_ROWS = {f"d{i}": i for i in range(20)}
POSITIVES = {"a": {"d1": 2, "d2": 1, "d3": 0}, "b": {"d4": 1}, "c": {"d5": 3}}
POOLS = {"a": ["d7", "d7", "d8"], "b": ["d9"], "c": ["d10", "d11"]}


def test_targets_from_gathered_rows_match_the_direct_computation() -> None:
    queries, p, n = oracle_pairs(
        ["a", "b", "c"], QUERIES, {"a": 0, "b": 1, "c": 2}, DOCS, DOC_ROWS, POSITIVES, POOLS
    )
    targets = targets_at(queries, p, n, 0.05)
    docs = torch.from_numpy(DOCS.astype(np.float32))
    direct = aggregate_targets(
        torch.from_numpy(QUERIES),
        docs,
        [{1: 2, 2: 1}, {4: 1}, {5: 3}],
        [[7, 7, 8], [9], [10, 11]],
        0.05,
    )
    assert torch.allclose(targets, direct, atol=1e-6)
    assert (
        torch.allclose(targets.sum(1), torch.ones(3), atol=1e-5) and torch.isfinite(targets).all()
    )


def test_uniform_kl_by_hand() -> None:
    assert uniform_kl(torch.full((2, 1024), 1 / 1024)) == pytest.approx(0.0, abs=1e-6)
    one_hot = torch.zeros(1, 1024)
    one_hot[0, 0] = 1.0
    assert uniform_kl(one_hot) == pytest.approx(np.log(1024), rel=1e-6)


def _data() -> tuple[torch.Tensor, ...]:
    q = F.normalize(torch.randn(50, 1024, generator=torch.Generator().manual_seed(1)), dim=-1)
    t = torch.softmax(torch.randn(50, 1024, generator=torch.Generator().manual_seed(2)) * 3, -1)
    return q[:40], t[:40], q[40:], t[40:]


def test_a_zero_anchor_weight_equals_no_anchor() -> None:
    plain, _ = train_selector(*_data(), TrainConfig(epochs=3))
    anchor = {k: torch.zeros_like(v) for k, v in plain.state_dict().items()}
    anchored, _ = train_selector(*_data(), TrainConfig(epochs=3, l2_to_anchor=0.0), anchor=anchor)
    assert all(torch.equal(plain.state_dict()[k], anchored.state_dict()[k]) for k in anchor)


def test_a_strong_anchor_pulls_toward_it() -> None:
    plain, _ = train_selector(*_data(), TrainConfig(epochs=3))
    anchor = {k: torch.zeros_like(v) for k, v in plain.state_dict().items()}
    anchored, _ = train_selector(*_data(), TrainConfig(epochs=3, l2_to_anchor=1.0), anchor=anchor)
    norm = lambda m: sum(float(v.norm()) for v in m.state_dict().values())  # noqa: E731
    assert norm(anchored) < norm(plain)


def test_patience_stops_training_early() -> None:
    _, history = train_selector(
        *_data(), TrainConfig(epochs=200, lr=1.0, schedule="constant", patience=2)
    )
    assert len(history) < 200


def test_a_saved_selector_reloads_identically(tmp_path: Path) -> None:
    model, _ = train_selector(*_data(), TrainConfig(epochs=2))
    with ArtifactWriter(
        tmp_path, "selectors/toy", start_manifest("train-selector", {}, seed=0)
    ) as w:
        save_selector(
            w,
            "official",
            model,
            kind="per_domain",
            environment="toy",
            training_manifest_sha256="ab" * 32,
        )
        directory = w.commit()
    record, loaded = load_selector(directory, "official")
    queries = _data()[0]
    assert record.kind == "per_domain" and torch.equal(model(queries), loaded(queries))
