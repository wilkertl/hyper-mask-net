"""Selector recipes, grids, and validation-only selection (F3 research D4, ADR-0002, ADR-0003).

Every cell of a family is trained, all cells are scored together on validation queries
(nDCG@10 at k = 307, training labels, exact search over the full corpus), and the best cell is
kept. Test and evaluation-set judgments are never involved.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from hyperdime.baselines.learning_to_select import GeometricSelector, LinearSelector
from hyperdime.contracts.hashing import file_sha256
from hyperdime.contracts.manifests import ArtifactWriter
from hyperdime.contracts.schemas import PayloadRef, SelectorCheckpoint, load_record
from hyperdime.evaluation.runs import QuerySet, evaluate_methods
from hyperdime.oracle.targets import targets_at, uniform_kl
from hyperdime.selection.topk import resolve_k
from hyperdime.training.selector_trainer import TrainConfig, train_selector

HEADLINE_K = resolve_k(0.3, 1024)
OFFICIAL_TEMPERATURES = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
OFFICIAL_EPOCHS = (10, 50, 200)
WIDE_TEMPERATURES = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
WIDE_EPOCHS = (3, 10, 50, 200, 500)
STOP_EPOCHS = 500
STOP_PATIENCE = 5
ANCHOR_WEIGHTS = (1e-4, 1e-3, 1e-2, 1e-1)


@dataclass(frozen=True)
class Cell:
    """One trained configuration of a family."""

    family: str
    temperature: float
    epochs: int
    l2_to_anchor: float = 0.0
    patience: int | None = None
    schedule: str = "cosine"

    @property
    def key(self) -> str:
        parts = [self.family, f"t{self.temperature:g}", f"e{self.epochs}"]
        if self.l2_to_anchor:
            parts.append(f"l2{self.l2_to_anchor:g}")
        if self.patience is not None:
            parts.append(f"p{self.patience}")
        return "-".join(parts)

    def config(self, seed: int, device: str) -> TrainConfig:
        return TrainConfig(
            epochs=self.epochs,
            seed=seed,
            device=device,
            schedule=self.schedule,  # type: ignore[arg-type]
            patience=self.patience,
            l2_to_anchor=self.l2_to_anchor,
        )


def official_cells() -> list[Cell]:
    return [
        Cell("official", t, e) for t, e in itertools.product(OFFICIAL_TEMPERATURES, OFFICIAL_EPOCHS)
    ]


def candidate_cells(official: Cell) -> list[Cell]:
    """ADR-0003 families; validation-stopped and anchored cells reuse the official choices."""
    wide = [Cell("wide-grid", t, e) for t, e in itertools.product(WIDE_TEMPERATURES, WIDE_EPOCHS)]
    stopped = [
        Cell("val-stopped", t, STOP_EPOCHS, patience=STOP_PATIENCE, schedule="constant")
        for t in OFFICIAL_TEMPERATURES
    ]
    anchored = [
        Cell("global-regularized", official.temperature, official.epochs, l2_to_anchor=w)
        for w in ANCHOR_WEIGHTS
    ]
    return wide + stopped + anchored


@dataclass(frozen=True)
class Supervision:
    """Training and validation queries with precomputed oracle pairs and validation judgments."""

    train: tuple[Tensor, Tensor, Tensor]
    validation: tuple[Tensor, Tensor, Tensor]
    validation_ids: list[str]
    validation_judgments: Mapping[str, Mapping[str, int]]


@dataclass
class Trained:
    cell: Cell
    model: LinearSelector
    val_kl: float
    uniform_kl: float
    score: float = float("nan")


def train_cells(
    cells: Sequence[Cell],
    data: Supervision,
    *,
    seed: int = 0,
    device: str = "cpu",
    anchor: Mapping[str, Tensor] | None = None,
) -> list[Trained]:
    """Train every cell; targets are computed once per temperature."""
    by_temperature: dict[float, tuple[Tensor, Tensor]] = {}
    trained = []
    for cell in cells:
        if cell.temperature not in by_temperature:
            by_temperature[cell.temperature] = (
                targets_at(*data.train, cell.temperature),
                targets_at(*data.validation, cell.temperature),
            )
        train_targets, val_targets = by_temperature[cell.temperature]
        model, history = train_selector(
            data.train[0],
            train_targets,
            data.validation[0],
            val_targets,
            cell.config(seed, device),
            anchor=anchor if cell.l2_to_anchor else None,
        )
        trained.append(
            Trained(cell, model, min(h["val_kl"] for h in history), uniform_kl(val_targets))
        )
    return trained


def score_on_validation(
    trained: Sequence[Trained],
    data: Supervision,
    documents: np.ndarray,
    doc_ids: Sequence[str],
    *,
    environment: str,
    device: str = "cpu",
    chunk_size: int = 1_000_000,
) -> None:
    """Fill each cell's validation nDCG@10 at k = 307, all cells in one corpus pass."""
    queries = data.validation[0]
    methods: dict[str, Tensor | None] = {t.cell.key: importance(t.model, queries) for t in trained}
    query_set = QuerySet("validation", data.validation_ids, queries, data.validation_judgments)
    evaluation = evaluate_methods(
        query_set, methods, [HEADLINE_K], documents, doc_ids,
        environment=environment, device=device, chunk_size=chunk_size,
    )  # fmt: skip
    scores = {r.selector: r.metrics["ndcg@10"] for r in evaluation.results}
    for t in trained:
        t.score = scores[t.cell.key]


def best(trained: Sequence[Trained]) -> Trained:
    """Highest validation score; ties go to the earlier cell (the official grid comes first)."""
    return max(trained, key=lambda t: (t.score, -trained.index(t)))


def importance(model: LinearSelector, queries: Tensor, batch: int = 8192) -> Tensor:
    with torch.no_grad():
        return torch.cat([model(queries[i : i + batch]) for i in range(0, len(queries), batch)])


def summary(trained: Sequence[Trained]) -> dict[str, Any]:
    return {
        t.cell.key: {
            "cell": asdict(t.cell),
            "validation_ndcg@10": t.score,
            "val_kl": t.val_kl,
            "uniform_kl": t.uniform_kl,
        }
        for t in trained
    }


def save_selector(
    writer: ArtifactWriter,
    name: str,
    model: LinearSelector,
    *,
    kind: str,
    environment: str | None,
    training_manifest_sha256: str,
) -> SelectorCheckpoint:
    """Save a selector as safetensors with its ``SelectorCheckpoint``; the training manifest is
    fingerprinted at the start of the run (``config_hash`` of the starting ``RunManifest``),
    since the published manifest is written last."""
    from safetensors.torch import save_file

    path = writer.path(f"{name}.safetensors")
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(path))
    record = SelectorCheckpoint(
        kind=kind,
        environment=environment,
        payload=PayloadRef(path=path.name, sha256=file_sha256(path), format="safetensors"),
        training_manifest_sha256=training_manifest_sha256,
    )
    writer.write_record(f"{name}.json", record)
    return record


def load_selector(artifact_dir: Path, name: str) -> tuple[SelectorCheckpoint, LinearSelector]:
    from safetensors.torch import load_file

    record = load_record(SelectorCheckpoint, artifact_dir / f"{name}.json")
    state = load_file(str(artifact_dir / record.payload.path))
    model = (
        GeometricSelector(state["variance"], learn_weight="prior_weight" in state)
        if "variance" in state
        else LinearSelector()
    )
    model.load_state_dict(state)
    return record, model.eval()
