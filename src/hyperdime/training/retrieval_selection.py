"""The retrieval-objective grid and its validation-only selection (F5 research D6).

Every cell is trained with seed 0 and scored on validation queries by exact full-corpus nDCG@10
with the hard top-k mask at the nine 10%-90% budgets, all cells in one corpus pass; the selected
cell has the highest mean over budgets among cells whose soft mask ends close to the hard one
(research D3). Test and evaluation-set judgments are never involved.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from torch import Tensor

from hyperdime.baselines.learning_to_select import LinearSelector
from hyperdime.evaluation.runs import QuerySet, evaluate_methods
from hyperdime.training.examples import RetrievalExamples
from hyperdime.training.retrieval_trainer import (
    BUDGETS,
    HEADLINE_K,
    RetrievalConfig,
    train_retrieval_selector,
)
from hyperdime.training.selection import Supervision, importance

VARIANTS = ("fixed-307", "multi-budget")
INITS = ("scratch", "l2s")
LEARNING_RATES = (1e-4, 1e-3)
SCORE_TEMPERATURES = (0.02, 0.05)
BETA_ENDS = (0.1, 0.02)
STEPS = 20_000
SOFT_MASS_GUARD = 0.9


@dataclass(frozen=True)
class RetrievalCell:
    variant: str
    init: str
    lr: float
    score_temperature: float
    beta_end: float
    steps: int = STEPS

    @property
    def key(self) -> str:
        return (
            f"{self.variant}-{self.init}-lr{self.lr:g}"
            f"-ts{self.score_temperature:g}-b{self.beta_end:g}"
        )

    @property
    def family(self) -> str:
        return f"{self.variant}-{self.init}"

    def config(self, seed: int, device: str) -> RetrievalConfig:
        budgets = (HEADLINE_K,) if self.variant == "fixed-307" else BUDGETS
        return RetrievalConfig(
            budgets=budgets,
            lr=self.lr,
            score_temperature=self.score_temperature,
            beta_end=self.beta_end,
            steps=self.steps,
            seed=seed,
            device=device,
        )


def retrieval_cells(steps: int = STEPS, inits: Sequence[str] = INITS) -> list[RetrievalCell]:
    """The 32 cells of research D6 in the order of the product; S2 passes ``inits=("prior",)``
    for its 16 cells that start from the geometric prior."""
    grid = itertools.product(VARIANTS, inits, LEARNING_RATES, SCORE_TEMPERATURES, BETA_ENDS)
    return [RetrievalCell(*values, steps=steps) for values in grid]


@dataclass
class TrainedRetrieval:
    cell: RetrievalCell
    model: LinearSelector
    history: list[dict[str, float]]
    scores: dict[int, float] = field(default_factory=dict)

    @property
    def mean(self) -> float:
        return float(np.mean(list(self.scores.values())))

    @property
    def soft_mass(self) -> float:
        return self.history[-1]["soft_mass"]

    @property
    def selectable(self) -> bool:
        return self.soft_mass >= SOFT_MASS_GUARD


def train_retrieval_cells(
    cells: Sequence[RetrievalCell],
    train: RetrievalExamples,
    validation: RetrievalExamples,
    *,
    seed: int = 0,
    device: str = "cpu",
    start: Mapping[str, Tensor] | None = None,
    prior: Tensor | None = None,
) -> list[TrainedRetrieval]:
    """Train every cell; ``l2s`` cells start from ``start`` and ``prior`` cells from the corpus
    variance ``prior``, which they need."""
    trained = []
    for cell in cells:
        if cell.init == "l2s" and start is None:
            raise ValueError(f"{cell.key} starts from an L2S selector; none was given")
        if cell.init == "prior" and prior is None:
            raise ValueError(f"{cell.key} starts from the geometric prior; none was given")
        model, history = train_retrieval_selector(
            train, validation, cell.config(seed, device),
            start if cell.init == "l2s" else None, prior if cell.init == "prior" else None,
        )  # fmt: skip
        trained.append(TrainedRetrieval(cell, model, history))
    return trained


def score_on_validation_budgets(
    trained: Sequence[TrainedRetrieval],
    data: Supervision,
    documents: np.ndarray,
    doc_ids: Sequence[str],
    *,
    environment: str,
    device: str = "cpu",
    chunk_size: int = 1_000_000,
    budgets: Sequence[int] = BUDGETS,
) -> None:
    """Fill each cell's validation nDCG@10 per budget, all cells in one corpus pass."""
    queries = data.validation[0]
    methods: dict[str, Tensor | None] = {t.cell.key: importance(t.model, queries) for t in trained}
    query_set = QuerySet("validation", data.validation_ids, queries, data.validation_judgments)
    evaluation = evaluate_methods(
        query_set, methods, list(budgets), documents, doc_ids,
        environment=environment, device=device, chunk_size=chunk_size,
    )  # fmt: skip
    for t in trained:
        t.scores = {
            r.k: r.metrics["ndcg@10"] for r in evaluation.results if r.selector == t.cell.key
        }


def best_retrieval(trained: Sequence[TrainedRetrieval]) -> TrainedRetrieval:
    """Highest mean over budgets among selectable cells; ties go to the earlier cell."""
    candidates = [t for t in trained if t.selectable]
    if not candidates:
        raise ValueError("no cell's soft mask ends close enough to the hard top-k (research D3)")
    return max(candidates, key=lambda t: (t.mean, -trained.index(t)))


def family_bests(trained: Sequence[TrainedRetrieval]) -> dict[str, TrainedRetrieval]:
    """The selectable cell with the highest mean in each (variant, init) family."""
    families: dict[str, list[TrainedRetrieval]] = {}
    for t in trained:
        if t.selectable:
            families.setdefault(t.cell.family, []).append(t)
    return {family: best_retrieval(members) for family, members in families.items()}


def retrieval_summary(trained: Sequence[TrainedRetrieval]) -> dict[str, Any]:
    return {
        t.cell.key: {
            "cell": asdict(t.cell),
            "validation_ndcg@10": {str(k): v for k, v in sorted(t.scores.items())},
            "validation_mean": t.mean,
            "best_step": max(t.history, key=lambda h: h["val_rerank_ndcg10"])["step"],
            "soft_mass": t.soft_mass,
            "selectable": t.selectable,
        }
        for t in trained
    }
