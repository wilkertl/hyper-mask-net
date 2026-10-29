"""The retrieval-objective grid and its validation-only selection (F5 research D6).

Every cell is trained with seed 0 and scored on validation queries by exact full-corpus nDCG@10
with the hard top-k mask at the nine 10%-90% budgets, all cells in one corpus pass; the selected
cell has the highest mean over budgets among cells whose soft mask ends close to the hard one
(research D3). Test and evaluation-set judgments are never involved.

The S4 tracks (``budget`` and its control ``weighted-aligned``, F9 research D3–D6) use the aligned
checkpoint rule: every evaluated state of a cell is a candidate, scored by exact full-corpus
validation nDCG@10 on a fixed subset of at most 1,000 validation queries; the best is kept, and the
guard reads the kept state's soft mass at the final mask temperature.
"""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from torch import Tensor

from hyperdime.baselines.learning_to_select import BudgetGeometricSelector, LinearSelector
from hyperdime.baselines.methods import budget_importance
from hyperdime.evaluation.runs import Importance, QuerySet, evaluate_methods
from hyperdime.training.examples import RetrievalExamples
from hyperdime.training.retrieval_trainer import (
    BUDGETS,
    HEADLINE_K,
    Candidates,
    RetrievalConfig,
    train_retrieval_candidates,
)
from hyperdime.training.retrieval_trainer import soft_mass as soft_mass_at
from hyperdime.training.selection import Supervision, importance

VARIANTS = ("fixed-307", "multi-budget")
INITS = ("scratch", "l2s")
LEARNING_RATES = (1e-4, 1e-3)
SCORE_TEMPERATURES = (0.02, 0.05)
BETA_ENDS = (0.1, 0.02)
STEPS = 20_000
SOFT_MASS_GUARD = 0.9
# S4's tracks: the budget selector and its control, S3's selector under the same rules.
ALIGNED_INITS = ("budget", "weighted-aligned")
PRIOR_INITS = ("prior", "weighted", *ALIGNED_INITS)
WEIGHTED_INITS = ("weighted", "weighted-aligned")
CANDIDATE_QUERIES = 1_000
CANDIDATE_SEED = 0


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

    @property
    def aligned(self) -> bool:
        """S4's tracks: the aligned checkpoint rule and nonnegative prior weights (F9)."""
        return self.init in ALIGNED_INITS

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
            checkpoint_rule="validation" if self.aligned else "rerank",
            nonnegative_prior=self.aligned,
        )


def retrieval_cells(steps: int = STEPS, inits: Sequence[str] = INITS) -> list[RetrievalCell]:
    """The 32 cells of research D6 in the order of the product; S2 passes ``inits=("prior",)``
    for its 16 cells that start from the geometric prior. S4's inits train ``multi-budget`` only
    (F9 research D6), 8 cells each."""
    grid = itertools.product(VARIANTS, inits, LEARNING_RATES, SCORE_TEMPERATURES, BETA_ENDS)
    return [
        RetrievalCell(*values, steps=steps)
        for values in grid
        if values[0] == "multi-budget" or values[1] not in ALIGNED_INITS
    ]


def candidate_queries(
    query_ids: Sequence[str], size: int = CANDIDATE_QUERIES, seed: int = CANDIDATE_SEED
) -> list[int]:
    """Rows of the at most ``size`` validation queries smallest under ``sha256(f"{seed}:{qid}")``,
    in that order: the same subset whatever the row order (F9 research D4)."""
    digest = {qid: hashlib.sha256(f"{seed}:{qid}".encode()).hexdigest() for qid in query_ids}
    rows = sorted(range(len(query_ids)), key=lambda i: digest[query_ids[i]])
    return rows[:size]


@dataclass
class TrainedRetrieval:
    """A trained cell. Under the aligned rule ``candidates`` holds its evaluated states until
    ``choose_candidates`` keeps one (``kept_step``) and records its soft mass at ``beta_end``."""

    cell: RetrievalCell
    model: LinearSelector
    history: list[dict[str, float]]
    scores: dict[int, float] = field(default_factory=dict)
    candidates: Candidates = field(default_factory=list)
    candidate_scores: dict[int, float] = field(default_factory=dict)
    kept_step: int | None = None
    kept_soft_mass: float | None = None

    @property
    def mean(self) -> float:
        return float(np.mean(list(self.scores.values())))

    @property
    def soft_mass(self) -> float:
        if not self.cell.aligned:
            return self.history[-1]["soft_mass"]
        if self.kept_soft_mass is None:
            raise ValueError(f"{self.cell.key}: no state is kept yet; run choose_candidates")
        return self.kept_soft_mass

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
    """Train every cell; ``l2s`` cells start from ``start``, and ``prior`` (S2), ``weighted`` (S3,
    with a learned weight on the prior), ``weighted-aligned`` and ``budget`` (S4, one weight per
    budget) cells from the corpus variance ``prior``."""
    trained = []
    for cell in cells:
        uses_prior = cell.init in PRIOR_INITS
        if cell.init == "l2s" and start is None:
            raise ValueError(f"{cell.key} starts from an L2S selector; none was given")
        if uses_prior and prior is None:
            raise ValueError(f"{cell.key} starts from the geometric prior; none was given")
        model, history, candidates = train_retrieval_candidates(
            train, validation, cell.config(seed, device),
            start if cell.init == "l2s" else None, prior if uses_prior else None,
            learn_prior_weight=cell.init in WEIGHTED_INITS, per_budget_prior=cell.init == "budget",
        )  # fmt: skip
        trained.append(TrainedRetrieval(cell, model, history, candidates=candidates))
    return trained


def selector_scores(model: LinearSelector, queries: Tensor, budgets: Sequence[int]) -> Importance:
    """Importance as ``evaluate_methods`` takes it: per budget for a budget selector."""
    if isinstance(model, BudgetGeometricSelector):
        return {k: v for k, v in budget_importance(queries, model).items() if k in budgets}
    return importance(model, queries)


def choose_candidates(
    trained: Sequence[TrainedRetrieval],
    data: Supervision,
    documents: np.ndarray,
    doc_ids: Sequence[str],
    *,
    environment: str,
    device: str = "cpu",
    chunk_size: int = 1_000_000,
    budgets: Sequence[int] = BUDGETS,
) -> list[str]:
    """Keep each aligned cell's best candidate (F9 research D3–D5).

    Candidates are scored on the ``candidate_queries`` subset by exact full-corpus nDCG@10 with
    the hard top-k mask (linear gains), mean over ``budgets``, one corpus pass per cell to bound
    memory; ties go to the earlier step. The kept state replaces the cell's model, and its soft
    mass at ``beta_end`` on at most 1,024 validation queries becomes the guard. Returns the
    subset's query ids."""
    rows = candidate_queries(data.validation_ids)
    queries = data.validation[0][rows]
    ids = [data.validation_ids[i] for i in rows]
    query_set = QuerySet("validation-candidates", ids, queries, data.validation_judgments)
    for t in trained:
        if not t.cell.aligned:
            continue
        if not t.candidates:
            raise ValueError(f"{t.cell.key} has no candidates to choose from")
        methods: dict[str, Importance] = {}
        for step, state in t.candidates:
            t.model.load_state_dict(state)
            methods[str(step)] = selector_scores(t.model.eval(), queries, budgets)
        evaluation = evaluate_methods(
            query_set, methods, list(budgets), documents, doc_ids,
            environment=environment, device=device, chunk_size=chunk_size,
        )  # fmt: skip
        by_step: dict[int, list[float]] = {}
        for r in evaluation.results:
            by_step.setdefault(int(r.selector), []).append(r.metrics["ndcg@10"])
        t.candidate_scores = {step: float(np.mean(v)) for step, v in sorted(by_step.items())}
        t.kept_step = max(t.candidate_scores, key=lambda step: (t.candidate_scores[step], -step))
        t.model.load_state_dict(dict(t.candidates)[t.kept_step])
        t.model.eval()
        t.kept_soft_mass = soft_mass_at(t.model, data.validation[0], budgets, t.cell.beta_end)
        t.candidates = []
    return ids


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
    methods: dict[str, Importance] = {
        t.cell.key: selector_scores(t.model, queries, budgets) for t in trained
    }
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
            **(_aligned_summary(t) if t.cell.aligned else {}),
        }
        for t in trained
    }


def _aligned_summary(t: TrainedRetrieval) -> dict[str, Any]:
    """The S4 additions to a cell's ``selection.json`` entry (F9 data model)."""
    rerank = {int(h["step"]): h["val_rerank_ndcg10"] for h in t.history}
    weights = t.model.state_dict().get("prior_weights", t.model.state_dict().get("prior_weight"))
    return {
        "checkpoint_rule": "validation",
        "candidate_scores": {
            str(step): {"ndcg@10": score, "val_rerank_ndcg10": rerank[step]}
            for step, score in t.candidate_scores.items()
        },
        "kept_step": t.kept_step,
        "prior_weights": [] if weights is None else weights.reshape(-1).tolist(),
        "projected_steps": int(t.history[-1].get("projected_steps", 0)),
    }
