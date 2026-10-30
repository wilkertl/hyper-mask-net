"""Per-dimension selector targets for experiment E2 (F10 research D6).

A target profile says, for each of the 1,024 dimensions, how much an environment's selector uses
it: the share of validation queries keeping the dimension in their top k, divided by k so each
budget's profile sums to 1, averaged over the nine 10%-90% budgets. The primary target comes from
the S1 selector, which has no geometric prior, so it is not built from ``q² · Var`` itself.
Validation judgments are never read: only the validation queries' embeddings.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from hyperdime.artifacts import latest_artifact
from hyperdime.baselines.learning_to_select import (
    BUDGETS,
    BudgetGeometricSelector,
    GeometricSelector,
    LinearSelector,
)
from hyperdime.baselines.methods import selector_importance, variance_importance
from hyperdime.contracts.hashing import file_sha256
from hyperdime.contracts.schemas import SelectorCheckpoint, load_record
from hyperdime.contracts.validation import ContractError
from hyperdime.evaluation.descriptor_probe import seed_mean
from hyperdime.evaluation.selector_shift import selection_frequency
from hyperdime.training.retrieval_selection import candidate_queries
from hyperdime.training.selection import load_selector

VALIDATION_QUERIES = 10_000
QUERY_SEED = 0
BATCH = 8192


def validation_sample(query_ids: Sequence[str], size: int = VALIDATION_QUERIES) -> list[str]:
    """At most ``size`` validation query ids by a seeded hash, the same whatever their order."""
    return [query_ids[row] for row in candidate_queries(query_ids, size, QUERY_SEED)]


def frequency_profiles(
    scores: Callable[[Tensor], Tensor], queries: Tensor, budgets: Sequence[int]
) -> dict[int, np.ndarray]:
    """Per budget, ``selection_frequency(scores, k) / k`` (sums to 1); scores are computed in
    batches and frequencies accumulated, so large query sets fit in memory."""
    counts = {k: torch.zeros(queries.shape[1], dtype=torch.float64) for k in budgets}
    with torch.no_grad():
        for start in range(0, len(queries), BATCH):
            batch = scores(queries[start : start + BATCH])
            for k in budgets:
                counts[k] += selection_frequency(batch, k).double() * len(batch)
    return {k: (counts[k] / len(queries) / k).numpy() for k in budgets}


def target_profiles(
    s1: Mapping[int, LinearSelector],
    s4: Mapping[int, BudgetGeometricSelector],
    variance: Tensor,
    queries: Tensor,
    budgets: Sequence[int] = BUDGETS,
) -> dict[str, np.ndarray]:
    """Every target profile of one environment, keyed as in the data model."""
    found: dict[str, np.ndarray] = {}
    for seed, model in sorted(s1.items()):
        per_budget = frequency_profiles(model, queries, budgets)
        for k, profile in per_budget.items():
            found[f"s1/seed-{seed}/budget-{k}"] = profile
        found[f"s1/seed-{seed}/mean"] = np.mean(list(per_budget.values()), axis=0)
        importance = selector_importance(queries, model).double()
        found[f"s1/seed-{seed}/log_importance"] = importance.mean(dim=0).numpy()
        found[f"s1/seed-{seed}/bias"] = model.projection.bias.detach().double().numpy()
    for seed, model in sorted(s4.items()):
        # The correction ``W q + b`` alone; the prior's weight is per budget, so it is skipped.
        correction = frequency_profiles(_correction(model), queries, budgets)
        found[f"s4_correction/seed-{seed}/mean"] = np.mean(list(correction.values()), axis=0)
    for name in ("s1", "s4_correction"):
        seeds = seed_profiles(found, name)
        if seeds:
            found[f"{name}/mean"] = seed_mean([seeds[s] for s in sorted(seeds)])
    if s1:
        found["s1/bias"] = seed_mean([found[f"s1/seed-{s}/bias"] for s in sorted(s1)])
    prior = frequency_profiles(lambda q: variance_importance(q, variance), queries, budgets)
    found["variance/mean"] = np.mean(list(prior.values()), axis=0)
    return found


def _correction(model: BudgetGeometricSelector) -> Callable[[Tensor], Tensor]:
    def scores(queries: Tensor) -> Tensor:
        found: Tensor = model.projection(queries.float())
        return found

    return scores


def seed_profiles(profiles: Mapping[str, np.ndarray], target: str = "s1") -> dict[int, np.ndarray]:
    """``{seed: profile}`` of one target's per-seed means, for the shift check."""
    prefix = f"{target}/seed-"
    return {
        int(key.removeprefix(prefix).split("/")[0]): value
        for key, value in profiles.items()
        if key.startswith(prefix) and key.endswith("/mean")
    }


def seeded_selectors(
    root: Path, environment: str, track: str, seeds: Sequence[int]
) -> tuple[dict[int, LinearSelector], dict[str, Any]]:
    """The latest seeded selectors of ``track`` (``retrieval`` for S1, ``budget`` for S4), each
    checked to retrain its grid's selected cell and to match its payload hash."""
    grid_dir = latest_artifact(root, f"selectors/{track}-grid", environment)
    grid = json.loads((grid_dir / "selection.json").read_text(encoding="utf-8"))
    sources: dict[str, Any] = {"grid": grid_dir.name, "cell": grid["selected"], "seeds": {}}
    models: dict[int, LinearSelector] = {}
    for seed in seeds:
        directory = latest_artifact(root, f"selectors/{track}-seeded/{environment}", f"seed-{seed}")
        record = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
        if record["source"] != {"run": grid_dir.name, "cell": grid["selected"]}:
            raise ContractError(f"{directory} is not the latest grid's selected cell")
        checkpoint = load_record(SelectorCheckpoint, directory / "selector.json")
        if file_sha256(directory / checkpoint.payload.path) != checkpoint.payload.sha256:
            raise ContractError(f"{directory}: selector payload does not match its record")
        models[seed] = load_selector(directory, "selector")[1].eval()
        sources["seeds"][str(seed)] = {"run": directory.name, "sha256": checkpoint.payload.sha256}
    return models, sources


def corpus_variance(s4: Mapping[int, LinearSelector]) -> Tensor:
    """The corpus variance the S4 selectors were built with (document embeddings only); every
    seed must carry the same one."""
    found = [m.variance for m in s4.values() if isinstance(m, GeometricSelector)]
    if not found or any(not torch.equal(found[0], v) for v in found[1:]):
        raise ContractError("the S4 seeds do not share one corpus variance")
    return found[0]
