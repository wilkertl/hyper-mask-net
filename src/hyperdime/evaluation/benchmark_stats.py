"""F12 statistics (ADR-0018, "Endpoint and statistics").

One-sided paired sign-flip test (H1: mean difference > 0) with ``p = (1 + #{T* >= T}) / (B + 1)``,
or exact over all ``2^n`` sign vectors when ``2^n <= B``. Paired percentile bootstrap over queries
for a pointwise two-sided 95% interval and a one-sided Bonferroni lower bound at ``alpha / m``.
Random streams are PCG64 children of ``SeedSequence(20261112)`` keyed by (procedure, corpus, arm)
codes, drawn in batches of at most 10^7 elements.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping

import numpy as np

ROOT_SEED = 20261112
DRAWS = 100_000
MAX_ELEMENTS = 10_000_000
PROCEDURES = {"bootstrap": 1, "signflip": 2}
CORPORA = {"scifact": 1, "nfcorpus": 2, "msmarco": 3}
ARMS = {
    "positive": 1,
    "contrastive": 2,
    "rank-guided": 3,
    "s1-scratch": 4,
    "s1-official-init": 5,
    "global-mask": 6,
    "variance": 7,
    "full": 8,
    "mrl-prefix": 9,  # added after the SciFact and NFCorpus locks; existing keys are unchanged
}


def stream(procedure: str, corpus: str, arm: str) -> np.random.Generator:
    key = (PROCEDURES[procedure], CORPORA[corpus], ARMS[arm])
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(ROOT_SEED, spawn_key=key)))


def _batch(n: int, draws: int) -> int:
    return max(1, min(draws, MAX_ELEMENTS // max(n, 1)))


def sign_flip_p(diffs: np.ndarray, rng: np.random.Generator, draws: int = DRAWS) -> float:
    """One-sided p-value for ``mean(diffs) > 0``; ties with the observed value count as >=."""
    n = len(diffs)
    observed = diffs.mean()
    if 2**n <= draws:
        signs = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
        return float(((signs @ diffs) / n >= observed).sum() / len(signs))
    count, done = 0, 0
    while done < draws:
        size = min(_batch(n, draws), draws - done)
        signs = rng.choice((-1.0, 1.0), size=(size, n))
        count += int(((signs @ diffs) / n >= observed).sum())
        done += size
    return (1 + count) / (draws + 1)


def bootstrap_means(diffs: np.ndarray, rng: np.random.Generator, draws: int = DRAWS) -> np.ndarray:
    n = len(diffs)
    means = np.empty(draws)
    done = 0
    while done < draws:
        size = min(_batch(n, draws), draws - done)
        index = rng.integers(0, n, size=(size, n))
        means[done : done + size] = diffs[index].mean(1)
        done += size
    return means


def intervals(means: np.ndarray, family_size: int, alpha: float = 0.05) -> Mapping[str, float]:
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "pointwise_low": float(low),
        "pointwise_high": float(high),
        "bonferroni_lower": float(np.quantile(means, alpha / family_size)),
    }


def holm(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values (monotone step-down)."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[name] = running
    return adjusted


def claim_level(
    difference: float, holm_p: float | None, bonferroni_lower: float, delta: float = 0.005
) -> str:
    if holm_p is None:
        return "pending (locked family incomplete)"
    if not (difference > 0 and holm_p < 0.05):
        return "not superior"
    if bonferroni_lower >= delta:
        return "established minimum gain"
    return "practical candidate" if difference >= delta else "superior"
