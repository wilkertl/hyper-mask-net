"""Paired significance tests, bootstrap intervals, and Holm correction over per-query values."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def paired_randomization_test(
    a: Sequence[float], b: Sequence[float], n_permutations: int = 10_000, seed: int = 0
) -> float:
    """Two-sided sign-flip test of ``mean(a - b) = 0``; returns ``(hits + 1) / (n + 1)``."""
    diffs = _diffs(a, b)
    observed = abs(diffs.mean())
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(n_permutations, diffs.size))
    hits = int((np.abs((signs * diffs).mean(axis=1)) >= observed - 1e-12).sum())
    return (hits + 1) / (n_permutations + 1)


def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    n_resamples: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for ``mean(a - b)``, resampling queries."""
    diffs = _diffs(a, b)
    rows = np.random.default_rng(seed).integers(0, diffs.size, size=(n_resamples, diffs.size))
    means = diffs[rows].mean(axis=1)
    tail = (1 - level) / 2
    return float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))


def holm(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values, monotone and capped at 1."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p))
        adjusted[name] = running
    return adjusted


def _diffs(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    if len(a) != len(b) or not a:
        raise ValueError("a and b must be non-empty and paired.")
    return np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
