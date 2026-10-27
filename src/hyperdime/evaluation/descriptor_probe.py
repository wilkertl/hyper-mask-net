"""Experiment E2: do per-dimension space descriptors predict how a selector shifts (F10)?

The unit is the dimension: each environment gives 1,024 rows (descriptor row, selector target),
and every test holds one environment out. The shift check (research D0) runs first, from targets
alone: if the profile the other environments share already predicts the held-out one up to seed
noise, there is no environment-specific shift for a descriptor to explain.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

# The shift check's threshold, fixed before any target was computed (research D0).
NO_SHIFT_GAP = 0.05
NO_SHIFT_MIN_FOLDS = 2

_A = ("doc_mean", "doc_std", "doc_mean_abs", "doc_rms",
      "doc_q05", "doc_q25", "doc_q50", "doc_q75", "doc_q95")  # fmt: skip
_TAILS = ("doc_skew", "doc_kurt", "doc_tail3")
_E = ("red_corr_mean_abs", "red_corr_max_abs", "red_corr_top8",
      "red_pc_share_8", "red_pc_share_32", "red_pc_share_128")  # fmt: skip
_B = tuple("qry_" + name.removeprefix("doc_") for name in _A + _TAILS)
_STATS = ("mean", "std", "q05", "q50", "q95")
_C = tuple(f"pair_{kind}_{stat}" for kind in ("rand", "top") for stat in _STATS)
_D = ("contrast_k5", "contrast_k10", "contrast_k20")


@dataclass(frozen=True)
class DescriptorSet:
    """One rung of the ladder: the descriptor columns a probe may use, and their protocol."""

    name: str
    columns: tuple[str, ...]
    protocol: str


# Set 1 uses the standard deviation: a one-feature linear probe ranks dimensions as that
# feature does, and std ranks them as the variance (research D7).
LADDER = (
    DescriptorSet("variance", ("doc_std",), "P0"),
    DescriptorSet("A", _A, "P0"),
    DescriptorSet("A+E", _A + _E, "P0"),
    DescriptorSet("A+E+tails", _A + _E + _TAILS, "P0"),
    DescriptorSet("+B+C", _A + _E + _TAILS + _B + _C, "P1"),
    DescriptorSet("+D", _A + _E + _TAILS + _B + _C + _D, "P1"),
)


def average_ranks(values: np.ndarray) -> np.ndarray:
    """0-based ranks, ties sharing their average, divided by ``n - 1`` (0 for one value)."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=ranks)
    averaged = (sums / counts)[inverse]
    return averaged / max(len(values) - 1, 1)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation; tied values share their average rank."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be vectors of equal length.")
    return float(np.corrcoef(average_ranks(a), average_ranks(b))[0, 1])


def spearman_brown(rho: float, replicates: int) -> float:
    """The reliability of an average of ``replicates`` profiles from that of one, ``ρ``."""
    return replicates * rho / (1 + (replicates - 1) * rho)


def shift_check(profiles: Mapping[str, Mapping[int, np.ndarray]]) -> dict[str, Any]:
    """Per held-out environment, B1 (the mean of the other environments' seed-averaged profiles)
    against the noise ceiling (the mean pairwise seed Spearman stepped up to the seed average).

    ``no_shift`` holds when the ceiling exceeds B1 by less than ``NO_SHIFT_GAP`` in at least
    ``NO_SHIFT_MIN_FOLDS`` environments (research D0).
    """
    if len(profiles) < 3:
        raise ValueError("the shift check needs at least three environments")
    if any(len(seeds) < 2 for seeds in profiles.values()):
        raise ValueError("the noise ceiling needs at least two seeds per environment")
    means = {env: seed_mean(seeds) for env, seeds in profiles.items()}
    folds = {}
    for env, seeds in profiles.items():
        shared = np.mean([means[other] for other in profiles if other != env], axis=0)
        pairs = itertools.combinations(sorted(seeds), 2)
        rho = float(np.mean([spearman(seeds[i], seeds[j]) for i, j in pairs]))
        ceiling = spearman_brown(rho, len(seeds))
        b1 = spearman(shared, means[env])
        folds[env] = {"b1": b1, "seed_spearman": rho, "ceiling": ceiling, "gap": ceiling - b1}
    narrow = sum(fold["gap"] < NO_SHIFT_GAP for fold in folds.values())
    return {
        "folds": folds,
        "no_shift": narrow >= NO_SHIFT_MIN_FOLDS,
        "threshold": {"gap": NO_SHIFT_GAP, "min_folds": NO_SHIFT_MIN_FOLDS},
    }


def seed_mean(seeds: Mapping[int, np.ndarray] | Sequence[np.ndarray]) -> np.ndarray:
    values = seeds.values() if isinstance(seeds, Mapping) else seeds
    return np.mean([np.asarray(v, dtype=np.float64) for v in values], axis=0)
