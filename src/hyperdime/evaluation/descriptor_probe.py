"""Experiment E2: do per-dimension space descriptors predict how a selector shifts (F10)?

The unit is the dimension: each environment gives 1,024 rows (descriptor row, selector target),
and every test holds one environment out. The shift check (research D0) runs first, from targets
alone: if the profile the other environments share already predicts the held-out one up to seed
noise, there is no environment-specific shift for a descriptor to explain.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from hyperdime.evaluation.statistics import holm

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


# The ridge penalties every probe chooses from (FR-004).
PENALTIES = (0.1, 1.0, 10.0, 100.0, 1000.0)
PERMUTATION_SEED = 0
_FLAT = 1e-12  # a pooled standard deviation at or below this drops the feature for the fold

Probe = Callable[[Mapping[str, np.ndarray], Mapping[str, np.ndarray], str], "ProbeFit"]


@dataclass(frozen=True)
class ProbeFit:
    """A probe fitted without the held-out environment, scored on it."""

    spearman: float
    penalty: float | None
    coefficients: dict[str, float]  # standardized, by feature name; dropped features absent
    prediction: np.ndarray


def ridge_fit(x: np.ndarray, y: np.ndarray, penalty: float) -> np.ndarray:
    """``argmin ‖y − xβ‖² + λ‖β‖²`` without intercept (the data are centred or standardized)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    return np.linalg.solve(x.T @ x + penalty * np.eye(x.shape[1]), x.T @ y)


def ridge_predict(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    found: np.ndarray = np.asarray(x, dtype=np.float64) @ beta
    return found


def gcv_penalty(x: np.ndarray, y: np.ndarray, penalties: Sequence[float] = PENALTIES) -> float:
    """The penalty minimizing generalized cross-validation ``n · RSS / (n − tr H)²``; the first
    of equal scores wins."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    projected = u.T @ y
    scores = []
    for penalty in penalties:
        shrink = s**2 / (s**2 + penalty)
        residual = y - u @ (shrink * projected)
        scores.append(len(y) * float(residual @ residual) / (len(y) - float(shrink.sum())) ** 2)
    return float(penalties[int(np.argmin(scores))])


def standardizer(rows: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pooled ``(mean, std, kept)`` over the training environments' rows; a feature whose pooled
    std is zero is not kept (protocol rule 2: the held-out environment is never used)."""
    pooled = np.concatenate([np.asarray(r, dtype=np.float64) for r in rows])
    mean, std = pooled.mean(axis=0), pooled.std(axis=0)
    return mean, std, std > _FLAT


def _scaled(
    descriptors: Mapping[str, np.ndarray], fit_on: Sequence[str]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    mean, std, kept = standardizer([descriptors[e] for e in fit_on])
    scaled = {
        env: (np.asarray(v, dtype=np.float64)[:, kept] - mean[kept]) / std[kept]
        for env, v in descriptors.items()
    }
    return scaled, kept


def _named(beta: np.ndarray, kept: np.ndarray, names: Sequence[str] | None) -> dict[str, float]:
    labels = list(names) if names is not None else [str(i) for i in range(len(kept))]
    return {label: float(b) for label, b in zip(np.asarray(labels)[kept], beta, strict=True)}


def shift_probe(
    descriptors: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    heldout: str,
    names: Sequence[str] | None = None,
) -> ProbeFit:
    """Research D10: from the shared profile (B1), predict where the held-out environment departs
    from it by its descriptor's departure from the training mean.

    Ridge on the training environments' deviations from their mean (ranks and standardized
    descriptors; with two environments, the 1,024 contrast rows), penalty by GCV; the held-out
    prediction is ``rank(B1) + (R_T − R̄) · β``, so with ``β = 0`` it ranks exactly as B1.
    """
    train = [env for env in targets if env != heldout]
    scaled, kept = _scaled(descriptors, train)
    ranks = {env: average_ranks(targets[env]) for env in train}
    mean_x = np.mean([scaled[e] for e in train], axis=0)
    mean_r = np.mean([ranks[e] for e in train], axis=0)
    # The last environment's deviations are minus the sum of the others': dropped as redundant.
    x = np.concatenate([scaled[e] - mean_x for e in train[:-1]])
    y = np.concatenate([ranks[e] - mean_r for e in train[:-1]])
    base = average_ranks(np.mean([targets[e] for e in train], axis=0))
    if x.shape[1] == 0:
        return ProbeFit(spearman(base, targets[heldout]), None, {}, base)
    penalty = gcv_penalty(x, y)
    beta = ridge_fit(x, y, penalty)
    prediction = base + ridge_predict(scaled[heldout] - mean_x, beta)
    return ProbeFit(spearman(prediction, targets[heldout]), penalty, _named(beta, kept, names),
                    prediction)  # fmt: skip


def direct_probe(
    descriptors: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    heldout: str,
    names: Sequence[str] | None = None,
) -> ProbeFit:
    """Research D7: ridge from standardized descriptors to target ranks; the penalty is chosen by
    leave-one-environment-out over the training environments (fit on the others, Spearman on the
    one left out, mean)."""
    train = [env for env in targets if env != heldout]
    scores = np.zeros(len(PENALTIES))
    for inner in train:
        fit_on = [env for env in train if env != inner]
        scaled, _ = _scaled({e: descriptors[e] for e in [*fit_on, inner]}, fit_on)
        x = np.concatenate([scaled[e] for e in fit_on])
        y = np.concatenate([average_ranks(targets[e]) for e in fit_on])
        for i, penalty in enumerate(PENALTIES):
            if x.shape[1] == 0:
                continue
            found = ridge_predict(scaled[inner], ridge_fit(x, y, penalty))
            scores[i] += _safe_spearman(found, targets[inner]) / len(train)
    penalty = float(PENALTIES[int(np.argmax(scores))])
    scaled, kept = _scaled(descriptors, train)
    x = np.concatenate([scaled[e] for e in train])
    y = np.concatenate([average_ranks(targets[e]) for e in train])
    if x.shape[1] == 0:
        flat = np.zeros(len(targets[heldout]))
        return ProbeFit(_safe_spearman(flat, targets[heldout]), None, {}, flat)
    beta = ridge_fit(x, y, penalty)
    prediction = ridge_predict(scaled[heldout], beta)
    return ProbeFit(_safe_spearman(prediction, targets[heldout]), penalty,
                    _named(beta, kept, names), prediction)  # fmt: skip


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman, 0 for a constant prediction (no ranking at all)."""
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return 0.0
    return spearman(a, b)


def permutation_null(
    probe: Probe,
    descriptors: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    heldout: str,
    permutations: int,
    seed: int = PERMUTATION_SEED,
) -> np.ndarray:
    """Held-out Spearman of the probe refitted (penalty included) after independently permuting
    the descriptor rows within every environment, training and held-out (research D9)."""
    rng = np.random.default_rng(seed)
    null = np.empty(permutations)
    for i in range(permutations):
        shuffled = {env: v[rng.permutation(len(v))] for env, v in sorted(descriptors.items())}
        null[i] = probe(shuffled, targets, heldout).spearman
    return null


def permutation_p(observed: float, null: np.ndarray) -> float:
    """One-sided ``(1 + #{null ≥ observed}) / (1 + n)``."""
    return float((1 + np.sum(np.asarray(null) >= observed)) / (1 + len(null)))


PROBES: dict[str, Callable[..., ProbeFit]] = {"shift": shift_probe, "direct": direct_probe}
PRIMARY = "s1"
SECONDARY = ("s4_correction", "s1/bias")


def allowed_sets(names: Sequence[str]) -> list[tuple[int, DescriptorSet]]:
    """The ladder's sets (numbered from 1) whose columns the descriptor has."""
    return [(i, s) for i, s in enumerate(LADDER, 1) if set(s.columns) <= set(names)]


def target_sets(
    profiles: Mapping[str, Mapping[str, np.ndarray]], seeds: Sequence[int], budgets: Sequence[int]
) -> list[tuple[str, str, dict[str, np.ndarray]]]:
    """``(target, seed, {env: profile})``: the primary target averaged over seeds, then each
    seed's, then the secondary targets (seed-averaged), then S1 per budget."""
    envs = sorted(profiles)
    found = [(PRIMARY, "mean", {e: profiles[e][f"{PRIMARY}/mean"] for e in envs})]
    found += [
        (PRIMARY, str(s), {e: profiles[e][f"{PRIMARY}/seed-{s}/mean"] for e in envs}) for s in seeds
    ]
    found += [(t, "mean", {e: profiles[e][t if "/" in t else f"{t}/mean"] for e in envs})
              for t in SECONDARY]  # fmt: skip
    found += [
        (f"s1/budget-{k}", "mean",
         {e: seed_mean([profiles[e][f"s1/seed-{s}/budget-{k}"] for s in seeds]) for e in envs})
        for k in budgets
    ]  # fmt: skip
    return found


def e2_probes(
    descriptors: Mapping[str, np.ndarray],
    names: Sequence[str],
    profiles: Mapping[str, Mapping[str, np.ndarray]],
    seeds: Sequence[int],
    budgets: Sequence[int],
    permutations: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every probe row of E2 (data model "Probe result") and the primary target's coefficients.

    Per held-out environment and target: B1 (the training environments' mean profile), B2 (the
    held-out environment's ``variance`` profile), and each allowed set in both forms. Only the
    seed-averaged primary target gets permutation tests; Holm runs over the sets evaluated
    within the fold and form.
    """
    sets = allowed_sets(names)
    rows: list[dict[str, Any]] = []
    coefficients: dict[str, Any] = {}
    for heldout in sorted(profiles):
        for target, seed, values in target_sets(profiles, seeds, budgets):
            primary = target == PRIMARY and seed == "mean"
            train = [e for e in sorted(values) if e != heldout]
            b1 = np.mean([values[e] for e in train], axis=0)
            base: dict[str, Any] = {
                "heldout": heldout, "target": target, "seed": seed, "protocol": "—", "form": "—",
                "penalty": None, "p": None, "p_holm": None, "n_permutations": 0,
            }  # fmt: skip
            rows.append({**base, "set": "B1", "name": "dimension identity",
                         "spearman": spearman(b1, values[heldout])})  # fmt: skip
            variance = profiles[heldout]["variance/mean"]
            rows.append({**base, "set": "B2", "name": "variance",
                         "spearman": spearman(variance, values[heldout])})  # fmt: skip
            for form, probe in PROBES.items():
                found: list[dict[str, Any]] = []
                for index, chosen in sets:
                    columns = [list(names).index(c) for c in chosen.columns]
                    subset = {e: descriptors[e][:, columns] for e in sorted(values)}
                    fit = probe(subset, values, heldout, chosen.columns)
                    row = {**base, "set": str(index), "name": chosen.name,
                           "protocol": chosen.protocol, "form": form, "spearman": fit.spearman,
                           "penalty": fit.penalty}  # fmt: skip
                    if primary and permutations:
                        null = permutation_null(probe, subset, values, heldout, permutations)
                        row["p"] = permutation_p(fit.spearman, null)
                        row["n_permutations"] = permutations
                        coefficients.setdefault(heldout, {}).setdefault(chosen.name, {})[form] = (
                            fit.coefficients
                        )
                    found.append(row)
                if primary and permutations:
                    adjusted = holm({r["set"]: r["p"] for r in found})
                    for r in found:
                        r["p_holm"] = adjusted[r["set"]]
                rows += found
    return rows, coefficients
