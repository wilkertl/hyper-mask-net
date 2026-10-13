"""Transfer of selectors across environments and their fit to targets (F4 research D2–D5).

Seeds are averaged per query before any paired test, so each ordered (train, test) pair is one
test over the test environment's queries. Nothing here reads a judgment: callers pass per-query
metrics, oracle scores, or query ids.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import torch
from torch import Tensor

from hyperdime.contracts.schemas import TransferCell
from hyperdime.evaluation.selector_shift import overlap_at_k
from hyperdime.evaluation.statistics import holm, paired_bootstrap_ci, paired_randomization_test
from hyperdime.selection.topk import top_k_mask

ALPHA = 0.05


def fit_gain(
    selector_scores: Sequence[Tensor], global_scores: Tensor, oracle_scores: Tensor, k: int
) -> Tensor:
    """Per-query gain of a selector's seeds over global importance at matching the oracle mask.

    ``gain(q) = mean_s overlap@k(selector_s, oracle) - overlap@k(global, oracle)`` (research D2).
    """
    oracle = top_k_mask(oracle_scores, k)
    seeds = torch.stack([overlap_at_k(top_k_mask(s, k), oracle) for s in selector_scores])
    return seeds.mean(dim=0) - overlap_at_k(top_k_mask(global_scores, k), oracle)


def fits(mean_gain: float, p_holm: float, alpha: float = ALPHA) -> bool:
    """The ranking-fit rule: a positive mean gain with a Holm-adjusted p at or below alpha."""
    return mean_gain > 0 and p_holm <= alpha


def seed_average(
    per_query: Sequence[Mapping[str, Mapping[str, float]]],
) -> dict[str, dict[str, float]]:
    """Per-query mean over seeds of every metric; every seed must cover the same queries."""
    queries = set(per_query[0])
    if any(set(seed) != queries for seed in per_query):
        raise ValueError("seeds must be scored on the same queries")
    return {
        qid: {
            metric: sum(seed[qid][metric] for seed in per_query) / len(per_query)
            for metric in per_query[0][qid]
        }
        for qid in sorted(queries)
    }


@dataclass(frozen=True)
class Degradation:
    """In-domain minus cross-domain nDCG@10 of one ordered pair on one query set (research D4)."""

    train: str
    test: str
    query_set: str
    in_domain: float
    cross: float
    delta: float
    p_value: float | None
    p_holm: float | None
    ci: tuple[float, float] | None


Key = tuple[str, str, str]  # (train environment, test environment, query set)


def degradation_tests(
    scores: Mapping[Key, Mapping[str, float]],
    gated_set: str = "test",
    n: int = 10_000,
    seed: int = 0,
) -> list[Degradation]:
    """One paired test per ordered pair; Holm over the cross-environment pairs of ``gated_set``.

    ``scores`` maps (train, test, query set) to seed-averaged per-query nDCG@10; each query set
    needs its in-domain entry (test, test, set).
    """
    rows: dict[Key, Degradation] = {}
    for key in sorted(scores):
        train, test, query_set = key
        own = scores[(test, test, query_set)]
        cell = scores[key]
        if set(cell) != set(own):
            raise ValueError(f"{key} is not scored on the in-domain queries")
        qids = sorted(own)
        a, b = [own[q] for q in qids], [cell[q] for q in qids]
        mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
        if train == test:
            rows[key] = Degradation(train, test, query_set, mean_a, mean_b, 0.0, None, None, None)
            continue
        rows[key] = Degradation(
            train, test, query_set, mean_a, mean_b, mean_a - mean_b,
            paired_randomization_test(a, b, n, seed), None, paired_bootstrap_ci(a, b, n, seed=seed),
        )  # fmt: skip
    gated = {
        f"{k[0]}->{k[1]}": r.p_value
        for k, r in rows.items()
        if k[2] == gated_set and r.p_value is not None
    }
    adjusted = holm(gated)
    return [
        replace(row, p_holm=adjusted.get(f"{key[0]}->{key[1]}")) if key[2] == gated_set else row
        for key, row in rows.items()
    ]


def criterion_1(
    rows: Sequence[Degradation], gated_set: str = "test", alpha: float = ALPHA
) -> tuple[int, int, bool]:
    """G2 criterion 1: cross-domain significantly below in-domain for at least half of the
    ordered pairs. Returns (significant drops, pairs, holds)."""
    pairs = [r for r in rows if r.query_set == gated_set and r.train != r.test]
    drops = sum(1 for r in pairs if r.delta > 0 and r.p_holm is not None and r.p_holm <= alpha)
    return drops, len(pairs), bool(pairs) and 2 * drops >= len(pairs)


@dataclass(frozen=True)
class Battery:
    """Unlabeled test queries drawn equally from every environment (research D5)."""

    ids: dict[str, list[str]]
    n: int
    seed: int
    sha256: str


def sample_battery(ids: Mapping[str, Sequence[str]], seed: int = 0) -> Battery:
    """``n`` = the smallest environment's count; each environment's ids are sorted, then ``n`` are
    drawn without replacement with a fresh generator seeded ``seed``, and kept sorted."""
    n = min(len(qids) for qids in ids.values())
    chosen = {}
    for env in sorted(ids):
        pool = sorted(ids[env])
        order = torch.randperm(len(pool), generator=torch.Generator().manual_seed(seed))
        chosen[env] = sorted(pool[i] for i in order[:n].tolist())
    flat = sorted(f"{env}/{q}" for env, qids in chosen.items() for q in qids)
    digest = hashlib.sha256("\n".join(flat).encode()).hexdigest()
    return Battery(chosen, n, seed, digest)


def transfer_cell(
    row: Degradation,
    metrics: Mapping[str, float],
    overlap: float,
    seeds: Sequence[int],
    k: int,
    results: Sequence[str],
) -> TransferCell:
    """The ``TransferCell`` record of one tested pair; ``metrics`` are seed-averaged means."""
    return TransferCell(
        train_environment=row.train,
        test_environment=row.test,
        query_set=row.query_set,
        seeds=list(seeds),
        k=k,
        metrics=dict(metrics),
        in_domain_ndcg10=row.in_domain,
        degradation=row.delta,
        p_value=row.p_value,
        p_holm=row.p_holm,
        ci_low=None if row.ci is None else row.ci[0],
        ci_high=None if row.ci is None else row.ci[1],
        overlap_at_k=overlap,
        results=list(results),
    )
