"""Aggregation of the ranking diagnostic's report (ADR-0016)."""

import numpy as np
import pytest

from hyperdime.evaluation.representation_ranking import ROWS
from hyperdime.reports.p3_ranking import (
    ALL_LABEL_FAMILY,
    PRIMARY_FAMILY,
    RESAMPLES,
    Environment,
    claims,
    contrast,
    interval,
    resampled_queries,
    weighted_mean,
)


def _environment(delta: float, queries: int = 40) -> Environment:
    budgets = [102]
    shape = (len(ROWS), 2, queries)
    names = ("ndcg@10", "mrr@10", "recall@100", "rr", "base_rr", "logrank", "base_logrank",
             "recall10", "base_recall10", "recall100", "base_recall100")  # fmt: skip
    arrays = {name: np.full(shape, np.nan) for name in names}
    base = np.linspace(0.2, 0.6, queries)
    for name in ("ndcg@10", "mrr@10", "recall@100"):
        arrays[name][ROWS.index("full"), 1] = base
        arrays[name][ROWS.index("shared"), 0] = base + delta
        arrays[name][ROWS.index("1-1"), 0] = base
    for name in ("rr", "logrank", "recall10", "recall100"):
        arrays[name][ROWS.index("1-1"), 0] = base + delta
        arrays[f"base_{name}"][ROWS.index("1-1"), 0] = base
    strata = np.zeros(queries, dtype=np.int64)
    return Environment(
        "env", arrays, np.ones(queries), resampled_queries(strata, RESAMPLES, 0), budgets, {}
    )


def test_resamples_stay_within_strata() -> None:
    strata = np.array([0, 0, 1, 1, 1])
    found = resampled_queries(strata, 200, 3)
    assert found.shape == (200, 5)
    assert np.isin(found[:, :2], [0, 1]).all()
    assert np.isin(found[:, 2:], [2, 3, 4]).all()


def test_resamples_of_a_large_stratified_environment_are_valid_positions() -> None:
    # The shape of the NFCorpus selection, where one report run stopped on an out-of-range index
    # that no later run repeated.
    strata = np.repeat([0, 1, 2], [1000, 900, 262])
    found = resampled_queries(strata, RESAMPLES, 1)
    assert found.dtype == np.int64
    assert found.min() >= 0
    assert found.max() < strata.size
    assert np.array_equal(found, resampled_queries(strata, RESAMPLES, 1))


def test_weighted_mean_ignores_ineligible_queries() -> None:
    values = np.array([1.0, np.nan, 3.0])
    assert weighted_mean(values, np.array([1.0, 5.0, 3.0])) == pytest.approx((1 + 9) / 4)


def test_contrast_compares_only_queries_with_both_values() -> None:
    environment = _environment(0.1, 10)
    values = np.ones(10)
    base = np.zeros(10)
    values[0] = np.nan
    found = contrast(environment, values, base)
    assert found["queries"] == 9
    assert found["delta"] == pytest.approx(1.0)
    assert found["low"] == found["high"] == pytest.approx(1.0)


def test_interval_covers_a_mean_and_widens_with_spread() -> None:
    rng = np.random.default_rng(0)
    draws = rng.normal(0.5, 1.0, 200)
    resamples = resampled_queries(np.zeros(200, np.int64), RESAMPLES, 1)
    low, high = interval(draws, np.ones(200), resamples)
    assert low < draws.mean() < high
    assert high - low == pytest.approx(2 * 1.96 * draws.std() / np.sqrt(200), rel=0.2)


def test_held_out_and_all_label_claims_are_separate_holm_families() -> None:
    table = claims([_environment(0.2)])
    header, rows = table[0], table[1:]
    family = header.index("family")
    raw = header.index("p (sign-flip)")
    adjusted = header.index("p (Holm, within family)")
    primary = [row for row in rows if row[family] == PRIMARY_FAMILY]
    secondary = [row for row in rows if row[family] == ALL_LABEL_FAMILY]
    assert len(primary) == 4
    assert len(secondary) == 6
    assert {row[header.index("arm")] for row in primary} == {"1-1"}
    assert all(float(row[adjusted]) >= float(row[raw]) for row in rows)
