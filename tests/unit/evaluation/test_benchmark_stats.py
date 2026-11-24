"""F12 statistics on hand data (T030)."""

from __future__ import annotations

import numpy as np

from hyperdime.evaluation.benchmark_stats import (
    bootstrap_means,
    claim_level,
    holm,
    intervals,
    sign_flip_p,
    stream,
)


def test_exact_sign_flip_when_the_enumeration_is_small() -> None:
    # 3 positive differences: only the all-plus vector reaches the observed mean -> 1/8
    assert (
        sign_flip_p(np.array([1.0, 2.0, 3.0]), stream("signflip", "scifact", "positive")) == 1 / 8
    )


def test_monte_carlo_p_uses_plus_one_and_is_reproducible() -> None:
    diffs = np.linspace(-0.5, 1.0, 40)
    a = sign_flip_p(diffs, stream("signflip", "scifact", "positive"), draws=2_000)
    b = sign_flip_p(diffs, stream("signflip", "scifact", "positive"), draws=2_000)
    assert a == b and a >= 1 / 2_001
    zero = sign_flip_p(np.zeros(40), stream("signflip", "scifact", "positive"), draws=500)
    assert zero == 1.0  # every flip ties the observed mean


def test_streams_are_independent_per_procedure() -> None:
    assert (
        stream("bootstrap", "scifact", "positive").random()
        != stream("signflip", "scifact", "positive").random()
    )


def test_bootstrap_bounds_order_and_constant_data() -> None:
    means = bootstrap_means(
        np.full(30, 0.01), stream("bootstrap", "scifact", "positive"), draws=1_000
    )
    bounds = intervals(means, family_size=9)
    assert np.isclose(bounds["pointwise_low"], 0.01) and np.isclose(
        bounds["bonferroni_lower"], 0.01
    )
    spread = intervals(
        bootstrap_means(np.arange(30.0), stream("bootstrap", "scifact", "contrastive"), 5_000), 9
    )
    assert spread["bonferroni_lower"] < spread["pointwise_low"] < spread["pointwise_high"]


def test_holm_by_hand() -> None:
    assert holm({"a": 0.01, "b": 0.04, "c": 0.03}) == {"a": 0.03, "c": 0.06, "b": 0.06}


def test_claim_levels() -> None:
    assert claim_level(0.01, None, 0.0).startswith("pending")
    assert claim_level(0.003, 0.01, -0.001) == "superior"
    assert claim_level(0.006, 0.01, 0.001) == "practical candidate"
    assert claim_level(0.02, 0.01, 0.006) == "established minimum gain"
    assert claim_level(0.02, 0.2, 0.006) == "not superior"
