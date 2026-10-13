"""Seed averaging, degradation tests, and G2 criterion 1 (F4 research D4)."""

import pytest
import torch

from hyperdime.evaluation.transfer import criterion_1, degradation_tests, seed_average


def test_seed_average_by_hand() -> None:
    seeds = [
        {"q1": {"ndcg@10": 0.0}, "q2": {"ndcg@10": 0.3}},
        {"q1": {"ndcg@10": 0.3}, "q2": {"ndcg@10": 0.6}},
        {"q1": {"ndcg@10": 0.6}, "q2": {"ndcg@10": 0.0}},
    ]
    averaged = seed_average(seeds)
    assert averaged["q1"]["ndcg@10"] == pytest.approx(0.3)
    assert averaged["q2"]["ndcg@10"] == pytest.approx(0.3)


def test_seed_average_refuses_seeds_scored_on_other_queries() -> None:
    with pytest.raises(ValueError, match="same queries"):
        seed_average([{"q1": {"ndcg@10": 0.1}}, {"q2": {"ndcg@10": 0.1}}])


def _values(shift: float, n: int = 60) -> tuple[dict[str, float], dict[str, float]]:
    base = torch.rand(n, generator=torch.Generator().manual_seed(0)).tolist()
    noise = (torch.rand(n, generator=torch.Generator().manual_seed(1)) * 0.02).tolist()
    in_domain = {f"q{i}": v for i, v in enumerate(base)}
    cross = {f"q{i}": v - shift + e for i, (v, e) in enumerate(zip(base, noise, strict=True))}
    return in_domain, cross


def test_a_known_shift_is_recovered_with_its_interval() -> None:
    in_domain, cross = _values(shift=0.1)
    (row,) = [
        r
        for r in degradation_tests({("b", "b", "test"): in_domain, ("a", "b", "test"): cross})
        if r.train == "a"
    ]
    expected = sum(in_domain[q] - cross[q] for q in in_domain) / len(in_domain)
    assert row.delta == pytest.approx(expected)
    assert row.p_value is not None and row.p_value <= 0.05
    assert row.ci is not None and row.ci[0] <= expected <= row.ci[1]


def test_no_shift_is_not_significant() -> None:
    in_domain, _ = _values(shift=0.0)
    rows = degradation_tests({("b", "b", "test"): in_domain, ("a", "b", "test"): dict(in_domain)})
    cross = next(r for r in rows if r.train == "a")
    assert cross.delta == 0 and cross.p_value is not None and cross.p_value > 0.05


def _six_pairs(shifted: set[tuple[str, str]]) -> dict[tuple[str, str, str], dict[str, float]]:
    in_domain, cross = _values(shift=0.1)
    scores = {(e, e, "test"): in_domain for e in "abc"}
    for train in "abc":
        for test in "abc":
            if train != test:
                scores[(train, test, "test")] = cross if (train, test) in shifted else in_domain
    scores[("a", "c", "trec")] = cross
    scores[("c", "c", "trec")] = in_domain
    return scores


def test_holm_adjusts_over_exactly_the_six_ordered_pairs() -> None:
    rows = degradation_tests(_six_pairs({("a", "b")}))
    gated = {(r.train, r.test): r for r in rows if r.query_set == "test" and r.train != r.test}
    pair = gated[("a", "b")]
    assert (len(gated), pair.p_holm) == (6, pytest.approx(min(1.0, 6 * pair.p_value)))


def test_evaluation_sets_are_tested_but_outside_the_holm_family() -> None:
    rows = degradation_tests(_six_pairs({("a", "b")}))
    trec = next(r for r in rows if r.query_set == "trec" and r.train == "a")
    assert trec.p_value is not None and trec.p_value <= 0.05 and trec.p_holm is None


def test_in_domain_rows_have_no_test() -> None:
    rows = degradation_tests(_six_pairs(set()))
    in_domain = {(r.delta, r.p_value, r.p_holm, r.ci) for r in rows if r.train == r.test}
    assert in_domain == {(0.0, None, None, None)}


def test_criterion_1_counts_significant_drops_and_holds_at_half() -> None:
    three = {("a", "b"), ("b", "c"), ("c", "a")}
    assert criterion_1(degradation_tests(_six_pairs(three))) == (3, 6, True)
    assert criterion_1(degradation_tests(_six_pairs({("a", "b"), ("b", "c")}))) == (2, 6, False)


def test_criterion_1_ignores_significant_gains() -> None:
    scores = _six_pairs(set())
    in_domain, cross = _values(shift=0.1)
    scores[("a", "b", "test")], scores[("b", "b", "test")] = in_domain, cross  # cross is better
    assert criterion_1(degradation_tests(scores))[0] == 0
