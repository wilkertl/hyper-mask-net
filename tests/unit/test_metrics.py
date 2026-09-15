import math

import pytest

from hyperdime.evaluation.metrics import evaluate_run, query_metrics


def test_query_metrics_match_manual_computation() -> None:
    values = query_metrics(["d1", "d2", "d3"], {"d2": 1, "d3": 2, "d9": 1, "d0": 0}, [3])
    dcg = 1 / math.log2(3) + 2 / math.log2(4)
    idcg = 2 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert values["ndcg@3"] == pytest.approx(dcg / idcg)
    assert values["recall@3"] == pytest.approx(2 / 3)
    assert values["mrr@3"] == pytest.approx(1 / 2)


def test_evaluate_run_averages_judged_queries_and_skips_unjudged() -> None:
    run = {"q1": ["a", "b"], "q2": ["c", "d"], "q3": ["e"]}
    qrels = {"q1": {"a": 1}, "q2": {"d": 1}, "q3": {"e": 0}}
    means, per_query = evaluate_run(run, qrels, cutoffs=[2])
    assert set(per_query) == {"q1", "q2"}
    assert means["mrr@2"] == pytest.approx((1 + 1 / 2) / 2)
    assert means["recall@2"] == pytest.approx(1.0)


def test_evaluate_run_requires_rankings_for_judged_queries() -> None:
    with pytest.raises(KeyError):
        evaluate_run({}, {"q1": {"a": 1}})
