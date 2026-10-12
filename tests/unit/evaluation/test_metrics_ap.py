"""AP and TREC relevance thresholds, checked by hand and against ir-measures (F3 research D2)."""

import random

import ir_measures
import pytest
from ir_measures import AP, RR, R, nDCG

from hyperdime.evaluation.metrics import query_metrics

RANKING = ["c", "a", "x", "b", "y"]
JUDGMENTS = {"a": 2, "b": 1, "c": 0}


def test_average_precision_by_hand() -> None:
    # relevant (>= 1): a at rank 2, b at rank 4 -> (1/2 + 2/4) / 2
    assert query_metrics(RANKING, JUDGMENTS, (10,))["ap"] == pytest.approx(0.5)


def test_threshold_two_drops_grade_one_from_binary_metrics_only() -> None:
    graded = query_metrics(RANKING, JUDGMENTS, (10,), relevance_threshold=2)
    # only a counts: AP = 1/2, MRR@10 = 1/2, Recall@10 = 1; nDCG still uses grades 2 and 1
    assert graded["ap"] == pytest.approx(0.5)
    assert graded["mrr@10"] == pytest.approx(0.5)
    assert graded["recall@10"] == pytest.approx(1.0)
    assert graded["ndcg@10"] == query_metrics(RANKING, JUDGMENTS, (10,))["ndcg@10"]


def test_defaults_keep_the_previous_metrics() -> None:
    values = query_metrics(RANKING, JUDGMENTS, (10, 100))
    assert values["recall@100"] == pytest.approx(1.0) and values["mrr@10"] == pytest.approx(0.5)


@pytest.mark.parametrize("threshold", [1, 2])
def test_agreement_with_ir_measures(threshold: int) -> None:
    rng = random.Random(0)
    qrels, run, mine = [], [], {}
    for q in range(20):
        qid = f"q{q}"
        docs = [f"d{i}" for i in range(150)]
        judged = {d: rng.choice([0, 1, 2, 3]) for d in rng.sample(docs, 12)}
        ranking = rng.sample(docs, 120)
        qrels += [ir_measures.Qrel(qid, d, g) for d, g in judged.items()]
        run += [ir_measures.ScoredDoc(qid, d, float(120 - i)) for i, d in enumerate(ranking)]
        if any(g >= threshold for g in judged.values()):
            mine[qid] = query_metrics(ranking, judged, (10, 100), relevance_threshold=threshold)
    measures = [
        nDCG @ 10,
        RR(rel=threshold) @ 10,
        RR(rel=threshold) @ 100,
        R(rel=threshold) @ 100,
        AP(rel=threshold),
    ]
    reference = {
        (m.query_id, str(m.measure)): m.value for m in ir_measures.iter_calc(measures, qrels, run)
    }
    names = {
        "ndcg@10": nDCG @ 10,
        "mrr@10": RR(rel=threshold) @ 10,
        "mrr@100": RR(rel=threshold) @ 100,
        "recall@100": R(rel=threshold) @ 100,
        "ap": AP(rel=threshold),
    }
    for qid, values in mine.items():
        for name, measure in names.items():
            assert values[name] == pytest.approx(reference[(qid, str(measure))], abs=1e-6), (
                qid,
                name,
            )
