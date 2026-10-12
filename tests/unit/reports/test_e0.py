"""G1 criteria and significance from hand-built results (F3 US4)."""

from hyperdime.contracts.schemas import EvaluationResult
from hyperdime.reports.e0 import EnvironmentResults, g1, render, significance


def _result(method: str, k: int, values: list[float]) -> EvaluationResult:
    per_query = {f"q{i}": {"ndcg@10": v, "ap": v} for i, v in enumerate(values)}
    return EvaluationResult(
        environment="toy", protocol="in-domain", split="test", selector=method, k=k,
        metrics={"ndcg@10": sum(values) / len(values), "ap": sum(values) / len(values)},
        per_query=per_query, num_queries=len(values),
    )  # fmt: skip


def _env(
    oracle: float = 0.9, selector: float = 0.6, rand: float = 0.3, kl: float = 1.0
) -> EnvironmentResults:
    values = {"oracle": oracle, "l2s": selector, "random": rand, "prefix": 0.5}
    test = {("full", 1024): _result("full", 1024, [0.7] * 6)}
    test.update({(m, k): _result(m, k, [v] * 6) for m, v in values.items() for k in (256, 307)})
    official = {
        "chosen": "c",
        "cells": {"c": {"val_kl": kl, "uniform_kl": 2.0, "validation_ndcg@10": 0.5}},
    }
    checks = {"test": {"k1024_identical": {"full": True, "l2s": True}}}
    return EnvironmentResults(
        "toy", "run", "a" * 40, {"test": test}, checks, official, None, {"official": "r"}
    )


def test_g1_passes_when_every_criterion_holds() -> None:
    assert [c["met"] for c in g1([_env()], True)] == [True, True, True, True]


def test_g1_fails_on_each_criterion() -> None:
    assert not g1([_env(kl=3.0)], True)[0]["met"]
    broken = _env()
    broken.checks["test"]["k1024_identical"]["l2s"] = False
    assert not g1([broken], True)[1]["met"]
    assert not g1([_env(selector=0.2)], True)[2]["met"]
    assert not g1([_env()], False)[3]["met"]


def test_significance_is_holm_adjusted_and_signed() -> None:
    rows = significance([_env()])
    assert {(r["method"], r["against"]) for r in rows} >= {("l2s", "full"), ("oracle", "l2s")}
    assert all(r["p_holm"] >= r["p"] for r in rows)
    assert next(r for r in rows if (r["method"], r["against"]) == ("oracle", "full"))["diff"] > 0


def test_render_is_deterministic() -> None:
    assert render([_env()], True) == render([_env()], True)
