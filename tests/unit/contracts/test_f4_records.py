"""F4 records: round trips and field rules (specs/005-selector-shift/data-model.md)."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from hyperdime.contracts.schemas import (
    EvaluationResult,
    GateDecision,
    Record,
    SelectorProperties,
    TransferCell,
    dump_record,
    load_record,
)

SHA = "ab" * 32
NPY = {"path": "payload.npy", "sha256": SHA, "format": "npy"}

PROPERTIES: dict[str, Any] = {
    "environment": "scifact",
    "seed": 1,
    "selector_sha256": SHA,
    "battery_sha256": SHA,
    "k_values": [307],
    "mean_importance": NPY,
    "selection_frequency": NPY,
}
CELL: dict[str, Any] = {
    "train_environment": "scifact",
    "test_environment": "nfcorpus",
    "query_set": "test",
    "seeds": [0, 1, 2],
    "k": 307,
    "metrics": {"ndcg@10": 0.3},
    "in_domain_ndcg10": 0.35,
    "degradation": 0.05,
    "p_value": 0.01,
    "p_holm": 0.04,
    "ci_low": 0.01,
    "ci_high": 0.09,
    "overlap_at_k": 0.6,
    "results": ["selector:scifact:s0", "selector:scifact:s1", "selector:scifact:s2"],
}
DECISION: dict[str, Any] = {
    "gate": "G2",
    "outcome": "pass",
    "criteria": [{"name": "c1", "value": 4.0, "threshold": ">= 3 of 6", "met": True}],
    "context": {"fit": "met in 3 of 3 environments"},
    "evidence": ["20261013T000000Z-abcdef01"],
    "adr": "docs/decisions/ADR-0007-g2-selector-shift.md",
    "tag": "g2-selector-shift",
}


@pytest.mark.parametrize(
    ("cls", "fields"),
    [(SelectorProperties, PROPERTIES), (TransferCell, CELL), (GateDecision, DECISION)],
)
def test_f4_records_round_trip(cls: type[Record], fields: dict[str, Any], tmp_path: Path) -> None:
    record = cls(**fields)
    dump_record(record, tmp_path / "record.json")
    assert load_record(cls, tmp_path / "record.json") == record


def test_an_evaluation_result_may_be_a_transfer() -> None:
    result = EvaluationResult(
        environment="nfcorpus",
        protocol="transfer",
        split="test",
        selector="selector:scifact:s0",
        k=307,
        metrics={"ndcg@10": 0.3},
        per_query={"q1": {"ndcg@10": 0.3}},
        num_queries=1,
    )
    assert result.protocol == "transfer"


@pytest.mark.parametrize(
    "change",
    [{"overlap_at_k": 1.5}, {"overlap_at_k": -0.1}, {"p_value": 1.2}, {"p_holm": -0.01}],
)
def test_transfer_cell_rejects_values_outside_their_range(change: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        TransferCell(**{**CELL, **change})


@pytest.mark.parametrize("change", [{"outcome": "maybe"}, {"gate": "G6"}])
def test_gate_decision_rejects_unknown_outcomes_and_gates(change: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        GateDecision(**{**DECISION, **change})


@pytest.mark.parametrize("k", [0, 1025])
def test_selector_properties_reject_budgets_outside_the_dimension(k: int) -> None:
    with pytest.raises(ValidationError):
        SelectorProperties(**{**PROPERTIES, "k_values": [k]})
