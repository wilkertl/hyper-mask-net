"""The S1 decision reuses the gate decision record (F5 research D9)."""

import pytest
from pydantic import ValidationError

from hyperdime.contracts.schemas import GateDecision

S1 = {
    "gate": "S1",
    "outcome": "fail",
    "criteria": [{"name": "SC-001", "value": 1.0, "threshold": ">= 2 of 3", "met": False}],
    "context": {},
    "evidence": ["20261015T000000Z-abcdef01"],
    "adr": "docs/decisions/ADR-0009-s1-selector-objective.md",
    "tag": "s1-selector-objective",
}


def test_an_s1_decision_round_trips() -> None:
    record = GateDecision(**S1)
    assert GateDecision.model_validate_json(record.model_dump_json()) == record


def test_an_unknown_experiment_is_still_rejected() -> None:
    with pytest.raises(ValidationError):
        GateDecision(**{**S1, "gate": "E9"})
