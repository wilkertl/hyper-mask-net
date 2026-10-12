"""Config fingerprints: canonical JSON and SHA-256 (research R5)."""

import hashlib

import pytest
from pydantic import BaseModel

from hyperdime.contracts.hashing import NO_DATASET, canonical_json, config_hash
from hyperdime.contracts.validation import ContractError


def test_canonical_json_sorts_keys_and_drops_whitespace() -> None:
    assert canonical_json({"b": 1, "a": [1.5, "x"]}) == '{"a":[1.5,"x"],"b":1}'


def test_config_hash_by_hand() -> None:
    assert config_hash({"a": 1}) == hashlib.sha256(b'{"a":1}').hexdigest()


def test_config_hash_ignores_key_order_and_tracks_values() -> None:
    first = config_hash({"a": 1, "b": {"c": 2}})
    assert first == config_hash({"b": {"c": 2}, "a": 1})
    assert first != config_hash({"a": 1, "b": {"c": 3}})


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_non_finite_values_cannot_be_fingerprinted(value: float) -> None:
    with pytest.raises(ContractError, match="non-finite"):
        config_hash({"a": value})


def test_config_hash_of_a_model_hashes_its_json_dump() -> None:
    class Settings(BaseModel):
        b: int = 2
        a: float = 0.5

    assert config_hash(Settings()) == config_hash({"a": 0.5, "b": 2})


def test_no_dataset_marker() -> None:
    assert NO_DATASET == "none"
