"""Hydra-composed configs validated into RunConfig (research R2, R12; spec US3)."""

import os
from pathlib import Path

import pytest

from hyperdime.contracts.config import RunConfig, load_config
from hyperdime.contracts.hashing import config_hash
from hyperdime.contracts.validation import ConfigError

FIXTURES = Path(__file__).parents[2] / "fixtures" / "configs"


def test_sample_config_loads_and_validates() -> None:
    config = load_config(FIXTURES / "valid", "experiment/diagnose")
    assert isinstance(config, RunConfig)
    assert config.config_version == 1
    assert config.model.embedding_dim == 1024 and config.model.frozen
    assert config.data.environment == "scifact"


def test_repository_configs_match_the_fixture() -> None:
    root = Path(__file__).parents[3] / "configs"
    assert load_config(root, "experiment/diagnose") == load_config(
        FIXTURES / "valid", "experiment/diagnose"
    )


@pytest.mark.parametrize(
    ("name", "field"),
    [
        ("missing_revision", "model.revision"),
        ("unknown_key", "model.unknown_key"),
        ("wrong_dim", "model.embedding_dim"),
        ("not_frozen", "model.frozen"),
        ("branch_revision", "model.revision"),
    ],
)
def test_invalid_configs_name_the_field(name: str, field: str) -> None:
    with pytest.raises(ConfigError) as error:
        load_config(FIXTURES / "invalid", f"experiment/{name}")
    assert str(error.value).startswith(f"{field}: ")


def test_overrides_apply_and_equal_values_hash_the_same() -> None:
    base = load_config(FIXTURES / "valid", "experiment/diagnose", ["run.seed=7"])
    assert base.run.seed == 7
    scientific = load_config(
        FIXTURES / "valid", "experiment/diagnose", ["run.seed=7", "data.validation_fraction=1e-1"]
    )
    assert config_hash(scientific) == config_hash(base)


def test_unknown_override_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match=r"model\.unknown_key"):
        load_config(FIXTURES / "valid", "experiment/diagnose", ["+model.unknown_key=1"])


def test_loading_leaves_the_working_directory_and_hydra_state_alone() -> None:
    before = os.getcwd()
    load_config(FIXTURES / "valid", "experiment/diagnose")
    load_config(FIXTURES / "valid", "experiment/diagnose")
    assert os.getcwd() == before
