"""The environment registry: pinned sources and consistency with configs (research D1)."""

import pytest

from hyperdime.contracts.config import DataConfig
from hyperdime.contracts.validation import ConfigError, ContractError
from hyperdime.data.environments import ENVIRONMENTS, environment_spec, spec_for


def test_registry_pins_every_source() -> None:
    assert set(ENVIRONMENTS) == {"scifact", "nfcorpus", "msmarco"}
    for spec in ENVIRONMENTS.values():
        assert len(spec.source_revision) == 40 and len(spec.qrels_revision) == 40
        assert spec.has_train_qrels and spec.corpus_group == spec.environment


def test_msmarco_has_the_trec_query_sets_and_a_training_cap() -> None:
    spec = environment_spec("msmarco")
    assert [s.name for s in spec.eval_query_sets] == [
        "trec-dl-2019",
        "trec-dl-2020",
        "trec-dl-hard",
    ]
    assert spec.train_query_cap == 50_000 and spec.test_split == "dev"


def test_unknown_environment_is_refused() -> None:
    with pytest.raises(ContractError, match="unknown environment 'robust04'"):
        environment_spec("robust04")


def test_config_source_must_match_the_registry() -> None:
    scifact = environment_spec("scifact")
    data = DataConfig(
        environment="scifact",
        source="BeIR/scifact",
        instruction=scifact.instruction,
        validation_fraction=0.1,
    )
    assert spec_for(data) is scifact
    with pytest.raises(ConfigError, match="data.source"):
        spec_for(data.model_copy(update={"source": "BeIR/other"}))
