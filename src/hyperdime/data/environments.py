"""The registry of environments: pinned sources, splits, and query sets (F2 research D1)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from hyperdime.contracts.config import DataConfig
from hyperdime.contracts.schemas import Id
from hyperdime.contracts.validation import ConfigError, ContractError

Revision = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]


class _Spec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class EvalQuerySet(_Spec):
    """An evaluation-only query set over an environment's corpus (e.g. TREC DL 2019)."""

    name: Id
    loader: Literal["ir_datasets"]
    source: Annotated[str, StringConstraints(min_length=1)]


class EnvironmentSpec(_Spec):
    spec_version: Literal[1] = 1
    environment: Id
    loader: Literal["beir_hf"] = "beir_hf"
    source: Annotated[str, StringConstraints(min_length=1)]
    source_revision: Revision
    qrels_source: Annotated[str, StringConstraints(min_length=1)]
    qrels_revision: Revision
    train_split: str | None
    test_split: Annotated[str, StringConstraints(min_length=1)]
    train_query_cap: Annotated[int, Field(ge=1)] | None = None
    corpus_group: Id
    eval_query_sets: tuple[EvalQuerySet, ...] = ()
    instruction: Annotated[str, StringConstraints(min_length=1)]
    license: Literal["public", "licensed"] = "public"

    @property
    def has_train_qrels(self) -> bool:
        return self.train_split is not None


ENVIRONMENTS: Mapping[str, EnvironmentSpec] = {
    spec.environment: spec
    for spec in (
        EnvironmentSpec(
            environment="scifact",
            source="BeIR/scifact",
            source_revision="b3b5335604bf5ee3c4447671af975ea25143d4f5",
            qrels_source="BeIR/scifact-qrels",
            qrels_revision="2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029",
            train_split="train",
            test_split="test",
            corpus_group="scifact",
            instruction=(
                "Given a scientific claim, retrieve documents that support or refute the claim"
            ),
        ),
        EnvironmentSpec(
            environment="nfcorpus",
            source="BeIR/nfcorpus",
            source_revision="b5026a0e96e8a7ac4f95f482a596389289d46269",
            qrels_source="BeIR/nfcorpus-qrels",
            qrels_revision="a451b3b26d3ae1358f259c1a3a4dd61fcea35a65",
            train_split="train",
            test_split="test",
            corpus_group="nfcorpus",
            instruction=(
                "Given a question, retrieve relevant documents that best answer the question"
            ),
        ),
        EnvironmentSpec(
            environment="msmarco",
            source="BeIR/msmarco",
            source_revision="a918e0d11a77ed33f42f29d98340b655593b96ad",
            qrels_source="BeIR/msmarco-qrels",
            qrels_revision="253fbf8a3f8d4a0932b63882b5162bedc84779f5",
            train_split="train",
            test_split="dev",
            train_query_cap=50_000,
            corpus_group="msmarco",
            eval_query_sets=(
                EvalQuerySet(
                    name="trec-dl-2019",
                    loader="ir_datasets",
                    source="msmarco-passage/trec-dl-2019/judged",
                ),
                EvalQuerySet(
                    name="trec-dl-2020",
                    loader="ir_datasets",
                    source="msmarco-passage/trec-dl-2020/judged",
                ),
                EvalQuerySet(
                    name="trec-dl-hard", loader="ir_datasets", source="msmarco-passage/trec-dl-hard"
                ),
            ),
            instruction=(
                "Given a web search query, retrieve relevant passages that answer the query"
            ),
        ),
    )
}


def environment_spec(name: str) -> EnvironmentSpec:
    try:
        return ENVIRONMENTS[name]
    except KeyError:
        raise ContractError(
            f"unknown environment {name!r}; known: {sorted(ENVIRONMENTS)}"
        ) from None


def spec_for(data: DataConfig) -> EnvironmentSpec:
    """The registry entry ``data`` selects; its source and instruction must agree with it."""
    spec = environment_spec(data.environment)
    if data.source != spec.source:
        raise ConfigError(f"data.source: {data.source!r} does not match {spec.source!r}")
    if data.instruction != spec.instruction:
        raise ConfigError(f"data.instruction: does not match the {spec.environment} registry entry")
    return spec
