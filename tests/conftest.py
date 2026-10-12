"""Shared fakes: an in-memory BEIR-shaped source and evaluation query set (no network)."""

from collections.abc import Iterator
from dataclasses import dataclass, field

import pytest

from hyperdime.contracts.config import DataConfig
from hyperdime.data.environments import EnvironmentSpec, EvalQuerySet

Qrels = dict[str, dict[str, int]]


@dataclass
class FakeSource:
    corpus: list[tuple[str, str, str]]
    queries: list[tuple[str, str]]
    splits: dict[str, Qrels]
    corpus_reads: int = 0

    def iter_corpus(self) -> Iterator[tuple[str, str, str]]:
        self.corpus_reads += 1
        yield from self.corpus

    def iter_queries(self) -> Iterator[tuple[str, str]]:
        yield from self.queries

    def qrels(self, split: str) -> Qrels:
        return {q: dict(labels) for q, labels in self.splits[split].items()}


@dataclass
class FakeQuerySet:
    queries: list[tuple[str, str]]
    judgments: Qrels = field(default_factory=dict)

    def iter_queries(self) -> Iterator[tuple[str, str]]:
        yield from self.queries

    def qrels(self) -> Qrels:
        return {q: dict(labels) for q, labels in self.judgments.items()}


def toy_spec(*, eval_sets: bool = True, cap: int | None = None) -> EnvironmentSpec:
    return EnvironmentSpec(
        environment="toy",
        source="BeIR/toy",
        source_revision="a" * 40,
        qrels_source="BeIR/toy-qrels",
        qrels_revision="b" * 40,
        train_split="train",
        test_split="test",
        train_query_cap=cap,
        corpus_group="toy",
        eval_query_sets=(EvalQuerySet(name="evalset", loader="ir_datasets", source="toy/eval"),)
        if eval_sets
        else (),
        instruction="Find it",
    )


def toy_data(spec: EnvironmentSpec) -> DataConfig:
    return DataConfig(
        environment=spec.environment,
        source=spec.source,
        instruction=spec.instruction,
        validation_fraction=0.1,
    )


def toy_source() -> FakeSource:
    """12 documents plus one empty one; 30 queries; 20 judged for train, 10 for test.

    q0 also has a judgment to a missing document (dropped); q19's only judgment is to a missing
    document, so q19 is excluded from training.
    """
    corpus = [(f"d{i}", f"Title {i}" if i % 2 else "", f"text of document {i}") for i in range(12)]
    corpus.append(("d12", "", "   "))
    queries = [(f"q{i}", f"query number {i}") for i in range(30)]
    train: Qrels = {f"q{i}": {f"d{i % 12}": 1} for i in range(19)}
    train["q0"]["missing"] = 1
    train["q19"] = {"missing": 2}
    test: Qrels = {f"q{i}": {f"d{i % 12}": 2, f"d{(i + 1) % 12}": 0} for i in range(20, 30)}
    return FakeSource(corpus, queries, {"train": train, "test": test})


def toy_eval_sets() -> dict[str, FakeQuerySet]:
    return {
        "evalset": FakeQuerySet(
            [("e0", "eval query 0"), ("e1", "eval query 1"), ("e2", "eval query 2")],
            {"e0": {"d1": 3, "d2": 0}, "e1": {"d4": 1}, "e2": {"missing": 1}},
        )
    }


@pytest.fixture
def source() -> FakeSource:
    return toy_source()
