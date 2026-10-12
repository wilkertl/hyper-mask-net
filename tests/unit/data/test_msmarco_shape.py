"""MS MARCO's shape: a `dev` test split, three evaluation query sets, a training cap (F2 US3)."""

from pathlib import Path

import pytest

from hyperdime.contracts.manifests import ArtifactWriter, start_manifest
from hyperdime.contracts.schemas import DatasetManifest
from hyperdime.data.environments import EnvironmentSpec, EvalQuerySet
from hyperdime.data.prepare import load_environment, prepare_environment

from conftest import FakeQuerySet, FakeSource, toy_data, toy_source

SETS = ("trec-dl-2019", "trec-dl-2020", "trec-dl-hard")


def _spec() -> EnvironmentSpec:
    return EnvironmentSpec(
        environment="marco",
        source="BeIR/marco",
        source_revision="a" * 40,
        qrels_source="BeIR/marco-qrels",
        qrels_revision="b" * 40,
        train_split="train",
        test_split="dev",
        train_query_cap=12,
        corpus_group="marco",
        eval_query_sets=tuple(
            EvalQuerySet(name=n, loader="ir_datasets", source=f"x/{n}") for n in SETS
        ),
        instruction="Find it",
    )


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> tuple[FakeSource, DatasetManifest, Path]:
    source = toy_source()
    source.splits["dev"] = source.splits.pop("test")
    eval_sets = {
        name: FakeQuerySet(
            [(f"{name}-{i}", f"q {i}") for i in range(3)],
            {f"{name}-{i}": {f"d{i}": 3} for i in range(3)},
        )
        for name in SETS
    }
    spec, root = _spec(), tmp_path_factory.mktemp("marco")
    with ArtifactWriter(
        root, "datasets/marco", start_manifest("prepare-data", {}, seed=0)
    ) as writer:
        manifest = prepare_environment(spec, toy_data(spec), writer, source, eval_sets)
        return source, manifest, writer.commit()


def test_the_corpus_is_streamed_once(prepared: tuple[FakeSource, DatasetManifest, Path]) -> None:
    assert prepared[0].corpus_reads == 1


def test_dev_is_the_test_split_and_training_is_capped(
    prepared: tuple[FakeSource, DatasetManifest, Path],
) -> None:
    manifest = prepared[1]
    assert manifest.num_queries["train"] + manifest.num_queries["validation"] == 12
    assert manifest.num_queries["test"] == 10


@pytest.mark.parametrize("name", SETS)
def test_each_evaluation_set_judges_corpus_documents(
    prepared: tuple[FakeSource, DatasetManifest, Path], name: str
) -> None:
    bundle = load_environment(prepared[2], include_test_qrels=True)
    assert prepared[1].num_queries[name] == 3
    assert {j.doc_id for j in bundle.qrels[name]} <= {f"d{i}" for i in range(12)}
