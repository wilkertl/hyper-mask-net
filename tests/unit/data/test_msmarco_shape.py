"""MS MARCO's shape: a `dev` test split, three evaluation query sets, a training cap (F2 US3)."""

from pathlib import Path

from hyperdime.contracts.manifests import ArtifactWriter, start_manifest
from hyperdime.data.environments import EnvironmentSpec, EvalQuerySet
from hyperdime.data.prepare import load_environment, prepare_environment

from conftest import FakeQuerySet, toy_data, toy_source

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


def test_marco_shaped_preparation(tmp_path: Path) -> None:
    source = toy_source()
    source.splits["dev"] = source.splits.pop("test")
    eval_sets = {
        name: FakeQuerySet(
            [(f"{name}-{i}", f"q {i}") for i in range(3)],
            {f"{name}-{i}": {f"d{i}": 3} for i in range(3)},
        )
        for name in SETS
    }
    spec = _spec()
    with ArtifactWriter(
        tmp_path, "datasets/marco", start_manifest("prepare-data", {}, seed=0)
    ) as writer:
        manifest = prepare_environment(spec, toy_data(spec), writer, source, eval_sets)
        directory = writer.commit()
    assert source.corpus_reads == 1  # the corpus is streamed once, never materialized twice
    assert manifest.num_queries["train"] + manifest.num_queries["validation"] == 12
    assert manifest.num_queries["test"] == 10
    for name in SETS:
        assert manifest.num_queries[name] == 3
        assert (directory / f"qrels-{name}.jsonl").exists()
    bundle = load_environment(directory, include_test_qrels=True)
    corpus = {f"d{i}" for i in range(12)}
    for name in SETS:
        assert {j.doc_id for j in bundle.qrels[name]} <= corpus
