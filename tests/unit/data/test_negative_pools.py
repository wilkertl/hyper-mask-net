"""ADR-0002 hard-negative pools for training and validation queries (F2 US4)."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from hyperdime.contracts.manifests import ArtifactWriter, start_manifest
from hyperdime.data.negatives import (
    build_negative_pools,
    sample_from_rankings,
    sample_hard_negatives,
)
from hyperdime.data.prepare import (
    DatasetBundle,
    iter_documents,
    load_environment,
    prepare_environment,
)

from conftest import StubEncoder, toy_data, toy_eval_sets, toy_source, toy_spec


def _bundle(root: Path) -> tuple[DatasetBundle, Path]:
    spec = toy_spec()
    with ArtifactWriter(root, "datasets/toy", start_manifest("prepare-data", {}, seed=0)) as writer:
        prepare_environment(spec, toy_data(spec), writer, toy_source(), toy_eval_sets())
        directory = writer.commit()
    return load_environment(directory), directory


def _pools(root: Path, *, count: int = 4, seed: int = 0):  # type: ignore[no-untyped-def]
    bundle, directory = _bundle(root)
    doc_ids = [d.doc_id for d in iter_documents(directory)]
    encoder = StubEncoder()
    documents = encoder.encode(doc_ids).numpy().astype(np.float16)
    query_ids = sorted(bundle.queries)
    queries = encoder.encode(query_ids).numpy()
    pools = build_negative_pools(
        bundle,
        queries,
        query_ids,
        documents,
        doc_ids,
        depth=8,
        count=count,
        seed=seed,
        chunk_size=5,
    )
    return bundle, pools


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    return _pools(tmp_path_factory.mktemp("pools"))


def test_pools_cover_train_and_validation_only(built) -> None:  # type: ignore[no-untyped-def]
    bundle, pools = built
    assert {p.split for p in pools} == {"train", "validation"}
    assert {p.query_id for p in pools} == set(
        bundle.query_ids("train") + bundle.query_ids("validation")
    )


def test_pools_exclude_positives_and_the_query_id(built) -> None:  # type: ignore[no-untyped-def]
    bundle, pools = built
    positives = {**bundle.positives("train"), **bundle.positives("validation")}
    for pool in pools:
        assert len(pool.doc_ids) == 4 and pool.depth == 8 and pool.pool_size == 4
        assert not set(pool.doc_ids) & positives[pool.query_id]


def test_same_seed_same_pools(tmp_path: Path) -> None:
    _, first = _pools(tmp_path / "a")
    _, second = _pools(tmp_path / "b")
    _, other = _pools(tmp_path / "c", seed=1)
    assert first == second and first != other


def test_sampling_matches_the_spike(tmp_path: Path) -> None:
    generator = torch.Generator().manual_seed(3)
    documents = F.normalize(torch.randn(40, 16, generator=generator), dim=-1)
    queries = F.normalize(torch.randn(6, 16, generator=generator), dim=-1)
    doc_ids = [f"d{i}" for i in range(40)]
    excluded = [{f"d{i}"} for i in range(6)]
    spike = sample_hard_negatives(
        queries, documents, doc_ids, excluded, 5, 10, torch.Generator().manual_seed(0)
    )
    from hyperdime.retrieval.exact import exact_search

    _, rankings = exact_search(queries, documents, 10)
    shared = sample_from_rankings(rankings, doc_ids, excluded, 5, torch.Generator().manual_seed(0))
    assert shared == spike
