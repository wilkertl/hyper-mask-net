"""The P0 and P1 space descriptors: columns, record, and invariances (F10 research D3-D5, D13)."""

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from hyperdime.space.descriptor import (
    FEATURES,
    build_descriptors,
    descriptor_record,
    marginal_columns,
)
from hyperdime.space.interaction import INTERACTION_FEATURES

P0_COLUMNS = [
    "doc_mean", "doc_std", "doc_mean_abs", "doc_rms",
    "doc_q05", "doc_q25", "doc_q50", "doc_q75", "doc_q95",
    "doc_skew", "doc_kurt", "doc_tail3",
    "red_corr_mean_abs", "red_corr_max_abs", "red_corr_top8",
    "red_pc_share_8", "red_pc_share_32", "red_pc_share_128",
]  # fmt: skip
DOCS = np.random.default_rng(0).normal(size=(40, 6)).astype(np.float16)
IDS = [f"doc-{i}" for i in range(40)]
QUERIES = np.random.default_rng(2).normal(size=(5, 6)).astype(np.float16)
QUERY_IDS = [f"q-{i}" for i in range(5)]


def _build(docs: np.ndarray = DOCS, ids: list[str] = IDS, **kwargs: int) -> np.ndarray:
    return build_descriptors(docs, ids, "P0", chunk_size=7, **kwargs)[0]  # type: ignore[arg-type]


def test_the_p0_columns_and_groups_follow_the_data_model() -> None:
    assert [name for name, _ in FEATURES["P0"]] == P0_COLUMNS
    assert {g for n, g in FEATURES["P0"] if n.startswith("doc_")} == {"A"}
    assert {g for n, g in FEATURES["P0"] if n.startswith("red_")} == {"E"}
    values, covariance, names, groups = build_descriptors(DOCS, IDS, "P0", chunk_size=7)
    assert values.shape == (6, 18) and values.dtype == np.float64 and np.isfinite(values).all()
    assert covariance.shape == (6, 6) and names == P0_COLUMNS and groups == ["A", "E"]
    assert values[:, 1] ** 2 == pytest.approx(np.diagonal(covariance))  # doc_std, the covariance


def test_the_record_is_p0_without_queries(tmp_path: Path) -> None:
    payload = tmp_path / "descriptor-P0.npy"
    np.save(payload, _build())
    record = descriptor_record("toy", "P0", payload, P0_COLUMNS, ["A", "E"], 40)
    assert (record.protocol, record.groups, record.num_queries) == ("P0", ["A", "E"], 0)
    with pytest.raises(ValidationError, match="num_queries"):
        descriptor_record("toy", "P0", payload, P0_COLUMNS, ["A", "E"], 40, num_queries=5)


def test_document_row_order_does_not_change_it() -> None:
    order = np.random.default_rng(1).permutation(40)
    shuffled = (DOCS[order], [IDS[i] for i in order])
    assert np.allclose(_build(*shuffled), _build(), atol=1e-12)
    # Also when the quantiles use a hashed sample of the documents.
    assert np.allclose(_build(*shuffled, quantile_size=10), _build(quantile_size=10), atol=1e-12)


def test_permuting_dimensions_permutes_the_rows() -> None:
    order = [3, 0, 5, 1, 4, 2]
    assert np.allclose(_build(DOCS[:, order]), _build()[order], atol=1e-12)


def _p1(queries: np.ndarray = QUERIES, query_ids: list[str] = QUERY_IDS) -> np.ndarray:
    return build_descriptors(DOCS, IDS, "P1", chunk_size=7, queries=queries,
                             query_ids=query_ids)[0]  # fmt: skip


def test_the_p1_columns_extend_p0_with_groups_b_c_d() -> None:
    names = [name for name, _ in FEATURES["P1"]]
    assert names[:18] == P0_COLUMNS and len(names) == 43
    assert names[18:30] == [c.replace("doc_", "qry_") for c in P0_COLUMNS[:12]]
    assert names[30:] == list(INTERACTION_FEATURES)


def test_the_p1_descriptor_holds_the_p0_descriptor() -> None:
    names = [name for name, _ in FEATURES["P1"]]
    values, _, found, groups = build_descriptors(DOCS, IDS, "P1", chunk_size=7, queries=QUERIES,
                                                 query_ids=QUERY_IDS)  # fmt: skip
    assert values.shape == (6, 43) and np.isfinite(values).all() and found == names
    assert groups == ["A", "B", "C", "D", "E"]
    assert np.allclose(values[:, :18], _build(), atol=1e-12)  # the P0 columns are P0's


def test_group_b_is_group_a_on_the_queries() -> None:
    expected = marginal_columns(QUERIES, QUERY_IDS, chunk_size=7)
    assert np.allclose(_p1()[:, 18:30], expected, atol=1e-12)
    assert np.allclose(expected, _build(QUERIES, QUERY_IDS)[:, :12], atol=1e-12)


def test_query_row_order_does_not_change_p1() -> None:
    order = [3, 0, 4, 1, 2]
    assert np.allclose(_p1(QUERIES[order], [QUERY_IDS[i] for i in order]), _p1(), atol=1e-12)


def test_p1_needs_named_calibration_queries() -> None:
    with pytest.raises(ValueError, match="calibration"):
        build_descriptors(DOCS, IDS, "P1")
    with pytest.raises(ValueError, match="calibration"):
        build_descriptors(DOCS, IDS, "P1", queries=QUERIES, query_ids=QUERY_IDS[:2])
    with pytest.raises(ValueError, match="not built"):
        build_descriptors(DOCS, IDS, "P2")
