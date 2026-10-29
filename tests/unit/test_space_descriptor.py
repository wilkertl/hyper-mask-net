"""The P0 space descriptor: columns, record, and invariances (F10 research D3-D4, D13)."""

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from hyperdime.space.descriptor import FEATURES, build_descriptors, descriptor_record

P0_COLUMNS = [
    "doc_mean", "doc_std", "doc_mean_abs", "doc_rms",
    "doc_q05", "doc_q25", "doc_q50", "doc_q75", "doc_q95",
    "doc_skew", "doc_kurt", "doc_tail3",
    "red_corr_mean_abs", "red_corr_max_abs", "red_corr_top8",
    "red_pc_share_8", "red_pc_share_32", "red_pc_share_128",
]  # fmt: skip
DOCS = np.random.default_rng(0).normal(size=(40, 6)).astype(np.float16)
IDS = [f"doc-{i}" for i in range(40)]


def _build(docs: np.ndarray = DOCS, ids: list[str] = IDS, **kwargs: int) -> np.ndarray:
    return build_descriptors(docs, ids, "P0", chunk_size=7, **kwargs)[0]  # type: ignore[arg-type]


def test_the_p0_columns_and_groups_follow_the_data_model() -> None:
    assert [name for name, _ in FEATURES] == P0_COLUMNS
    assert {g for n, g in FEATURES if n.startswith("doc_")} == {"A"}
    assert {g for n, g in FEATURES if n.startswith("red_")} == {"E"}
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


def test_only_p0_is_built_in_stage_one() -> None:
    with pytest.raises(ValueError, match="not built yet"):
        build_descriptors(DOCS, IDS, "P1")
