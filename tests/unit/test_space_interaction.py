"""Groups C and D and the calibration draws, by hand on D = 3, 2 queries, 6 documents
(F10 research D5)."""

import numpy as np
import pytest

from hyperdime.space.interaction import (
    INTERACTION_FEATURES,
    background_rows,
    calibration_queries,
    interaction_features,
    top_rows,
)

DOCS = np.array(
    [[1, 0, 2], [0, 1, 0], [2, 2, 1], [0, 0, 1], [1, 1, 1], [3, 0, 0]], dtype=np.float16
)  # fmt: skip
QUERIES = np.array([[1, 2, 0], [2, 0, 1]], dtype=np.float16)
TOP = np.array([[2, 0, 4], [5, 0, 2]])  # three retrieved documents: fewer than every K
BACKGROUND = np.array([[1, 3], [3, 4]])


def _column(found: np.ndarray, name: str) -> np.ndarray:
    return found[:, INTERACTION_FEATURES.index(name)]


def test_pair_statistics_of_dimension_zero_by_hand() -> None:
    found = interaction_features(QUERIES, DOCS, TOP, BACKGROUND)
    assert found.shape == (3, 13)
    # Background products q_0 · d_0: query 0 with d1, d3 -> 0, 0; query 1 with d3, d4 -> 0, 2.
    rand = [found[0, INTERACTION_FEATURES.index(f"pair_rand_{s}")] for s in
            ("mean", "std", "q05", "q50", "q95")]  # fmt: skip
    assert rand == pytest.approx([0.5, 0.75**0.5, 0.0, 0.0, 1.7])
    # Top products: query 0 with d2, d0, d4 -> 2, 1, 1; query 1 with d5, d0, d2 -> 6, 2, 4.
    top = [found[0, INTERACTION_FEATURES.index(f"pair_top_{s}")] for s in
           ("mean", "std", "q05", "q50", "q95")]  # fmt: skip
    assert top == pytest.approx([8 / 3, 29**0.5 / 3, 1.0, 2.0, 5.5])


def test_contrast_uses_the_retrieved_documents_there_are() -> None:
    found = interaction_features(QUERIES, DOCS, TOP, BACKGROUND)
    # Query 0: 1 · (4/3 - 0); query 1: 2 · (2 - 1/2). Every K exceeds the three retrieved.
    for k in (5, 10, 20):
        assert _column(found, f"contrast_k{k}")[0] == pytest.approx(13 / 6)
    # Dimension 2: query 0's q_2 is 0; query 1: 1 · ((0 + 2 + 1)/3 - (1 + 1)/2) = 0.
    assert _column(found, "contrast_k5")[2] == pytest.approx(0.0)


def test_contrast_k5_takes_the_first_five_of_six() -> None:
    top = np.array([[2, 5, 0, 4, 1, 3], [5, 2, 0, 4, 3, 1]])
    found = interaction_features(QUERIES, DOCS, top, BACKGROUND)
    # K = 5: query 0, 1 · 7/5; query 1, 2 · (7/5 - 1/2). All six: 1 · 7/6 and 2 · (7/6 - 1/2).
    assert _column(found, "contrast_k5")[0] == pytest.approx(1.6)
    assert _column(found, "contrast_k10")[0] == pytest.approx(1.25)
    assert _column(found, "contrast_k20")[0] == pytest.approx(1.25)


def test_query_row_order_does_not_change_it() -> None:
    order = [1, 0]
    found = interaction_features(QUERIES[order], DOCS, TOP[order], BACKGROUND[order])
    assert np.allclose(found, interaction_features(QUERIES, DOCS, TOP, BACKGROUND), atol=1e-12)


def test_top_rows_rank_documents_by_exact_inner_product() -> None:
    scores = QUERIES.astype(np.float32) @ DOCS.astype(np.float32).T
    rows = top_rows(QUERIES, DOCS, chunk_size=4)
    assert rows.shape == (2, 6)  # depth 20 capped at the six documents
    assert np.array_equal(np.take_along_axis(scores, rows, 1), -np.sort(-scores, axis=1))


def test_calibration_is_a_seeded_hash_sample_of_training_queries_only() -> None:
    splits = {"train": [f"t{i}" for i in range(8)], "validation": ["v0", "v1"], "test": ["x0"]}
    found = calibration_queries(splits, size=3)
    assert len(found) == 3 and found == sorted(found) and set(found) <= set(splits["train"])
    assert calibration_queries({**splits, "train": splits["train"][::-1]}, size=3) == found
    assert calibration_queries(splits) == sorted(splits["train"])  # all, under the cap
    with pytest.raises(ValueError, match="train"):
        calibration_queries({"validation": ["v0"], "test": ["x0"]})


DOC_IDS = [f"d{i}" for i in range(10)]


def test_background_draws_are_seeded_and_without_replacement() -> None:
    found = background_rows(["q0", "q1"], DOC_IDS, size=4)
    assert found.shape == (2, 4) and all(len(set(row)) == 4 for row in found.tolist())
    assert np.array_equal(found, background_rows(["q0", "q1"], DOC_IDS, size=4))
    assert np.array_equal(background_rows(["q1", "q0"], DOC_IDS, size=4), found[::-1])


def test_background_draws_follow_document_ids_not_rows() -> None:
    reversed_ids = DOC_IDS[::-1]
    found = background_rows(["q0", "q1"], DOC_IDS, size=4)
    shuffled = background_rows(["q0", "q1"], reversed_ids, size=4)
    names = [{DOC_IDS[r] for r in row} for row in found]
    assert [{reversed_ids[r] for r in row} for row in shuffled] == names


def test_with_fewer_documents_than_the_size_the_background_is_every_document() -> None:
    assert sorted(background_rows(["q0"], DOC_IDS[:3])[0].tolist()) == [0, 1, 2]
