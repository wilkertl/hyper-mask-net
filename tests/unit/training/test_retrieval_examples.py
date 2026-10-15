"""Training examples for the retrieval objective (F5 research D1)."""

from typing import cast

import numpy as np
import pytest
import torch

from hyperdime.training.examples import RetrievalExamples, retrieval_examples
from hyperdime.training.inputs import EnvironmentInputs, load_examples

DOCS = np.arange(12 * 4, dtype=np.float16).reshape(12, 4)
QUERIES = np.eye(3, 4, dtype=np.float32)
DOC_ROWS = {f"d{i}": i for i in range(10)}  # d10 and d11 are not in the corpus ids
POSITIVES = {"a": {"d1": 2, "d2": 1, "d3": 0}, "b": {"d4": 1}, "c": {"d99": 1}}
POOLS = {"a": ["d7", "d7", "d8"], "b": ["d7", "d9", "d1"], "c": ["d5", "d6", "d7"]}


@pytest.fixture(scope="module")
def examples() -> RetrievalExamples:
    return retrieval_examples(
        ["a", "b", "c"], QUERIES, {"a": 0, "b": 1, "c": 2}, DOCS, DOC_ROWS, POSITIVES, POOLS
    )


def test_queries_without_a_positive_in_the_corpus_are_skipped_and_counted(
    examples: RetrievalExamples,
) -> None:
    assert examples.skipped == 1
    assert torch.equal(examples.queries, torch.from_numpy(QUERIES[:2]))


def test_each_document_row_is_gathered_once(examples: RetrievalExamples) -> None:
    assert examples.documents.shape == (6, 4)  # d1, d2, d4, d7, d8, d9
    assert torch.equal(examples.documents[examples.negatives[0, 0]], torch.from_numpy(DOCS[7]))


def test_positives_keep_grades_above_zero_with_normalized_gains(
    examples: RetrievalExamples,
) -> None:
    # Gathered rows are d1, d2, d4, d7, d8, d9; gains 2**2 - 1 = 3 and 2**1 - 1 = 1.
    assert examples.positives.tolist() == [[0, 1], [2, -1]]
    assert examples.weights.flatten().tolist() == pytest.approx([0.75, 0.25, 1.0, 0.0])


def test_negative_pools_keep_their_repeats(examples: RetrievalExamples) -> None:
    first = examples.negatives[0].tolist()
    assert len(first) == 3 and first[0] == first[1] != first[2]


def test_examples_are_never_built_from_test_judgments() -> None:
    with pytest.raises(ValueError, match="not 'test'"):
        load_examples(cast(EnvironmentInputs, None), "test")
