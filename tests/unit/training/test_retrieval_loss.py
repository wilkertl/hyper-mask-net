"""Retrieval loss on masked query scores (F5 research D2)."""

import math

import pytest
import torch

from hyperdime.training.losses import masked_retrieval_loss

DOCS = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
QUERY = torch.tensor([[1.0, 0.0]])
FULL = torch.ones(1, 2)


def _loss(mask: torch.Tensor, negatives: list[int], temperature: float = 1.0) -> float:
    return masked_retrieval_loss(
        QUERY,
        mask,
        DOCS,
        positives=torch.tensor([[0]]),
        weights=torch.tensor([[1.0]]),
        negatives=torch.tensor([negatives]),
        score_temperature=temperature,
    ).item()


def test_one_positive_against_one_negative_by_hand() -> None:
    assert _loss(FULL, [1]) == pytest.approx(-math.log(math.e / (math.e + 1)))


def test_a_repeated_negative_counts_twice() -> None:
    assert _loss(FULL, [1, 1]) == pytest.approx(-math.log(math.e / (math.e + 2)))


def test_the_masked_query_is_not_renormalized() -> None:
    assert _loss(2 * FULL, [1]) == pytest.approx(_loss(FULL, [1], temperature=0.5))


def test_positives_are_weighted_by_their_normalized_gain() -> None:
    loss = masked_retrieval_loss(
        QUERY,
        FULL,
        DOCS,
        positives=torch.tensor([[0, 2]]),
        weights=torch.tensor([[0.25, 0.75]]),
        negatives=torch.tensor([[1]]),
        score_temperature=1.0,
    )
    log_z = math.log(math.e + math.e + 1)  # logits: positive 0 -> 1, positive 2 -> 1, negative -> 0
    assert loss.item() == pytest.approx(0.25 * (log_z - 1) + 0.75 * (log_z - 1))


def test_other_queries_positives_are_in_batch_negatives() -> None:
    loss = masked_retrieval_loss(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.ones(2, 2),
        DOCS,
        positives=torch.tensor([[0, -1], [1, -1]]),
        weights=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        negatives=torch.tensor([[2], [2]]),
        score_temperature=1.0,
    )
    # Each query: own positive 1, the other's positive 0, negative [1, 1] scores 1.
    assert loss.item() == pytest.approx(-math.log(math.e / (math.e + 1 + math.e)))


def test_a_shared_positive_is_not_an_in_batch_negative() -> None:
    loss = masked_retrieval_loss(
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        torch.ones(2, 2),
        DOCS,
        positives=torch.tensor([[0], [0]]),
        weights=torch.tensor([[1.0], [1.0]]),
        negatives=torch.tensor([[1], [1]]),
        score_temperature=1.0,
    )
    assert loss.item() == pytest.approx(-math.log(math.e / (math.e + 1)))


def test_a_document_positive_for_two_other_queries_is_one_in_batch_negative() -> None:
    loss = masked_retrieval_loss(
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
        torch.ones(3, 2),
        DOCS,
        positives=torch.tensor([[0], [1], [1]]),
        weights=torch.ones(3, 1),
        negatives=torch.tensor([[2], [2], [2]]),
        score_temperature=1.0,
    )
    # Query 0: own positive 1, document 1 (positive of queries 1 and 2) once at 0, negative 1.
    # Queries 1 and 2: own positive 1, document 0 at 0, negative 1.
    assert loss.item() == pytest.approx(-math.log(math.e / (math.e + 1 + math.e)))
