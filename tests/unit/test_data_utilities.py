import pytest
import torch

from hyperdime.data.negatives import mine_hard_negatives
from hyperdime.data.splits import split_ids


def test_split_ids_is_deterministic_disjoint_and_complete() -> None:
    ids = [f"q{i}" for i in range(20)]
    train, validation = split_ids(ids, validation_fraction=0.15, seed=13)
    assert (train, validation) == split_ids(list(reversed(ids)), 0.15, seed=13)
    assert len(validation) == 3
    assert not set(train) & set(validation)
    assert sorted(train + validation) == sorted(ids)


@pytest.mark.parametrize(
    ("ids", "fraction"), [(["a", "b"], 0.0), (["a", "a", "b"], 0.5), (["a"], 0.5)]
)
def test_split_ids_rejects_invalid_inputs(ids: list[str], fraction: float) -> None:
    with pytest.raises(ValueError):
        split_ids(ids, fraction, seed=0)


def test_mine_hard_negatives_skips_judged_positives() -> None:
    queries = torch.tensor([[1.0, 0.0]])
    documents = torch.tensor([[0.9, 0.0], [0.8, 0.0], [0.5, 0.0], [0.0, 1.0]])
    ids = ["pos", "hard1", "hard2", "easy"]
    assert mine_hard_negatives(queries, documents, ids, [{"pos"}], pool_size=2, depth=3) == [
        ["hard1", "hard2"]
    ]
    with pytest.raises(ValueError):
        mine_hard_negatives(queries, documents, ids, [{"pos", "hard1"}], pool_size=2, depth=3)
