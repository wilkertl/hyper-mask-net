"""Deterministic ID splits."""

from __future__ import annotations

import random
from collections.abc import Sequence


def split_ids(
    ids: Sequence[str], validation_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Split distinct IDs into sorted ``(train, validation)`` lists, reproducibly for a seed."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1).")
    if len(set(ids)) != len(ids):
        raise ValueError("ids must be distinct.")
    if len(ids) < 2:
        raise ValueError("at least two ids are required to split.")
    ordered = sorted(ids)
    random.Random(seed).shuffle(ordered)
    validation_count = min(max(1, round(len(ordered) * validation_fraction)), len(ordered) - 1)
    return sorted(ordered[validation_count:]), sorted(ordered[:validation_count])
