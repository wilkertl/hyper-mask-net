"""Batched kernels of P3 on any torch device: one ordering per pair, every ``k`` from it.

A dimension is in the top ``k`` of an ordering exactly when its rank is below ``k``, so it is in two
top-``k`` masks exactly when ``k > max(rank_a, rank_b)``, and a histogram of those maxima gives the
exact intersection at every ``k`` in ``O(D)`` per pair. No mask is built, here or anywhere: the
largest object is ``[pairs in a batch, D]``, and the corpus stays on disk (ADR-0014).

Products of float16 values are exact in float64, so an ordering is the same on every device. Sums
differ only in their last bits between devices; ``scripts/checks/p3_device_parity.py`` measures it.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
from torch import Tensor

CHUNK_PAIRS = 16384
CHUNK_GROUPS = 4096


class Clock:
    """Wall time per phase. A GPU is synchronized at each boundary, so queued work is not charged
    to whatever phase happens to come next."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.seconds: dict[str, float] = {}

    def _sync(self) -> None:
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        self._sync()
        started = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - started


def device_memory(device: str) -> dict[str, Any]:
    """Peak memory of the device since the last reset; nothing for a CPU, which RSS covers."""
    if not device.startswith("cuda"):
        return {"name": "cpu"}
    return {
        "name": torch.cuda.get_device_name(device),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        "total_mib": torch.cuda.get_device_properties(device).total_memory / 2**20,
    }


def reset_device_memory(device: str) -> None:
    if device.startswith("cuda"):
        torch.empty(0, device=device)  # the reset refuses a device that has not been used yet
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def gather(matrix: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """``matrix[rows]`` in float64, reading a memory map in file order; rows may repeat."""
    unique, inverse = np.unique(rows, return_inverse=True)
    found: np.ndarray = np.asarray(matrix[unique], dtype=np.float64)[inverse]
    return found


def stable_order(values: Tensor) -> Tensor:
    """Dimensions by descending value per row; ties go to the lower index on every device.

    Adding zero turns ``-0.0`` into ``0.0``: they compare equal, but a device's sort may tell their
    bits apart, and a contribution of zero can carry either sign.
    """
    return torch.argsort(values.neg() + 0.0, dim=1, stable=True)


def ranks_of(order: Tensor) -> Tensor:
    """Inverse of each row's ordering: entry ``j`` is the position of dimension ``j``."""
    positions = torch.arange(order.shape[1], device=order.device, dtype=torch.int32)
    ranks = torch.empty_like(order, dtype=torch.int32)
    ranks.scatter_(1, order, positions.expand_as(order))
    return ranks


def pair_curves(values: Tensor, order: Tensor) -> dict[str, Tensor]:
    """Three different things at every ``k``, ``[rows, D]``, along each row's own ordering.

    ``similarity`` is the raw unnormalized score of the top ``k``. ``retained_positive_share`` is
    the share of positive mass ``P`` kept, and ``removed_negative_share`` the share of negative
    mass ``N`` left out; a zero mass makes its share NaN.
    """
    ordered = values.gather(1, order)
    positive, negative = ordered.clamp(min=0), (-ordered).clamp(min=0)
    total_positive = positive.sum(dim=1, keepdim=True)
    total_negative = negative.sum(dim=1, keepdim=True)
    return {
        "similarity": ordered.cumsum(dim=1),
        "retained_positive_share": positive.cumsum(dim=1) / total_positive,
        "removed_negative_share": (total_negative - negative.cumsum(dim=1)) / total_negative,
    }


def intersection_curves(left: Tensor, right: Tensor) -> Tensor:
    """``[rows, D]``: how many dimensions two top-``k`` masks share, entry ``k - 1``."""
    rows, dim = left.shape
    offsets = torch.arange(rows, device=left.device, dtype=torch.int64)[:, None] * dim
    latest = torch.maximum(left, right).to(torch.int64) + offsets
    counts = torch.bincount(latest.ravel(), minlength=rows * dim)
    return counts.view(rows, dim).cumsum(dim=1)


def jaccard_curves(left: Tensor, right: Tensor) -> Tensor:
    """``[rows, D]`` exact Jaccard of the two top-``k`` masks at every ``k``."""
    shared = intersection_curves(left, right).to(torch.float64)
    k = torch.arange(1, left.shape[1] + 1, device=left.device, dtype=torch.float64)
    return shared / (2 * k - shared)


def group_jaccard(
    ranks: Tensor, first: Tensor, second: Tensor, group: Tensor, groups: int
) -> Tensor:
    """``[groups, D]`` mean Jaccard over each group's pairs of rows, in bounded memory."""
    total = torch.zeros((groups, ranks.shape[1]), device=ranks.device, dtype=torch.float64)
    for start in range(0, first.numel(), CHUNK_PAIRS):
        stop = start + CHUNK_PAIRS
        found = jaccard_curves(ranks[first[start:stop]], ranks[second[start:stop]])
        total.index_add_(0, group[start:stop], found)
    pairs = torch.zeros(groups, device=ranks.device, dtype=torch.float64)
    pairs.index_add_(0, group, torch.ones_like(group, dtype=torch.float64))
    return total / pairs[:, None]


def finite_totals(values: Tensor, slot: Tensor, slots: int) -> tuple[Tensor, Tensor]:
    """Sum and count of the finite entries of ``values`` (first axis: rows) per slot."""
    seen = values.isfinite()
    shape = (slots, *values.shape[1:])
    total = torch.zeros(shape, device=values.device, dtype=torch.float64)
    counts = torch.zeros(shape, device=values.device, dtype=torch.float64)
    total.index_add_(0, slot, torch.where(seen, values, torch.zeros_like(values)))
    counts.index_add_(0, slot, seen.to(torch.float64))
    return total, counts


def finite_mean(values: Tensor, slot: Tensor, slots: int) -> Tensor:
    """Mean of the finite entries per slot; NaN where a slot has none."""
    total, counts = finite_totals(values, slot, slots)
    return total / counts


def group_measures(
    values: Tensor, member: Tensor, group: Tensor, groups: int
) -> tuple[Tensor, Tensor]:
    """``(sign conflict share, cancelled share)`` of each group, with equal weights.

    ``member`` holds rows of ``values`` and ``group`` the group of each. A sign conflict needs
    ``d_j`` to change sign wherever ``q_j`` is nonzero, so it comes from the documents, not from
    the shared query. Conflict coverage can only grow when a document joins a group. The cancelled
    share need not: a new document can restore the mean's mass, or lose it.
    """
    device, dim = values.device, values.shape[1]
    conflicts, cancels = [], []
    for start in range(0, groups, CHUNK_GROUPS):
        stop = min(start + CHUNK_GROUPS, groups)
        chosen = ((group >= start) & (group < stop)).nonzero().ravel()
        rows, slot = values[member[chosen]], group[chosen] - start
        helps = torch.zeros((stop - start, dim), device=device, dtype=torch.int32)
        hurts = torch.zeros((stop - start, dim), device=device, dtype=torch.int32)
        helps.index_add_(0, slot, (rows > 0).to(torch.int32))
        hurts.index_add_(0, slot, (rows < 0).to(torch.int32))
        totals = torch.zeros((stop - start, dim), device=device, dtype=torch.float64)
        totals.index_add_(0, slot, rows)
        masses = torch.zeros(stop - start, device=device, dtype=torch.float64)
        masses.index_add_(0, slot, rows.abs().sum(dim=1))
        conflicts.append(((helps > 0) & (hurts > 0)).to(torch.float64).mean(dim=1))
        # Both masses share the group's size: mean L1 over mean mass needs no count.
        cancels.append(1.0 - totals.abs().sum(dim=1) / masses)
    return torch.cat(conflicts), torch.cat(cancels)


class Profiles:
    """Query-balanced selection frequency at every ``k``, kept as a histogram of ranks.

    ``histogram[r, j]`` adds a document's weight where dimension ``j`` has rank ``r``; a cumulative
    sum over ``r`` is then the frequency at every ``k``, so no mask is ever built. ``stratum_sizes``
    holds how many queries each stratum has.
    """

    def __init__(self, dim: int, assignments: np.ndarray, stratum_sizes: np.ndarray, device: str):
        self.dim = dim
        self.device = device
        self.dims = torch.arange(dim, device=device)
        self.assignments = torch.from_numpy(assignments.astype(np.int64)).to(device)
        cells = dim * dim
        draws = assignments.shape[0]
        self.total = torch.zeros(cells, device=device, dtype=torch.float64)
        self.strata = torch.zeros(stratum_sizes.size * cells, device=device, dtype=torch.float64)
        self.halves = torch.zeros(draws * 2 * cells, device=device, dtype=torch.float64)
        self.stratum_queries = stratum_sizes.astype(np.float64)

    def add(self, query: Tensor, stratum: Tensor, ranks: Tensor, weight: Tensor) -> None:
        """One row of ``ranks`` per document, with its weight; a query's weights sum to one over
        all it adds, whatever its count of documents or of repeated samples."""
        cells = self.dim * self.dim
        where = (ranks.to(torch.int64) * self.dim + self.dims).ravel()
        each = weight.repeat_interleave(self.dim)
        self.total.index_add_(0, where, each)
        self.strata.index_add_(0, stratum.repeat_interleave(self.dim) * cells + where, each)
        for draw in range(self.assignments.shape[0]):
            half = self.assignments[draw, query].repeat_interleave(self.dim)
            self.halves.index_add_(0, (draw * 2 + half) * cells + where, each)

    def frequency(self, queries: int) -> np.ndarray:
        found = self.total.view(self.dim, self.dim).cumsum(dim=0) / queries
        result: np.ndarray = found.cpu().numpy()
        return result

    def half_frequency(self) -> np.ndarray:
        labels = self.assignments.cpu().numpy()
        counts = np.stack([(labels == half).sum(axis=1) for half in (0, 1)], axis=1)
        shape = (labels.shape[0], 2, self.dim, self.dim)
        found = self.halves.view(shape).cumsum(dim=2).cpu().numpy()
        result: np.ndarray = found / counts[:, :, None, None]
        return result

    def stratum_frequency(self, stratum: int) -> np.ndarray:
        shape = (self.stratum_queries.size, self.dim, self.dim)
        found = self.strata.view(shape)[stratum].cumsum(dim=0).cpu().numpy()
        result: np.ndarray = found / self.stratum_queries[stratum]
        return result


def write_pair_orders(
    query_rows: np.ndarray,
    documents: np.ndarray,
    pair_rows: np.ndarray,
    pair_query: np.ndarray,
    contribution_file: np.ndarray,
    order_file: np.ndarray,
    *,
    device: str,
    batch_pairs: int,
    clock: Clock,
) -> None:
    """Compute every relevant pair's contributions and its one ordering, in batches, on disk.

    ``query_rows`` is ``[queries, D]``; a pair is row ``pair_rows[i]`` with query ``pair_query[i]``.
    """
    for start in range(0, pair_rows.size, batch_pairs):
        stop = start + batch_pairs
        with clock.span("read"):
            docs = gather(documents, pair_rows[start:stop])
            queries = np.asarray(query_rows[pair_query[start:stop]], dtype=np.float64)
        with clock.span("transfer"):
            docs_on, queries_on = (torch.from_numpy(a).to(device) for a in (docs, queries))
        with clock.span("compute"):
            values = queries_on * docs_on
            order = stable_order(values)
        with clock.span("transfer"):
            values_out = values.to(torch.float32).cpu().numpy()
            order_out = order.to(torch.int32).cpu().numpy()
        with clock.span("write"):
            contribution_file[start:stop] = values_out
            order_file[start:stop] = order_out
