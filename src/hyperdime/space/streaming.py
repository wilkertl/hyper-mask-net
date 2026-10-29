"""Descriptor group A from a corpus too large to hold: streamed moments, tails, and quantiles (P0).

Document embeddings only (F10 research D3). Power sums are accumulated in float64 around a pilot
centre (the first chunk's mean), so central moments do not cancel catastrophically; the tail share
needs the mean and standard deviation and so takes a second pass. Quantiles use every document up
to ``QUANTILE_ROWS`` and otherwise a fixed sample chosen by a seeded hash of document ids.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from hyperdime.space.marginal import QUANTILES
from hyperdime.training.retrieval_selection import candidate_queries

QUANTILE_ROWS = 1_000_000
QUANTILE_SEED = 0
QUANTILE_BLOCK = 128
TAIL_SIGMAS = 3.0


@dataclass(frozen=True)
class Moments:
    """Per-dimension statistics of a corpus; every array has shape ``(D,)``."""

    count: int
    mean: np.ndarray
    std: np.ndarray  # population
    mean_abs: np.ndarray
    rms: np.ndarray
    skew: np.ndarray  # m3 / m2^{3/2}; 0 where std is 0
    kurt: np.ndarray  # excess, m4 / m2^2 - 3; 0 where std is 0


def _chunks(documents: np.ndarray, chunk_size: int, device: str) -> Iterator[torch.Tensor]:
    for start in range(0, len(documents), chunk_size):
        chunk = torch.from_numpy(np.array(documents[start : start + chunk_size]))
        yield chunk.to(device).double()


def document_moments(
    documents: np.ndarray, chunk_size: int = 250_000, device: str = "cpu"
) -> Moments:
    """One streamed pass over ``documents`` (rows) for the mean, std, mean_abs, RMS, skewness,
    and excess kurtosis of every dimension."""
    count, dim = documents.shape
    if count == 0:
        raise ValueError("documents must have at least one row")
    centre = torch.from_numpy(np.asarray(documents[:chunk_size], dtype=np.float64).mean(axis=0))
    sums = torch.zeros(4, dim, dtype=torch.float64)
    total_abs = torch.zeros(dim, dtype=torch.float64)
    total_square = torch.zeros(dim, dtype=torch.float64)
    shift = centre.to(device)
    for chunk in _chunks(documents, chunk_size, device):
        y = chunk - shift
        power = y.clone()
        for p in range(4):
            sums[p] += power.sum(dim=0).cpu()
            power = power * y
        total_abs += chunk.abs().sum(dim=0).cpu()
        total_square += chunk.square().sum(dim=0).cpu()
    s1, s2, s3, s4 = (sums[p].numpy() / count for p in range(4))
    # Central moments of x from raw moments of y = x - centre.
    m2 = np.clip(s2 - s1**2, 0.0, None)
    m3 = s3 - 3 * s1 * s2 + 2 * s1**3
    m4 = s4 - 4 * s1 * s3 + 6 * s1**2 * s2 - 3 * s1**4
    std = np.sqrt(m2)
    varying = m2 > 0
    skew = np.divide(m3, m2**1.5, out=np.zeros(dim), where=varying)
    kurt = np.divide(m4, m2**2, out=np.full(dim, 3.0), where=varying) - 3.0
    return Moments(
        count,
        centre.numpy() + s1,
        std,
        total_abs.numpy() / count,
        np.sqrt(total_square.numpy() / count),
        skew,
        kurt,
    )


def tail_share(
    documents: np.ndarray, moments: Moments, chunk_size: int = 250_000, device: str = "cpu"
) -> np.ndarray:
    """Per dimension, the fraction of documents with ``|x - mean| > 3 · std`` (0 where std is 0)."""
    mean = torch.from_numpy(moments.mean).to(device)
    bound = torch.from_numpy(TAIL_SIGMAS * moments.std).to(device)
    counts = torch.zeros(len(moments.mean), dtype=torch.float64)
    for chunk in _chunks(documents, chunk_size, device):
        counts += ((chunk - mean).abs() > bound).sum(dim=0).double().cpu()
    return np.where(moments.std > 0, counts.numpy() / moments.count, 0.0)


def quantile_rows(doc_ids: Sequence[str], size: int = QUANTILE_ROWS) -> np.ndarray:
    """Sorted rows of the documents the quantiles use: all of them up to ``size``, otherwise the
    ``size`` smallest under a seeded hash of their ids (the same whatever the row order)."""
    if len(doc_ids) <= size:
        return np.arange(len(doc_ids))
    return np.sort(np.asarray(candidate_queries(doc_ids, size, QUANTILE_SEED)))


def document_quantiles(documents: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """``[D, 5]`` quantiles q05-q95 of ``documents[rows]`` per dimension (linear interpolation, as
    ``marginal_stats``), computed in float64 column blocks to bound memory."""
    sample = np.asarray(documents[rows])
    found = np.empty((sample.shape[1], len(QUANTILES)))
    for start in range(0, sample.shape[1], QUANTILE_BLOCK):
        block = sample[:, start : start + QUANTILE_BLOCK].astype(np.float64)
        found[start : start + QUANTILE_BLOCK] = np.quantile(block, QUANTILES, axis=0).T
    return found
