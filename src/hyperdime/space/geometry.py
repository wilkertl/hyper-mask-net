"""Streamed corpus geometry: per-dimension moments and the covariance spectrum (P1).

Computed from document embeddings only (protocol P0: no judgment of any split), in chunks so an
8.8M-passage corpus never needs to fit in memory; sums are accumulated in float64.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class CorpusGeometry:
    num_documents: int
    mean: np.ndarray  # (D,)
    variance: np.ndarray  # (D,), population variance
    eigenvalues: np.ndarray  # (D,), of the population covariance, descending

    @property
    def participation_ratio(self) -> float:
        """Effective dimensionality ``(Σλ)² / Σλ²`` of the covariance spectrum."""
        values = np.clip(self.eigenvalues, 0.0, None)
        return float(values.sum() ** 2 / (values**2).sum())

    @property
    def mean_energy(self) -> float:
        """Share of the mean squared norm ``E‖x‖²`` carried by the mean vector, ``‖μ‖² / E‖x‖²``."""
        mean_square = float((self.mean**2).sum())
        return mean_square / (mean_square + float(self.variance.sum()))


def corpus_geometry(
    documents: np.ndarray, chunk_size: int = 250_000, device: str = "cpu"
) -> CorpusGeometry:
    """Mean, variance, and covariance eigenvalues of ``documents`` (rows), streamed in chunks."""
    count, dim = documents.shape
    total = torch.zeros(dim, dtype=torch.float64)
    gram = torch.zeros(dim, dim, dtype=torch.float64)
    for start in range(0, count, chunk_size):
        chunk = torch.from_numpy(np.array(documents[start : start + chunk_size])).to(device)
        chunk = chunk.double() if device == "cpu" else chunk.float()
        total += chunk.sum(dim=0).double().cpu()
        gram += (chunk.T @ chunk).double().cpu()
    mean = total / count
    covariance = gram / count - torch.outer(mean, mean)
    eigenvalues = torch.linalg.eigvalsh(covariance).flip(0)
    return CorpusGeometry(
        count, mean.numpy(), torch.diagonal(covariance).numpy().copy(), eigenvalues.numpy()
    )
