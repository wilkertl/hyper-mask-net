"""Descriptor group E: how much each dimension duplicates the others (P0, F10 research D4).

From the corpus covariance only: correlations with the other dimensions, and the share of each
dimension's variance inside the top principal subspaces. A zero-variance dimension gets
correlations 0 and principal-subspace share 0.
"""

from __future__ import annotations

import numpy as np

TOP_CORRELATIONS = 8
PC_RANKS = (8, 32, 128)
REDUNDANCY_FEATURES = (
    "corr_mean_abs",
    "corr_max_abs",
    "corr_top8",
    *(f"pc_share_{r}" for r in PC_RANKS),
)


def redundancy_features(covariance: np.ndarray) -> dict[str, np.ndarray]:
    """``{name: (D,)}`` in ``REDUNDANCY_FEATURES`` order from a ``[D, D]`` covariance."""
    covariance = np.asarray(covariance, dtype=np.float64)
    dim = covariance.shape[0]
    if covariance.shape != (dim, dim) or dim < 2:
        raise ValueError("covariance must be square with at least two dimensions")
    variance = np.clip(np.diagonal(covariance), 0.0, None)
    std = np.sqrt(variance)
    scale = np.outer(std, std)
    correlation = np.divide(covariance, scale, out=np.zeros_like(covariance), where=scale > 0)
    off = ~np.eye(dim, dtype=bool)
    others = np.abs(correlation[off]).reshape(dim, dim - 1)
    top = min(TOP_CORRELATIONS, dim - 1)
    squared = np.sort(others**2, axis=1)[:, ::-1][:, :top]
    eigenvalues, vectors = np.linalg.eigh(covariance)
    eigenvalues, vectors = np.clip(eigenvalues[::-1], 0.0, None), vectors[:, ::-1]
    explained = np.cumsum(vectors**2 * eigenvalues, axis=1)  # [j, r-1]: top-r variance of j
    found = {
        "corr_mean_abs": others.mean(axis=1),
        "corr_max_abs": others.max(axis=1),
        "corr_top8": squared.mean(axis=1),
    }
    for r in PC_RANKS:
        within = explained[:, min(r, dim) - 1]
        found[f"pc_share_{r}"] = np.divide(within, variance, out=np.zeros(dim), where=variance > 0)
    return found
