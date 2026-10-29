"""The per-dimension space descriptor of an environment (F10 research D3-D4, D13).

Protocol P0 uses document embeddings only: group A (marginals with tails) and group E
(redundancy), 18 columns, one row per dimension. Rows follow the dimensions, so permuting the
dimensions permutes the rows, and no column depends on the order of the documents.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from hyperdime.contracts.hashing import file_sha256
from hyperdime.contracts.schemas import PayloadRef, SpaceDescriptor
from hyperdime.space.geometry import corpus_covariance
from hyperdime.space.marginal import MARGINAL_FEATURES
from hyperdime.space.redundancy import REDUNDANCY_FEATURES, redundancy_features
from hyperdime.space.streaming import (
    QUANTILE_ROWS,
    document_moments,
    document_quantiles,
    quantile_rows,
    tail_share,
)

TAIL_FEATURES = ("skew", "kurt", "tail3")
# (name, group) in column order; the data model fixes it.
FEATURES: tuple[tuple[str, str], ...] = (
    *((f"doc_{name}", "A") for name in MARGINAL_FEATURES + TAIL_FEATURES),
    *((f"red_{name}", "E") for name in REDUNDANCY_FEATURES),
)
PROTOCOLS = ("P0",)


def build_descriptors(
    documents: np.ndarray,
    doc_ids: Sequence[str],
    protocol: str = "P0",
    chunk_size: int = 250_000,
    device: str = "cpu",
    quantile_size: int = QUANTILE_ROWS,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """``(descriptor [D, h] float64, covariance [D, D], feature names, groups)``."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol {protocol!r} is not built yet; choose one of {PROTOCOLS}")
    if len(doc_ids) != len(documents):
        raise ValueError("doc_ids must name every document row")
    moments = document_moments(documents, chunk_size, device)
    tails = tail_share(documents, moments, chunk_size, device)
    quantiles = document_quantiles(documents, quantile_rows(doc_ids, quantile_size))
    covariance = corpus_covariance(documents, chunk_size, device)[1].numpy()
    redundancy = redundancy_features(covariance)
    columns = [moments.mean, moments.std, moments.mean_abs, moments.rms, *quantiles.T,
               moments.skew, moments.kurt, tails,
               *(redundancy[n] for n in REDUNDANCY_FEATURES)]  # fmt: skip
    names = [name for name, _ in FEATURES]
    groups = sorted({group for _, group in FEATURES})
    return np.stack(columns, axis=1), covariance, names, groups


def descriptor_record(
    environment: str,
    protocol: str,
    payload: Path,
    names: list[str],
    groups: list[str],
    num_documents: int,
    num_queries: int = 0,
) -> SpaceDescriptor:
    """The ``SpaceDescriptor`` of a saved ``.npy`` payload (validated: P0 has no queries)."""
    return SpaceDescriptor(
        environment=environment,
        protocol=protocol,
        feature_names=names,
        groups=groups,
        num_documents=num_documents,
        num_queries=num_queries,
        payload=PayloadRef(path=payload.name, sha256=file_sha256(payload), format="npy"),
    )
