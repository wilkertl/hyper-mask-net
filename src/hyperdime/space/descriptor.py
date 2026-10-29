"""The per-dimension space descriptor of an environment (F10 research D3-D4, D13).

Protocol P0 uses document embeddings only: group A (marginals with tails) and group E
(redundancy), 18 columns, one row per dimension. Protocol P1 adds unlabeled calibration queries:
group B (A's 12 columns on the queries), C (query-document products), and D (top-K contrast), 43
columns whose first 18 are P0's. Rows follow the dimensions, so permuting the dimensions permutes
the rows, and no column depends on the order of the documents or the queries.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from hyperdime.contracts.hashing import file_sha256
from hyperdime.contracts.schemas import PayloadRef, SpaceDescriptor
from hyperdime.space.geometry import corpus_covariance
from hyperdime.space.interaction import (
    BACKGROUND_SIZE,
    INTERACTION_FEATURES,
    background_rows,
    interaction_features,
    top_rows,
)
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
_P0: tuple[tuple[str, str], ...] = (
    *((f"doc_{name}", "A") for name in MARGINAL_FEATURES + TAIL_FEATURES),
    *((f"red_{name}", "E") for name in REDUNDANCY_FEATURES),
)
# (name, group) in column order per protocol; the data model fixes it.
FEATURES: dict[str, tuple[tuple[str, str], ...]] = {
    "P0": _P0,
    "P1": (
        *_P0,
        *((f"qry_{name}", "B") for name in MARGINAL_FEATURES + TAIL_FEATURES),
        *((name, "C") for name in INTERACTION_FEATURES if name.startswith("pair_")),
        *((name, "D") for name in INTERACTION_FEATURES if name.startswith("contrast_")),
    ),
}
PROTOCOLS = tuple(FEATURES)


def marginal_columns(
    rows: np.ndarray,
    ids: Sequence[str],
    chunk_size: int = 250_000,
    device: str = "cpu",
    quantile_size: int = QUANTILE_ROWS,
) -> np.ndarray:
    """``[D, 12]`` group A's columns (marginals, then tails) of the embedding ``rows``."""
    moments = document_moments(rows, chunk_size, device)
    tails = tail_share(rows, moments, chunk_size, device)
    quantiles = document_quantiles(rows, quantile_rows(ids, quantile_size))
    columns = [moments.mean, moments.std, moments.mean_abs, moments.rms, *quantiles.T,
               moments.skew, moments.kurt, tails]  # fmt: skip
    return np.stack(columns, axis=1)


def build_descriptors(
    documents: np.ndarray,
    doc_ids: Sequence[str],
    protocol: str = "P0",
    chunk_size: int = 250_000,
    device: str = "cpu",
    quantile_size: int = QUANTILE_ROWS,
    queries: np.ndarray | None = None,
    query_ids: Sequence[str] = (),
    background_size: int = BACKGROUND_SIZE,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """``(descriptor [D, h] float64, covariance [D, D], feature names, groups)``. P1 also needs
    the calibration ``queries`` (rows) and their ``query_ids``."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol {protocol!r} is not built; choose one of {PROTOCOLS}")
    if len(doc_ids) != len(documents):
        raise ValueError("doc_ids must name every document row")
    covariance = corpus_covariance(documents, chunk_size, device)[1].numpy()
    redundancy = redundancy_features(covariance)
    blocks = [marginal_columns(documents, doc_ids, chunk_size, device, quantile_size),
              np.stack([redundancy[n] for n in REDUNDANCY_FEATURES], axis=1)]  # fmt: skip
    if protocol == "P1":
        if queries is None or len(queries) == 0 or len(query_ids) != len(queries):
            raise ValueError("P1 needs calibration queries, each named by query_ids")
        top = top_rows(queries, documents, chunk_size, device)
        background = background_rows(query_ids, doc_ids, background_size)
        blocks += [marginal_columns(queries, query_ids, chunk_size, device, quantile_size),
                   interaction_features(queries, documents, top, background)]  # fmt: skip
    names = [name for name, _ in FEATURES[protocol]]
    groups = sorted({group for _, group in FEATURES[protocol]})
    return np.concatenate(blocks, axis=1), covariance, names, groups


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
