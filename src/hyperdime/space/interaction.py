"""Descriptor groups C and D from unlabeled calibration queries (P1, F10 research D5).

The calibration set is training-split queries only, at most ``CALIBRATION_QUERIES`` by a seeded
hash of their ids, so it never meets the validation queries the targets use. Each calibration
query's top-``TOP_DEPTH`` documents by exact full-query inner product stand in for relevance (no
judgment is read), and ``BACKGROUND_SIZE`` documents drawn per query stand in for the corpus.

- C: per dimension, the mean, std, q05, q50, and q95 of ``q_j · d_j`` over (query, background
  document) pairs and over (query, top-10 document) pairs;
- D: ``contrast_K[j]``, the mean over queries of ``q_j · (mean_{top-K} d_j - mean_{bg} d_j)``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import numpy as np
import torch

from hyperdime.retrieval.exact import exact_search_streamed
from hyperdime.training.retrieval_selection import candidate_queries

CALIBRATION_SPLIT = "train"
CALIBRATION_QUERIES = 10_000
CALIBRATION_SEED = 0
TOP_DEPTH = 20
TOP_PAIRS = 10
BACKGROUND_SIZE = 100
BACKGROUND_SEED = 0
CONTRAST_K = (5, 10, 20)
PAIR_STATS = ("mean", "std", "q05", "q50", "q95")
INTERACTION_FEATURES = (
    *(f"pair_{kind}_{stat}" for kind in ("rand", "top") for stat in PAIR_STATS),
    *(f"contrast_k{k}" for k in CONTRAST_K),
)
COLUMN_BLOCK = 64


def calibration_queries(
    splits: Mapping[str, Sequence[str]], size: int = CALIBRATION_QUERIES
) -> list[str]:
    """Sorted ids of the at most ``size`` training queries smallest under a seeded hash."""
    train = list(splits.get(CALIBRATION_SPLIT, ()))
    if not train:
        raise ValueError(f"P1 needs {CALIBRATION_SPLIT}-split queries; there are none")
    return sorted(train[row] for row in candidate_queries(train, size, CALIBRATION_SEED))


def background_rows(
    query_ids: Sequence[str],
    doc_ids: Sequence[str],
    size: int = BACKGROUND_SIZE,
    seed: int = BACKGROUND_SEED,
) -> np.ndarray:
    """``[Q, min(size, N)]`` document rows: per query, a draw without replacement seeded by its id,
    over the documents in id order, so neither query nor document row order changes it."""
    by_id = np.argsort(np.asarray(doc_ids), kind="stable")
    count = min(size, len(doc_ids))
    found = np.empty((len(query_ids), count), dtype=np.int64)
    for i, qid in enumerate(query_ids):
        key = int.from_bytes(hashlib.sha256(f"{seed}:{qid}".encode()).digest()[:8], "big")
        drawn = np.random.default_rng(key).choice(len(doc_ids), count, replace=False)
        found[i] = by_id[np.sort(drawn)]
    return found


def top_rows(
    queries: np.ndarray, documents: np.ndarray, chunk_size: int = 250_000, device: str = "cpu"
) -> np.ndarray:
    """``[Q, min(TOP_DEPTH, N)]`` rows of each query's top documents by exact inner product."""
    depth = min(TOP_DEPTH, len(documents))
    found = torch.from_numpy(np.asarray(queries, dtype=np.float32))
    rows = exact_search_streamed(found, documents, depth, chunk_size=chunk_size, device=device)
    return rows.cpu().numpy()


def _pair_stats(products: np.ndarray) -> list[np.ndarray]:
    """``PAIR_STATS`` per column of ``[pairs, b]`` products."""
    quantiles = np.quantile(products, (0.05, 0.5, 0.95), axis=0)
    return [products.mean(axis=0), products.std(axis=0), *quantiles]


def interaction_features(
    queries: np.ndarray, documents: np.ndarray, top: np.ndarray, background: np.ndarray
) -> np.ndarray:
    """``[D, 13]`` groups C and D in ``INTERACTION_FEATURES`` order from ``queries`` ``[Q, D]``,
    their ``top`` document rows ``[Q, T]`` (best first), and ``background`` rows ``[Q, B]``. With
    fewer than K top rows, ``contrast_K`` uses those there are."""
    if not len(queries) == len(top) == len(background):
        raise ValueError("queries, top, and background must have one row per query")
    rows = np.unique(np.concatenate([top.ravel(), background.ravel()]))
    stored = np.asarray(documents[rows])  # each document once, in its stored dtype
    top_at, background_at = np.searchsorted(rows, top), np.searchsorted(rows, background)
    found = np.empty((queries.shape[1], len(INTERACTION_FEATURES)))
    for start in range(0, queries.shape[1], COLUMN_BLOCK):
        block = slice(start, start + COLUMN_BLOCK)
        q = np.asarray(queries[:, block], dtype=np.float64)
        docs = stored[:, block].astype(np.float64)
        drawn, best = docs[background_at], docs[top_at]  # [Q, B, b], [Q, T, b]
        pairs_rand = (q[:, None] * drawn).reshape(-1, q.shape[1])
        pairs_top = (q[:, None] * best[:, :TOP_PAIRS]).reshape(-1, q.shape[1])
        contrast = [(q * (best[:, :k].mean(axis=1) - drawn.mean(axis=1))).mean(axis=0)
                    for k in CONTRAST_K]  # fmt: skip
        found[block] = np.stack([*_pair_stats(pairs_rand), *_pair_stats(pairs_top), *contrast], 1)
    return found
