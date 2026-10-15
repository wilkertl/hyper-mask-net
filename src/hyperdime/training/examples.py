"""Training examples for the retrieval objective: queries with their own documents (F5 D1)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class RetrievalExamples:
    """Queries with indices into ``documents`` (unique gathered rows, kept in their dtype).

    ``positives`` is padded with -1 and ``weights`` with 0; weights are the gains
    ``2**grade - 1`` normalized per query. ``negatives`` are the sampled pools, repeats kept.
    """

    queries: Tensor
    documents: Tensor
    positives: Tensor
    weights: Tensor
    negatives: Tensor
    skipped: int

    def __len__(self) -> int:
        return self.queries.shape[0]


def retrieval_examples(
    query_ids: Sequence[str],
    query_vectors: np.ndarray,
    query_rows: Mapping[str, int],
    documents: np.ndarray,
    doc_rows: Mapping[str, int],
    judgments: Mapping[str, Mapping[str, int]],
    pools: Mapping[str, Sequence[str]],
) -> RetrievalExamples:
    """Examples for ``query_ids``, gathering each needed document row of the (possibly
    memory-mapped) matrix once; a query with no positive in the corpus is skipped and counted."""
    kept, labels = [], []
    for qid in query_ids:
        found = {doc_rows[d]: g for d, g in judgments[qid].items() if g > 0 and d in doc_rows}
        if found:
            kept.append(qid)
            labels.append(found)
    needed = sorted({row for found in labels for row in found} | {
        doc_rows[d] for qid in kept for d in pools[qid]
    })  # fmt: skip
    local = {row: i for i, row in enumerate(needed)}
    width = max((len(found) for found in labels), default=1)
    positives = torch.full((len(kept), width), -1, dtype=torch.long)
    weights = torch.zeros(len(kept), width)
    for i, found in enumerate(labels):
        gains = 2.0 ** torch.tensor(list(found.values()), dtype=torch.float32) - 1.0
        positives[i, : len(found)] = torch.tensor([local[row] for row in found])
        weights[i, : len(found)] = gains / gains.sum()
    negatives = torch.tensor([[local[doc_rows[d]] for d in pools[qid]] for qid in kept])
    queries = np.asarray(query_vectors[[query_rows[qid] for qid in kept]], dtype=np.float32)
    return RetrievalExamples(
        torch.from_numpy(queries),
        torch.from_numpy(np.asarray(documents[needed])),
        positives,
        weights,
        negatives.reshape(len(kept), -1),
        len(query_ids) - len(kept),
    )
