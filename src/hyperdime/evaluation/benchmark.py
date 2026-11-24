"""Exact full-corpus evaluation for the F12 benchmark (ADR-0018, "Evaluation").

Scores are float64 dot products of float16 vectors, which are exact in any order of summation (see
``representation_ranking``), so ties are true ties. A positive's rank counts every document scoring
higher plus every tied document on a lower row (the project's lower-row rule). Every metric here
depends only on the positives' ranks: documents without a positive grade carry no gain.

nDCG@10 uses the project's linear gain (``metrics.query_metrics``, as trec_eval) with the ideal
taken over all of the query's positives. The authors' evaluator uses ``2^g - 1`` and an ideal over
the retrieved list; that convention is used only in the parity check against their code.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import numpy as np
import torch
from torch import Tensor

from hyperdime.selection.topk import resolve_k

DIM = 1024
DECILES = tuple(resolve_k(step / 10, DIM) for step in range(1, 11))
AUTHORS_GRID = tuple(max(1, math.ceil(DIM * percent / 100)) for percent in range(10, 101, 10))
PRIMARY = DECILES[:-1]


def positive_ranks(scores: Tensor, positive_rows: Tensor) -> Tensor:
    """1-based ranks of ``positive_rows`` under ``scores`` ([..., N]), lower-row ties."""
    ranks = []
    rows = torch.arange(scores.shape[-1], device=scores.device)
    for row in positive_rows.tolist():
        own = scores[..., row : row + 1]
        higher = (scores > own).sum(-1)
        tied_lower = ((scores == own) & (rows < row)).sum(-1)
        ranks.append(1 + higher + tied_lower)
    return torch.stack(ranks, -1)


def ideal_dcg(gains: Sequence[float], cutoff: int = 10) -> float:
    ordered = sorted(gains, reverse=True)[:cutoff]
    return sum(gain / math.log2(position + 2) for position, gain in enumerate(ordered))


def ndcg10(ranks: Tensor, gains: Tensor, idcg: float) -> Tensor:
    """Linear-gain nDCG@10 from positive ranks ([..., P]) and gains ([P])."""
    discount = 1.0 / torch.log2(1.0 + ranks.to(torch.float64))
    return (gains * discount * (ranks <= 10)).sum(-1) / idcg


def query_measures(ranks: Tensor, gains: Tensor) -> dict[str, float]:
    """nDCG@10, MRR@10, Recall@100, nDCG@100 of one query from its positive ranks ([P])."""
    rank_list = ranks.tolist()
    gain_list = gains.tolist()
    first = min(rank_list)
    dcg100 = sum(
        g / math.log2(1 + r) for r, g in zip(rank_list, gain_list, strict=True) if r <= 100
    )
    return {
        "ndcg@10": float(ndcg10(ranks, gains, ideal_dcg(gain_list))),
        "mrr@10": 1.0 / first if first <= 10 else 0.0,
        "recall@100": sum(r <= 100 for r in rank_list) / len(rank_list),
        "ndcg@100": dcg100 / ideal_dcg(gain_list, 100),
    }


def masked_scores(documents: Tensor, query: Tensor, dims: Tensor | None) -> Tensor:
    """Exact scores of every document for the query restricted to ``dims`` (no renormalization)."""
    if dims is None:
        return documents @ query
    return documents[:, dims] @ query[dims]


class CorpusScorer:
    """Exact full-corpus ranks without a float64 copy of the corpus (MS MARCO-sized corpora).

    The float16 document matrix stays on ``device`` (or is streamed from host memory when
    ``resident`` is false) and is widened to float64 one chunk at a time. Products of float16
    values and their sums are exact in float64, so every score, and therefore every rank under the
    lower-row rule, equals :func:`masked_scores` with :func:`positive_ranks` bit for bit. A masked
    query is the query with zeros outside its dimensions; the zero terms add exactly nothing.
    """

    def __init__(
        self,
        documents: np.ndarray,
        device: str | torch.device,
        *,
        chunk_rows: int = 1 << 18,
        resident: bool = True,
        max_elements: int = 1 << 27,
    ) -> None:
        self.device = torch.device(device)
        self.rows = int(documents.shape[0])
        self.chunk_rows = chunk_rows
        self.max_elements = max_elements
        self._host = np.asarray(documents)
        self._resident = (
            torch.from_numpy(self._host).to(self.device) if resident else None
        )  # float16

    def _chunks(self) -> Iterator[tuple[int, Tensor]]:
        for start in range(0, self.rows, self.chunk_rows):
            stop = min(start + self.chunk_rows, self.rows)
            if self._resident is not None:
                block = self._resident[start:stop]
            else:
                block = torch.from_numpy(self._host[start:stop]).to(self.device)
            yield start, block.to(torch.float64)

    def rows_f64(self, rows: Sequence[int] | Tensor) -> Tensor:
        index = torch.as_tensor(rows, dtype=torch.long)
        if self._resident is not None:
            return self._resident[index.to(self.device)].to(torch.float64)
        # the index may live on the GPU (positive_ranks moves it there); numpy needs it on the CPU
        return torch.from_numpy(self._host[index.cpu().numpy()]).to(self.device, torch.float64)

    def positive_ranks(self, queries: Tensor, positives: Sequence[Tensor]) -> list[Tensor]:
        """1-based ranks of each query's positive rows; ``queries`` is [B, D] float64 masked."""
        queries = queries.to(self.device, torch.float64)
        owners = torch.cat(
            [torch.full((len(p),), b, dtype=torch.long) for b, p in enumerate(positives)]
        ).to(self.device)
        rows = torch.cat([torch.as_tensor(p, dtype=torch.long) for p in positives]).to(self.device)
        own = (self.rows_f64(rows) * queries[owners]).sum(1)  # [M] exact
        counts = torch.zeros(len(rows), dtype=torch.long, device=self.device)
        step = max(1, self.max_elements // max(len(rows), 1))
        for start, block in self._chunks():
            for offset in range(0, block.shape[0], step):
                part = block[offset : offset + step]
                scores = (part @ queries.T)[:, owners]  # [c, M]
                index = torch.arange(
                    start + offset, start + offset + part.shape[0], device=self.device
                )
                higher = scores > own
                tied_lower = (scores == own) & (index[:, None] < rows[None, :])
                counts += (higher | tied_lower).sum(0)
        ranks = 1 + counts
        out, at = [], 0
        for p in positives:
            out.append(ranks[at : at + len(p)].cpu())
            at += len(p)
        return out

    def top_many(
        self, queries: Tensor, n: int, excludes: Sequence[Sequence[int]]
    ) -> list[list[int]]:
        """:meth:`top` for a batch of queries in one pass over the corpus."""
        queries = queries.to(self.device, torch.float64)
        count = queries.shape[0]
        best_scores = torch.empty((count, 0), dtype=torch.float64, device=self.device)
        best_rows = torch.empty((count, 0), dtype=torch.long, device=self.device)
        skip = [torch.as_tensor(list(e), dtype=torch.long, device=self.device) for e in excludes]
        for start, block in self._chunks():
            scores = queries @ block.T  # [B, c]
            index = torch.arange(start, start + block.shape[0], device=self.device)
            for b, rows in enumerate(skip):
                if len(rows):
                    scores[b, torch.isin(index, rows)] = -torch.inf
            scores = torch.cat([best_scores, scores], 1)
            rows_all = torch.cat([best_rows, index.expand(count, -1)], 1)
            order = torch.sort(-scores, dim=1, stable=True).indices[:, :n]
            best_scores = torch.gather(scores, 1, order)
            best_rows = torch.gather(rows_all, 1, order)
        return [[int(r) for r in row] for row in best_rows.tolist()]

    def variance(self) -> Tensor:
        """Per-dimension population variance of the documents, two passes in float64."""
        total = torch.zeros(self._host.shape[1], dtype=torch.float64, device=self.device)
        for _, block in self._chunks():
            total += block.sum(0)
        mean = total / self.rows
        squares = torch.zeros_like(total)
        for _, block in self._chunks():
            squares += ((block - mean) ** 2).sum(0)
        return squares / self.rows

    def top(self, query: Tensor, n: int, exclude: Sequence[int] = ()) -> list[int]:
        """The ``n`` best rows for one float64 query, ties to the lower row, ``exclude`` skipped."""
        query = query.to(self.device, torch.float64)
        skip = torch.as_tensor(list(exclude), dtype=torch.long, device=self.device)
        best_scores = torch.empty(0, dtype=torch.float64, device=self.device)
        best_rows = torch.empty(0, dtype=torch.long, device=self.device)
        for start, block in self._chunks():
            scores = block @ query
            index = torch.arange(start, start + block.shape[0], device=self.device)
            if len(skip):
                keep = ~torch.isin(index, skip)
                scores, index = scores[keep], index[keep]
            scores = torch.cat([best_scores, scores])
            index = torch.cat([best_rows, index])
            # rows are increasing within the concatenation, so a stable sort keeps lower rows first
            order = torch.sort(-scores, stable=True).indices[:n]
            best_scores, best_rows = scores[order], index[order]
        return [int(r) for r in best_rows.tolist()]


def masked_query(query: Tensor, dims: Tensor | None) -> Tensor:
    """The float64 query with zeros outside ``dims`` (``None`` keeps every dimension)."""
    query = query.to(torch.float64)
    if dims is None:
        return query.clone()
    out = torch.zeros_like(query)
    out[dims] = query[dims]
    return out
