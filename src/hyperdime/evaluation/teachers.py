"""F12 teacher masks on fit queries (ADR-0018, "Teachers").

positive:    top-k of ``q * p_bar`` (gain-weighted positive mean, weights ``2^g - 1``)
contrastive: top-k of ``q * (p_bar - n_bar)`` with the recorded ``l2s-official`` negatives
rank-guided: steepest-ascent search over swaps (one dimension out, one in; k fixed), accepting a
             swap only if it is strictly better under the lexicographic objective (nDCG@10, GDR)

``GDR(q, m) = sum_p w_p / log2(1 + r_p(m))`` with ``w_p = (2^g_p - 1) / sum(2^g - 1)`` and
``r_p`` the 1-based full-corpus rank under the lower-row rule. Every swap is scored exactly against
the full corpus (float64 of float16 vectors is exact), and after each accepted swap the scores are
recomputed from scratch and must equal the incremental ones. Among equally good swaps the one with
the lowest (out, in) dimension pair wins. No result is claimed to be globally optimal.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import torch
from torch import Tensor

from hyperdime.evaluation.benchmark import CorpusScorer, ideal_dcg, masked_query, positive_ranks


def top_k_dims(scores: np.ndarray, k: int) -> np.ndarray:
    """The k highest-scoring dimensions, ties to the lower index, sorted ascending."""
    order = np.argsort(-scores, kind="stable")[:k]
    return np.sort(order).astype(np.int64)


def positive_mean(positives: np.ndarray, grades: np.ndarray) -> np.ndarray:
    weights = np.power(2.0, grades) - 1.0
    return cast(np.ndarray, (positives * (weights / weights.sum())[:, None]).sum(0))


def positive_mask(query: np.ndarray, p_bar: np.ndarray, k: int) -> np.ndarray:
    return top_k_dims(query * p_bar, k)


def contrastive_mask(query: np.ndarray, p_bar: np.ndarray, n_bar: np.ndarray, k: int) -> np.ndarray:
    return top_k_dims(query * (p_bar - n_bar), k)


def gdr_weights(grades: Tensor) -> Tensor:
    gains = torch.pow(2.0, grades.to(torch.float64)) - 1.0
    return gains / gains.sum()


@dataclass
class Objective:
    """Lexicographic (nDCG@10, GDR) of one query; float64 values computed in one fixed order."""

    gains: Tensor  # linear gains for nDCG@10, [P]
    weights: Tensor  # GDR weights, [P]
    idcg: float

    def __call__(self, ranks: Tensor) -> tuple[Tensor, Tensor]:
        discount = 1.0 / torch.log2(1.0 + ranks.to(torch.float64))
        ndcg = (self.gains * discount * (ranks <= 10)).sum(-1) / self.idcg
        gdr = (self.weights * discount).sum(-1)
        return ndcg, gdr


def make_objective(grades: Tensor, device: torch.device) -> Objective:
    gains = grades.to(torch.float64).to(device)
    return Objective(gains, gdr_weights(grades).to(device), ideal_dcg(gains.tolist()))


@dataclass
class SearchResult:
    dims: np.ndarray
    ndcg: float
    gdr: float
    start_ndcg: float
    start_gdr: float
    steps: int
    hit_cap: bool
    seconds: float
    step_seconds: list[float] = field(default_factory=list)


def _value(pair: tuple[Tensor, Tensor]) -> tuple[float, float]:
    return float(pair[0]), float(pair[1])


def better(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] > b[0] or (a[0] == b[0] and a[1] > b[1])


def rank_guided_search(
    documents: Tensor,
    query: Tensor,
    positive_rows: Tensor,
    objective: Objective,
    start: np.ndarray,
    cap: int,
    max_elements: int = 1 << 26,
) -> SearchResult:
    """Search from ``start`` (sorted dims) for at most ``cap`` accepted swaps.

    ``documents`` is [N, D] float64 on the search device; ``query`` [D] float64 on the same device.
    """
    began = time.perf_counter()
    device = documents.device
    dim = documents.shape[1]
    mask = torch.zeros(dim, dtype=torch.bool, device=device)
    mask[torch.as_tensor(start, device=device)] = True
    contribution = documents * query  # [N, D], exact
    scores = contribution[:, mask].sum(1)
    current = _value(objective(positive_ranks(scores, positive_rows)))
    start_value = current
    steps, step_seconds = 0, []
    n_docs = documents.shape[0]
    while steps < cap:
        tick = time.perf_counter()
        selected = torch.nonzero(mask).squeeze(1)
        unselected = torch.nonzero(~mask).squeeze(1)
        add = contribution[:, unselected].T.contiguous()  # [nI, N]
        chunk = max(1, max_elements // (len(unselected) * n_docs))
        best: tuple[float, float] | None = None
        best_pair = (-1, -1)
        for start_out in range(0, len(selected), chunk):
            outs = selected[start_out : start_out + chunk]
            base = scores[None, :] - contribution[:, outs].T  # [b, N]
            candidate = base[:, None, :] + add[None, :, :]  # [b, nI, N]
            ndcg, gdr = objective(positive_ranks(candidate, positive_rows))
            top_ndcg = ndcg.max()
            gdr_masked = torch.where(ndcg == top_ndcg, gdr, torch.full_like(gdr, -math.inf))
            top_gdr = gdr_masked.max()
            value = (float(top_ndcg), float(top_gdr))
            if best is None or better(value, best):
                flat = torch.nonzero((ndcg == top_ndcg) & (gdr_masked == top_gdr))[0]
                best = value
                best_pair = (int(outs[flat[0]]), int(unselected[flat[1]]))
        assert best is not None
        if not better(best, current):
            step_seconds.append(time.perf_counter() - tick)
            break
        out_dim, in_dim = best_pair
        scores = scores - contribution[:, out_dim] + contribution[:, in_dim]
        mask[out_dim] = False
        mask[in_dim] = True
        recomputed = contribution[:, mask].sum(1)
        if not torch.equal(recomputed, scores):
            raise RuntimeError("incremental scores differ from an exact recomputation")
        accepted = _value(objective(positive_ranks(recomputed, positive_rows)))
        if accepted != best:
            raise RuntimeError("accepted swap's exact objective differs from its candidate value")
        current = accepted
        steps += 1
        step_seconds.append(time.perf_counter() - tick)
    hit_cap = steps == cap
    return SearchResult(
        dims=torch.nonzero(mask).squeeze(1).cpu().numpy(),
        ndcg=current[0],
        gdr=current[1],
        start_ndcg=start_value[0],
        start_gdr=start_value[1],
        steps=steps,
        hit_cap=hit_cap,
        seconds=time.perf_counter() - began,
        step_seconds=step_seconds,
    )


@dataclass
class ShortlistResult(SearchResult):
    rejected: int = 0  # proposals that failed the exact full-corpus check


def shortlist_search(
    scorer: CorpusScorer,
    query: Tensor,
    positive_rows: Tensor,
    objective: Objective,
    start: np.ndarray,
    cap: int,
    shortlist: int,
) -> ShortlistResult:
    """Rank-guided search for corpora too large for exact swap scans (ADR-0018, "Teachers").

    Each step proposes the best swap scored on a sub-corpus: the positives and the ``shortlist``
    best other documents under the current mask, kept in corpus row order so the lower-row rule
    holds inside it. The proposal is accepted only if its exact full-corpus objective is strictly
    better than the current exact value; otherwise the search stops and the rejection is counted.
    Reported values are exact full-corpus values, and the result is never below the start.
    """
    began = time.perf_counter()
    device = scorer.device
    query = query.to(device, torch.float64)
    positives = [int(r) for r in positive_rows.tolist()]
    pos = torch.as_tensor(positives, device=device)

    def exact(dims: np.ndarray) -> tuple[float, float]:
        masked = masked_query(query, torch.as_tensor(dims, device=device))
        ranks = scorer.positive_ranks(masked[None], [pos])[0].to(device)
        return _value(objective(ranks))

    dims = np.sort(np.asarray(start, dtype=np.int64))
    current = exact(dims)
    start_value = current
    steps, rejected, step_seconds = 0, 0, []
    while steps < cap:
        tick = time.perf_counter()
        masked = masked_query(query, torch.as_tensor(dims, device=device))
        rows = sorted(set(scorer.top(masked, shortlist, exclude=positives)) | set(positives))
        local = {row: i for i, row in enumerate(rows)}
        proposal = rank_guided_search(
            scorer.rows_f64(rows),
            query,
            torch.as_tensor([local[r] for r in positives], device=device),
            objective,
            dims,
            cap=1,
        )
        if proposal.steps == 0:
            step_seconds.append(time.perf_counter() - tick)
            break
        candidate = np.sort(proposal.dims)
        value = exact(candidate)
        if not better(value, current):
            rejected += 1
            step_seconds.append(time.perf_counter() - tick)
            break
        dims, current = candidate, value
        steps += 1
        step_seconds.append(time.perf_counter() - tick)
    if better(start_value, current):
        raise RuntimeError("the search ended below its starting mask")
    return ShortlistResult(
        dims=dims,
        ndcg=current[0],
        gdr=current[1],
        start_ndcg=start_value[0],
        start_gdr=start_value[1],
        steps=steps,
        hit_cap=steps == cap,
        seconds=time.perf_counter() - began,
        step_seconds=step_seconds,
        rejected=rejected,
    )
