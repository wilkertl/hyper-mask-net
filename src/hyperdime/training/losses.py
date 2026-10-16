"""Training objectives."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def selector_kl_loss(target: Tensor, log_prediction: Tensor) -> Tensor:
    """KL(pi_q || pi_hat_q), averaged over the batch."""
    if target.shape != log_prediction.shape:
        raise ValueError("target and log_prediction must share a shape.")
    return F.kl_div(log_prediction, target, reduction="batchmean")


def masked_retrieval_loss(
    queries: Tensor,
    mask: Tensor,
    documents: Tensor,
    positives: Tensor,
    weights: Tensor,
    negatives: Tensor,
    score_temperature: float,
) -> Tensor:
    """Multi-positive softmax cross-entropy of the masked query's scores, averaged over the batch.

    Each query ``i`` scores ``<queries[i] * mask[i], d> / score_temperature`` (the masked query is
    not re-normalized) against its positives (``positives[i]``, padded with -1, weighted by
    ``weights[i]``, which sum to 1), its hard negatives (``negatives[i]``, repeats counted), and the
    distinct positive documents of the other queries in the batch (each once), except documents that
    are also its own positives. Indices point into ``documents``.
    """
    masked = queries * mask
    valid = positives >= 0
    rows = positives.clamp(min=0)
    positive_logits = torch.einsum("bd,bpd->bp", masked, documents[rows]) / score_temperature
    negative_logits = torch.einsum("bd,bnd->bn", masked, documents[negatives]) / score_temperature
    batch_docs, column = torch.unique(rows[valid], return_inverse=True)
    own = torch.zeros(len(queries), len(batch_docs), dtype=torch.bool, device=queries.device)
    own[valid.nonzero()[:, 0], column] = True
    batch_logits = masked @ documents[batch_docs].T / score_temperature
    candidates = torch.cat(
        [
            positive_logits.masked_fill(~valid, float("-inf")),
            negative_logits,
            batch_logits.masked_fill(own, float("-inf")),
        ],
        dim=-1,
    )
    log_z = torch.logsumexp(candidates, dim=-1, keepdim=True)
    log_p = (positive_logits - log_z).masked_fill(~valid, 0.0)
    return -(weights * log_p).sum(-1).mean()
