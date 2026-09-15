"""Training objectives."""

from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F


def selector_kl_loss(target: Tensor, log_prediction: Tensor) -> Tensor:
    """KL(pi_q || pi_hat_q), averaged over the batch."""
    if target.shape != log_prediction.shape:
        raise ValueError("target and log_prediction must share a shape.")
    return F.kl_div(log_prediction, target, reduction="batchmean")
