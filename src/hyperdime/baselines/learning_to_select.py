"""Query-only dimension importance predictor from Learning to Select."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from hyperdime.embeddings.qwen import EMBEDDING_DIM


class LinearSelector(nn.Module):
    """A single D-to-D linear layer followed by log-softmax over dimensions."""

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        super().__init__()
        self.projection = nn.Linear(dim, dim)

    def forward(self, query_embeddings: Tensor) -> Tensor:
        return F.log_softmax(self.projection(query_embeddings).float(), dim=-1)


class GeometricSelector(LinearSelector):
    """``log(q_d² · Var(x_d) + ε) + (W q + b)_d`` followed by log-softmax (S2).

    The prior is fixed (a buffer computed from the corpus's document embeddings), so the only
    learned parameters are the L2S linear map, which starts at zero: an untrained geometric
    selector ranks dimensions exactly as the ``variance`` method. With ``learn_weight`` (S3) the
    prior is scaled by one learned scalar ``prior_weight`` that starts at 1.
    """

    EPSILON = 1e-12
    variance: Tensor

    def __init__(self, variance: Tensor, learn_weight: bool = False) -> None:
        super().__init__(variance.shape[0])
        self.prior_weight: nn.Parameter | None = (
            nn.Parameter(torch.ones(())) if learn_weight else None
        )
        self.register_buffer("variance", variance.detach().float().clone())
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, query_embeddings: Tensor) -> Tensor:
        queries = query_embeddings.float()
        prior = torch.log(queries**2 * self.variance + self.EPSILON)
        if self.prior_weight is not None:
            prior = self.prior_weight * prior
        return F.log_softmax(prior + self.projection(queries), dim=-1)
