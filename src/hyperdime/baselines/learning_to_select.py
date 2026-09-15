"""Query-only dimension importance predictor from Learning to Select."""

from __future__ import annotations

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
