"""Query-only dimension importance predictor from Learning to Select."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from hyperdime.embeddings.qwen import EMBEDDING_DIM
from hyperdime.selection.topk import resolve_k

# The 10%-90% budgets of the 1024 dimensions, one prior weight each in the budget selector (S4).
BUDGETS = tuple(resolve_k(step / 10, EMBEDDING_DIM) for step in range(1, 10))


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

    def forward(self, query_embeddings: Tensor, k: int | None = None) -> Tensor:
        prior, correction = self.terms(query_embeddings, k)
        return F.log_softmax(prior + correction, dim=-1)

    def terms(self, query_embeddings: Tensor, k: int | None = None) -> tuple[Tensor, Tensor]:
        """The weighted prior ``a · log(q² · Var + ε)`` and the correction ``W q + b`` before the
        log-softmax; ``k`` matters only to selectors with a weight per budget."""
        queries = query_embeddings.float()
        prior = torch.log(queries**2 * self.variance + self.EPSILON)
        weight = self.weight_at(k)
        if weight is not None:
            prior = weight * prior
        return prior, self.projection(queries)

    def weight_at(self, k: int | None) -> Tensor | None:
        """The prior's weight at budget ``k``: the one learned weight, or none (weight 1)."""
        return self.prior_weight


class BudgetGeometricSelector(GeometricSelector):
    """``a_k · log(q_d² · Var(x_d) + ε) + (W q + b)_d`` followed by log-softmax (S4, research D1).

    One learned prior weight per budget, all starting at 1, with the linear map shared by every
    budget: an untrained budget selector ranks dimensions as the ``variance`` method at every
    budget. The budgets are a buffer, so a checkpoint describes itself; any other ``k`` is refused.
    """

    budgets: Tensor

    def __init__(self, variance: Tensor, budgets: Sequence[int] = BUDGETS) -> None:
        if not budgets or any(b <= a for a, b in zip(budgets, budgets[1:], strict=False)):
            raise ValueError(f"budgets must be non-empty and strictly increasing, got {budgets}")
        super().__init__(variance)
        self.prior_weights = nn.Parameter(torch.ones(len(budgets)))
        self.register_buffer("budgets", torch.tensor(list(budgets), dtype=torch.int64))

    def weight_at(self, k: int | None) -> Tensor:
        found = (self.budgets == k).nonzero() if k is not None else self.budgets[:0]
        if len(found) == 0:
            raise ValueError(f"budget {k} is not one of {self.budgets.tolist()}")
        return self.prior_weights[int(found[0, 0])]
