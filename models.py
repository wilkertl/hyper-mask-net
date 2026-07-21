"""PyTorch modules for query-aware dimension selection.

The encoder is frozen by default.  The Hyperhead attends over its contextual
query states and emits low-rank, per-query weights for a transient Mask-Net.
This keeps document embeddings and their FAISS index fixed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModel


Pooling = Literal["last_token", "cls", "mean"]


@dataclass(frozen=True)
class HyperMaskConfig:
    """Architecture and encoder settings persisted beside model weights."""

    model_name: str
    hidden_dim: int
    pooling: Pooling = "last_token"
    num_query_slots: int = 4
    num_attention_heads: int = 8
    mask_hidden_dim: int | None = None
    rank: int = 32
    dropout: float = 0.1
    prediction_temperature: float = 0.1
    normalize_embeddings: bool = True
    trust_remote_code: bool = True

    def resolved_mask_hidden_dim(self) -> int:
        return self.mask_hidden_dim or self.hidden_dim // 4

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MaskNet(nn.Module):
    """Apply a per-example, low-rank, dynamically generated two-layer MLP."""

    def __init__(self, hidden_dim: int, mask_hidden_dim: int, rank: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mask_hidden_dim = mask_hidden_dim
        self.rank = rank
        self.parameter_count = (
            mask_hidden_dim * rank
            + rank * hidden_dim
            + hidden_dim * rank
            + rank * mask_hidden_dim
            + mask_hidden_dim
            + hidden_dim
        )

    def unpack_parameters(self, flat_parameters: Tensor) -> tuple[Tensor, ...]:
        """Convert a `[batch, parameter_count]` vector to Mask-Net tensors."""
        if flat_parameters.ndim != 2 or flat_parameters.shape[-1] != self.parameter_count:
            raise ValueError(
                f"Expected [batch, {self.parameter_count}] parameters, got "
                f"{tuple(flat_parameters.shape)}."
            )
        batch_size = flat_parameters.shape[0]
        h, d, r = self.mask_hidden_dim, self.hidden_dim, self.rank
        shapes = ((h, r), (r, d), (d, r), (r, h), (h,), (d,))
        tensors: list[Tensor] = []
        offset = 0
        for shape in shapes:
            count = int(torch.tensor(shape).prod().item())
            tensors.append(flat_parameters[:, offset : offset + count].reshape(batch_size, *shape))
            offset += count
        return tuple(tensors)

    def forward(self, query_embedding: Tensor, flat_parameters: Tensor) -> Tensor:
        """Return unnormalized importance logits with shape `[batch, D]`."""
        if query_embedding.ndim != 2 or query_embedding.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected query embeddings [batch, {self.hidden_dim}], got "
                f"{tuple(query_embedding.shape)}."
            )
        a1, b1, a2, b2, bias1, bias2 = self.unpack_parameters(flat_parameters)
        # W1=[B,H,D], W2=[B,D,H]; factorization avoids generating dense weights.
        weight1 = a1 @ b1
        hidden = F.gelu(torch.einsum("bd,bhd->bh", query_embedding, weight1) + bias1)
        weight2 = a2 @ b2
        return torch.einsum("bh,bdh->bd", hidden, weight2) + bias2


class Hyperhead(nn.Module):
    """Cross-attend learned slots to tokens and generate Mask-Net parameters."""

    def __init__(self, config: HyperMaskConfig) -> None:
        super().__init__()
        if config.hidden_dim % config.num_attention_heads:
            raise ValueError("hidden_dim must be divisible by num_attention_heads.")
        self.config = config
        self.mask_net = MaskNet(
            config.hidden_dim, config.resolved_mask_hidden_dim(), config.rank
        )
        self.query_slots = nn.Parameter(
            torch.empty(config.num_query_slots, config.hidden_dim)
        )
        nn.init.normal_(self.query_slots, std=config.hidden_dim**-0.5)
        self.cross_attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.generator = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, self.mask_net.parameter_count),
        )

    def forward(self, token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
        """Generate `[batch, MaskNet.parameter_count]` ephemeral parameters."""
        if token_embeddings.ndim != 3:
            raise ValueError("token_embeddings must have shape [batch, sequence, hidden].")
        if attention_mask.shape != token_embeddings.shape[:2]:
            raise ValueError("attention_mask must have shape [batch, sequence].")
        slots = self.query_slots.unsqueeze(0).expand(token_embeddings.shape[0], -1, -1)
        attended, _ = self.cross_attention(
            slots,
            token_embeddings,
            token_embeddings,
            key_padding_mask=~attention_mask.bool(),
            need_weights=False,
        )
        return self.generator(attended.mean(dim=1))


class HyperMaskNet(nn.Module):
    """Frozen transformer encoder plus dynamic Hyperhead and Mask-Net."""

    def __init__(self, config: HyperMaskConfig, encoder: nn.Module | None = None) -> None:
        super().__init__()
        self.config = config
        self.encoder = encoder or AutoModel.from_pretrained(
            config.model_name, trust_remote_code=config.trust_remote_code
        )
        encoder_dim = getattr(self.encoder.config, "hidden_size", None)
        if encoder_dim != config.hidden_dim:
            raise ValueError(
                f"Encoder hidden size ({encoder_dim}) does not match config hidden_dim "
                f"({config.hidden_dim})."
            )
        self.hyperhead = Hyperhead(config)
        self.freeze_encoder()

    def freeze_encoder(self) -> None:
        """Freeze the retrieval encoder while retaining gradient flow to Hyperhead."""
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode: bool = True) -> "HyperMaskNet":
        super().train(mode)
        self.encoder.eval()  # Frozen encoders must not update dropout statistics.
        return self

    def pool(self, token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
        """Pool token representations with the configured embedding convention."""
        if self.config.pooling == "cls":
            pooled = token_embeddings[:, 0]
        elif self.config.pooling == "mean":
            weights = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
            pooled = (token_embeddings * weights).sum(1) / weights.sum(1).clamp_min(1)
        else:
            last_positions = attention_mask.long().sum(dim=1).sub(1).clamp_min(0)
            pooled = token_embeddings[
                torch.arange(token_embeddings.shape[0], device=token_embeddings.device),
                last_positions,
            ]
        return F.normalize(pooled, dim=-1) if self.config.normalize_embeddings else pooled

    def encode_query(self, **encoded_query: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return contextual tokens, pooled query embedding, and attention mask."""
        attention_mask = encoded_query["attention_mask"].bool()
        with torch.no_grad():
            outputs = self.encoder(**encoded_query)
            token_embeddings = outputs.last_hidden_state
            query_embedding = self.pool(token_embeddings, attention_mask)
        return token_embeddings, query_embedding, attention_mask

    def forward(self, **encoded_query: Tensor) -> Tensor:
        """Predict the per-query dimension importance distribution `[batch, D]`."""
        tokens, query_embedding, attention_mask = self.encode_query(**encoded_query)
        if isinstance(self.hyperhead, LinearMaskPredictor):
            return self.hyperhead(query_embedding)
        parameters = self.hyperhead(tokens, attention_mask)
        logits = self.hyperhead.mask_net(query_embedding, parameters)
        return F.softmax(logits.float() / self.config.prediction_temperature, dim=-1)

    @torch.no_grad()
    def predict_embedding_and_importance(
        self, **encoded_query: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return the unmasked embedding and predicted importance distribution."""
        tokens, query_embedding, attention_mask = self.encode_query(**encoded_query)
        if isinstance(self.hyperhead, LinearMaskPredictor):
            return query_embedding, self.hyperhead(query_embedding)
        parameters = self.hyperhead(tokens, attention_mask)
        logits = self.hyperhead.mask_net(query_embedding, parameters)
        importance = F.softmax(logits.float() / self.config.prediction_temperature, dim=-1)
        return query_embedding, importance


class LinearMaskPredictor(nn.Module):
    """Paper-faithful supervised DIME baseline: a single D-to-D projection."""

    def __init__(self, hidden_dim: int, prediction_temperature: float = 0.1) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.prediction_temperature = prediction_temperature

    def forward(self, query_embedding: Tensor) -> Tensor:
        return F.softmax(
            self.projection(query_embedding).float() / self.prediction_temperature, dim=-1
        )


def select_top_k(importance: Tensor, query_embedding: Tensor, k: int | float) -> Tensor:
    """Hard-mask all but the top-k dimensions of each query embedding."""
    if importance.shape != query_embedding.shape or importance.ndim != 2:
        raise ValueError("importance and query_embedding must both have shape [batch, D].")
    dimensions = importance.shape[-1]
    count = round(k * dimensions) if isinstance(k, float) else k
    if not isinstance(count, int) or not 1 <= count <= dimensions:
        raise ValueError(f"k must resolve to an integer in [1, {dimensions}], got {k}.")
    indices = importance.topk(count, dim=-1).indices
    mask = torch.zeros_like(importance).scatter_(1, indices, 1)
    return query_embedding * mask
