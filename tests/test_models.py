from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from models import HyperMaskConfig, HyperMaskNet, select_top_k
from train import kl_loss, oracle_importance


class TinyEncoder(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(32, hidden_size)

    def forward(self, input_ids: torch.Tensor, **_: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def test_hyperhead_receives_kl_gradients() -> None:
    torch.manual_seed(0)
    config = HyperMaskConfig(
        model_name="test",
        hidden_dim=16,
        num_attention_heads=4,
        num_query_slots=2,
        rank=4,
        mask_hidden_dim=4,
        trust_remote_code=False,
    )
    model = HyperMaskNet(config, encoder=TinyEncoder(16))
    ids = torch.randint(0, 32, (3, 5))
    mask = torch.ones_like(ids)
    prediction = model(input_ids=ids, attention_mask=mask)
    query = F.normalize(torch.randn(3, 16), dim=-1)
    target = oracle_importance(query, torch.randn(3, 16), torch.randn(3, 16), 0.1)
    kl_loss(target, prediction).backward()
    assert model.hyperhead.generator[-1].weight.grad is not None
    assert model.hyperhead.generator[-1].weight.grad.norm() > 0
    assert all(parameter.grad is None for parameter in model.encoder.parameters())


def test_top_k_mask_preserves_only_selected_coordinates() -> None:
    importance = torch.tensor([[0.1, 0.8, 0.2, 0.9]])
    embedding = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    masked = select_top_k(importance, embedding, 0.5)
    assert torch.equal(masked, torch.tensor([[0.0, 2.0, 0.0, 4.0]]))
