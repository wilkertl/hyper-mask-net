import torch
from torch.nn import functional as F

from hyperdime.baselines.learning_to_select import LinearSelector
from hyperdime.oracle.importance import oracle_importance
from hyperdime.training.losses import selector_kl_loss


def test_linear_selector_outputs_log_distribution_over_dimensions() -> None:
    torch.manual_seed(0)
    log_probs = LinearSelector(dim=8)(torch.randn(3, 8))
    assert log_probs.shape == (3, 8)
    assert torch.allclose(log_probs.exp().sum(dim=-1), torch.ones(3))


def test_kl_loss_is_zero_only_for_matching_distributions() -> None:
    target = torch.softmax(torch.randn(4, 6), dim=-1)
    assert selector_kl_loss(target, target.log()).abs() < 1e-6
    assert selector_kl_loss(target, torch.full_like(target, 1 / 6).log()) > 0


def test_selector_overfits_twenty_synthetic_queries() -> None:
    torch.manual_seed(0)
    dim = 32
    queries = F.normalize(torch.randn(20, dim), dim=-1)
    target = oracle_importance(
        queries,
        F.normalize(torch.randn(20, dim), dim=-1),
        F.normalize(torch.randn(20, dim), dim=-1),
        temperature=0.05,
    )
    selector = LinearSelector(dim)
    optimizer = torch.optim.Adam(selector.parameters(), lr=0.05)
    initial = selector_kl_loss(target, selector(queries)).item()
    for _ in range(300):
        optimizer.zero_grad()
        loss = selector_kl_loss(target, selector(queries))
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.1 * initial
