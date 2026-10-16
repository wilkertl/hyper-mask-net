"""Relaxed top-k mask for retrieval training (F5 research D3)."""

import math

import pytest
import torch

from hyperdime.selection.relaxed import soft_mass_on_topk, soft_top_k
from hyperdime.selection.topk import top_k_mask

SCORES = torch.tensor([[3.0, 1.0, 2.0, 0.0]])
# Standardized (population std): mean 1.5, std sqrt(5)/2, so z = [3, -1, 1, -3] / sqrt(5); the
# midpoint between the 2nd and 3rd largest z is 0.
Z = torch.tensor([[3.0, -1.0, 1.0, -3.0]]) / math.sqrt(5)


def test_the_mask_is_a_sigmoid_around_the_k_boundary_by_hand() -> None:
    assert torch.allclose(soft_top_k(SCORES, 2, beta=1.0), torch.sigmoid(Z), atol=1e-6)


def test_above_one_half_is_exactly_the_hard_top_k() -> None:
    mask = soft_top_k(SCORES, 2, beta=0.7)
    assert torch.equal(mask > 0.5, top_k_mask(SCORES, 2))


def test_a_sharp_mask_equals_the_hard_top_k() -> None:
    scores = torch.randn(5, 64, generator=torch.Generator().manual_seed(0))
    hard = top_k_mask(scores, 10).float()
    assert torch.allclose(soft_top_k(scores, 10, beta=1e-4), hard, atol=1e-6)


def test_keeping_every_dimension_is_the_full_query() -> None:
    assert torch.equal(soft_top_k(SCORES, 4, beta=0.1), torch.ones(1, 4))


def test_the_mask_ignores_the_scale_and_offset_of_the_scores() -> None:
    assert torch.allclose(soft_top_k(SCORES, 2, 0.5), soft_top_k(10 * SCORES - 7, 2, 0.5))


def test_the_threshold_carries_no_gradient() -> None:
    scores = SCORES.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(soft_top_k(scores, 2, beta=1.0).sum(), scores)
    free = scores.detach().clone().requires_grad_(True)
    z = (free - free.mean(-1, keepdim=True)) / free.std(-1, correction=0, keepdim=True)
    (expected,) = torch.autograd.grad(torch.sigmoid(z - 0.0).sum(), free)
    assert torch.allclose(grad, expected, atol=1e-6)


def test_the_soft_mass_on_the_top_k_by_hand() -> None:
    mask = torch.tensor([[0.9, 0.1, 0.6, 0.4]])
    assert soft_mass_on_topk(mask, SCORES, 2).tolist() == pytest.approx([1.5 / 2.0])
