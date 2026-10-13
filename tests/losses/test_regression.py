import math

import pytest
import torch
from torch.distributions import Independent, Normal

from torch_uncertainty.losses import (
    BetaNLL,
    DERLoss,
    DistributionNLLLoss,
)
from torch_uncertainty.utils.distributions import NormalInverseGamma


class TestDistributionNLL:
    """Testing the DistributionNLLLoss class."""

    def test_sum(self) -> None:
        loss = DistributionNLLLoss(reduction="sum")
        dist = Normal(0, 1)
        loss(dist, torch.tensor([0.0]))


class TestDERLoss:
    """Testing the DERLoss class."""

    def test_main(self) -> None:
        loss = DERLoss(reg_weight=1e-2)
        layer = NormalInverseGamma
        inputs = layer(torch.ones(1), torch.ones(1), torch.ones(1), torch.ones(1))
        targets = torch.tensor([[1.0]], dtype=torch.float32)

        assert loss(inputs, targets) == pytest.approx(2 * math.log(2))

        loss = DERLoss(
            reg_weight=1e-2,
            reduction="sum",
        )
        inputs = layer(
            torch.ones((2, 1)),
            torch.ones((2, 1)),
            torch.ones((2, 1)),
            torch.ones((2, 1)),
        )

        inputs = Independent(inputs, 0)

        assert loss(
            inputs,
            targets,
        ) == pytest.approx(4 * math.log(2))

        loss = DERLoss(
            reg_weight=1e-2,
            reduction="none",
        )

        assert loss(
            inputs,
            targets,
        ) == pytest.approx([2 * math.log(2), 2 * math.log(2)])

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    @pytest.mark.parametrize("independent", [False, True])
    @pytest.mark.parametrize("mask_kind", ["partial", "none", "all"])
    def test_masked_reduction(self, reduction: str, independent: bool, mask_kind: str) -> None:
        loc = torch.tensor([[0.2], [-0.5], [1.0]], dtype=torch.float64, requires_grad=True)
        lmbda = torch.full_like(loc, 2.0, requires_grad=True)
        alpha = torch.full_like(loc, 3.0, requires_grad=True)
        beta = torch.full_like(loc, 1.5, requires_grad=True)
        base = NormalInverseGamma(loc, lmbda, alpha, beta)
        dist = Independent(base, 0) if independent else base
        targets = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
        mask = torch.tensor([[False], [True], [False]])
        if mask_kind != "partial":
            mask = torch.full_like(mask, mask_kind == "all")
        actual = DERLoss(reg_weight=0.1, reduction=reduction)(dist, targets, ignore_mask=mask)
        per_item = -base.log_prob(targets) + 0.1 * (targets - loc).abs() * (2 * lmbda + alpha)
        retained = per_item[~mask]
        expected = retained.mean() if reduction == "mean" else retained.sum()
        torch.testing.assert_close(actual, expected, equal_nan=True)
        if mask_kind != "all":
            parameters = (loc, lmbda, alpha, beta)
            actual_grads = torch.autograd.grad(actual, parameters, retain_graph=True)
            expected_grads = torch.autograd.grad(expected, parameters)
            for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
                torch.testing.assert_close(actual_grad, expected_grad)
                torch.testing.assert_close(actual_grad[mask], torch.zeros_like(actual_grad[mask]))

    def test_failures(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"The regularization weight should be non-negative, but got ",
        ):
            DERLoss(reg_weight=-1)

        with pytest.raises(ValueError, match=r"is not a valid value for reduction."):
            DERLoss(reg_weight=1.0, reduction="median")


class TestBetaNLL:
    """Testing the BetaNLL class."""

    def test_main(self) -> None:
        loss = BetaNLL(beta=0.5)

        inputs = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        targets = torch.tensor([[1.0]], dtype=torch.float32)

        assert loss(*inputs.split(1, dim=-1), targets) == 0

        loss = BetaNLL(
            beta=0.5,
            reduction="sum",
        )

        assert (
            loss(
                *inputs.repeat(2, 1).split(1, dim=-1),
                targets.repeat(2, 1),
            )
            == 0
        )

        loss = BetaNLL(
            beta=0.5,
            reduction="none",
        )

        assert loss(
            *inputs.repeat(2, 1).split(1, dim=-1),
            targets.repeat(2, 1),
        ) == pytest.approx([0.0, 0.0])

    def test_failures(self) -> None:
        with pytest.raises(ValueError, match=r"The beta parameter should be in range "):
            BetaNLL(beta=-1)

        with pytest.raises(ValueError, match=r"is not a valid value for reduction."):
            BetaNLL(beta=1.0, reduction="median")
