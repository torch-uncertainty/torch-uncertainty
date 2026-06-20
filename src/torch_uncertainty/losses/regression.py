from typing import Literal

import torch
from torch import Tensor, nn
from torch.distributions import Distribution, Independent

from torch_uncertainty.utils.distributions import NormalInverseGamma


class DistributionNLLLoss(nn.Module):
    def __init__(self, reduction: Literal["mean", "sum"] | None = "mean") -> None:
        r"""Negative Log-Likelihood loss for probabilistic regression.

        Given a predictive distribution :math:`p_\theta(y \mid x)` and a target
        :math:`y`, the per-sample loss is

        .. math::
            \mathcal{L}_i = -\log p_\theta(y_i \mid x_i),

        reduced over the batch according to :attr:`reduction`. Positions flagged by
        :attr:`padding_mask` are excluded from the reduction (``nan``-safe).

        Args:
            reduction: Specifies the reduction to apply to the output.
                Must be one of ``'none'``, ``'mean'`` or ``'sum'``. Defaults to ``"mean"``.
        """
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        dist: Distribution,
        targets: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute the NLL of the targets given predicted distributions.

        Args:
            dist: The predicted distributions.
            targets: The target values.
            padding_mask: The padding mask. Sets the loss to ``0`` for padded values.
                Defaults to ``None``.
        """
        loss = -dist.log_prob(targets)
        if padding_mask is not None:
            loss = loss.masked_fill(padding_mask, float("nan"))

        if self.reduction == "mean":
            loss = loss.nanmean()
        elif self.reduction == "sum":
            loss = loss.nansum()
        return loss


class DERLoss(DistributionNLLLoss):
    def __init__(self, reg_weight: float, reduction: str | None = "mean") -> None:
        r"""The Deep Evidential Regression (DER) loss.

        Combines the negative log-likelihood of a Normal-Inverse-Gamma (NIG) predictive
        distribution with a regulariser that penalises evidence on incorrect predictions:

        .. math::
            \mathcal{L}(\boldsymbol{\theta}, y) = -\log p_\text{NIG}(y \mid \boldsymbol{\theta})
            + \lambda \, |y - \mu| \, (2\lambda_\text{NIG} + \alpha),

        where :math:`\boldsymbol{\theta} = (\mu, \lambda_\text{NIG}, \alpha, \beta)` are
        the NIG parameters predicted by the model and :math:`\lambda` is
        :attr:`reg_weight`. The regulariser shrinks the *virtual observation count*
        :math:`2\lambda_\text{NIG} + \alpha` whenever the prediction is wrong, thereby
        increasing the predictive variance.

        Args:
            reg_weight: The weight :math:`\lambda` of the regularization term.
            reduction: Specifies the reduction to apply to the output.
                Must be one of ``'none'``, ``'mean'`` or ``'sum'``.

        References:
            [1] `Amini, A., Schwarting, W., Soleimany, A., & Rus, D. (2020). Deep evidential
            regression. NeurIPS 2020 <https://arxiv.org/abs/1910.02600>`_.
        """
        super().__init__(reduction=None)

        if reduction not in ("none", "mean", "sum") and reduction is not None:
            raise ValueError(f"{reduction} is not a valid value for reduction.")
        self.der_reduction = reduction

        if reg_weight < 0:
            raise ValueError(
                f"The regularization weight should be non-negative, but got {reg_weight}."
            )
        self.reg_weight = reg_weight

    def _reg(self, dist: NormalInverseGamma | Independent, targets: Tensor) -> Tensor:
        if isinstance(dist, Independent):
            dist = dist.base_dist

        return torch.norm(targets - dist.loc, 1, dim=1, keepdim=True) * (
            2 * dist.lmbda + dist.alpha
        )

    def forward(
        self,
        dist: Distribution,
        targets: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        if not isinstance(dist, NormalInverseGamma | Independent):  # coverage: ignore
            raise TypeError(
                f"DER only works for NormalInverseGamma or Independent[NormalInverseGamma] distributions. Got {type(dist)} instead."
            )
        loss_nll = super().forward(dist, targets, padding_mask=padding_mask)
        loss_reg = self._reg(dist, targets)
        loss = loss_nll + self.reg_weight * loss_reg

        if self.der_reduction == "mean":
            return loss.mean()
        if self.der_reduction == "sum":
            return loss.sum()
        return loss


class BetaNLL(nn.Module):
    def __init__(self, beta: float = 0.5, reduction: str | None = "mean") -> None:
        r"""The :math:`\beta`-Negative Log-Likelihood loss (Seitzer et al., 2022).

        A re-weighted version of the Gaussian NLL that scales each per-sample loss by
        the (stop-gradient) predicted variance raised to the power :math:`\beta`:

        .. math::
            \mathcal{L}_i = \sigma_i^{2\beta} \cdot
            \left( \frac{(y_i - \mu_i)^2}{2 \sigma_i^2} + \frac{1}{2} \log \sigma_i^2 \right),

        where :math:`(\mu_i, \sigma_i^2)` are the predicted mean and variance.
        :math:`\beta = 0` recovers the standard Gaussian NLL (which down-weights
        high-variance — i.e. uncertain — samples); :math:`\beta = 1` recovers the MSE
        scaled by :math:`\sigma^2`. Intermediate values interpolate between the two
        regimes and counteract the tendency of Gaussian NLL to neglect noisy targets.

        Args:
            beta: Parameter in :math:`[0, 1]` controlling the relative weighting between
                data points: ``0`` is the standard Gaussian NLL (high weight on
                low-error points); ``1`` recovers a variance-scaled MSE with equal
                weighting.
            reduction: Specifies the reduction to apply to the output.
                Must be one of ``'none'``, ``'mean'`` or ``'sum'``.

        References:
            [1] `Seitzer, M., Tavakoli, A., Antic, D., & Martius, G. (2022). On the pitfalls
            of heteroscedastic uncertainty estimation with probabilistic neural networks. ICLR 2022
            <https://arxiv.org/abs/2203.09168>`_.
        """
        super().__init__()

        if beta < 0 or beta > 1:
            raise ValueError(f"The beta parameter should be in range [0, 1], but got {beta}.")
        self.beta = beta
        self.nll_loss = nn.GaussianNLLLoss(reduction="none")
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"{reduction} is not a valid value for reduction.")
        self.reduction = reduction

    def forward(self, mean: Tensor, targets: Tensor, variance: Tensor) -> Tensor:
        loss = self.nll_loss(mean, targets, variance) * (variance.detach() ** self.beta)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
