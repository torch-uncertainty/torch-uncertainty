import math

import torch
from torch import Tensor, nn


class PinballLoss(nn.Module):
    def __init__(self, quantile: float, reduction: str | None = "mean") -> None:
        r"""The Pinball loss for quantile regression.

        Also known as the quantile loss or check loss, the pinball loss at
        quantile level :math:`\tau \in (0, 1)` is:

        .. math::
            \mathcal{L}_\tau(y, \hat{y}) =
            \max\!\left(\tau\,(y - \hat{y}),\,(\tau - 1)\,(y - \hat{y})\right)
            = \begin{cases}
                \tau\,(y - \hat{y}) & \text{if } y \geq \hat{y}, \\
                (1 - \tau)\,(\hat{y} - y) & \text{if } y < \hat{y}.
            \end{cases}

        For :math:`\tau = 0.5` the loss coincides with the mean absolute error
        (MAE) scaled by :math:`\tfrac{1}{2}`.

        Args:
            quantile: The quantile level :math:`\tau \in (0, 1)`.
            reduction: Specifies the reduction to apply to the output.
                Must be one of ``'none'``, ``'mean'`` or ``'sum'``. Defaults
                to ``"mean"``.

        References:
            [1] `Koenker, R., & Bassett Jr, G. (1978). Regression quantiles.
            Econometrica, <https://www.jstor.org/stable/1913643>`_.
        """
        super().__init__()

        if not 0 < quantile < 1:
            raise ValueError(f"The quantile parameter should be in (0, 1), but got {quantile}.")
        self.quantile = quantile

        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"{reduction} is not a valid value for reduction.")
        self.reduction = reduction

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute the pinball loss.

        Args:
            predictions: The predicted quantile values.
            targets: The target values.
        """
        residual = targets - predictions
        loss = torch.maximum(self.quantile * residual, (self.quantile - 1) * residual)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class RQRLoss(nn.Module):
    def __init__(
        self,
        coverage_level: float,
        width_weight: float | None = None,
        reduction: str | None = "mean",
    ) -> None:
        r"""The Relaxed Quantile Regression (RQR) loss.

        RQR directly learns the two endpoints :math:`(\mu_1, \mu_2)` of a
        prediction interval with target coverage :math:`\alpha`, without assigning
        either endpoint to a predefined quantile. Let

        .. math::
            \kappa = (y - \mu_1)(y - \mu_2).

        Since :math:`\kappa < 0` exactly when :math:`y` is inside the interval, the
        unregularized loss is

        .. math::
            \mathcal{L}^{\mathrm{RQR}}_\alpha =
            \begin{cases}
                \alpha \kappa & \text{if } \kappa \geq 0, \\
                (\alpha - 1)\kappa & \text{if } \kappa < 0.
            \end{cases}

        When :attr:`width_weight` is positive, this implements the width-minimizing
        RQR-W variant. Its squared-width penalty biases coverage by
        :math:`-2\lambda`; therefore, the loss uses the corrected level
        :math:`\hat{\alpha} = \alpha + 2\lambda`:

        .. math::
            \mathcal{L}^{\mathrm{RQR-W}}_\alpha =
            \mathcal{L}^{\mathrm{RQR}}_{\alpha + 2\lambda}
            + \frac{\lambda}{2}(\mu_2-\mu_1)^2.

        Args:
            coverage_level: The target coverage level :math:`\alpha \in (0, 1)`.
            width_weight: The width regularization weight :math:`\lambda`, which
                must satisfy :math:`0 \leq \lambda \leq (1-\alpha)/2`. A value of
                ``0`` recovers the unregularized RQR loss. Defaults to ``0.0``.
            reduction: Specifies the reduction to apply to the output.
                Must be one of ``'none'``, ``'mean'`` or ``'sum'``. Defaults to
                ``"mean"``.

        References:
            [1] `Pouplin, T., Jeffares, A., Seedat, N., & van der Schaar, M. (2024).
            Relaxed quantile regression: Prediction intervals for asymmetric noise.
            ICML 2024 <https://arxiv.org/abs/2406.03258>`_.
        """
        super().__init__()

        if not 0 < coverage_level < 1:
            raise ValueError(f"The coverage level should be in (0, 1), but got {coverage_level}.")
        self.coverage_level = coverage_level

        if width_weight is None:
            width_weight = 0.0
        corrected_level = coverage_level + 2 * width_weight
        if width_weight < 0 or not math.isfinite(width_weight) or corrected_level > 1:
            raise ValueError(
                "The width weight should be in "
                f"[0, (1 - coverage_level) / 2], but got {width_weight}."
            )
        self.width_weight = width_weight

        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"{reduction} is not a valid value for reduction.")
        self.reduction = reduction

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute the RQR loss.

        Args:
            predictions: The two interval endpoints, with shape ``(..., 2)``.
            targets: The target values, with shape ``(...)``.
        """
        if predictions.ndim == 0 or predictions.shape[-1] != 2:
            raise ValueError(
                "Expected `predictions` to have exactly two interval endpoints "
                f"in its last dimension, but got shape {predictions.shape}."
            )
        if targets.shape != predictions.shape[:-1]:
            raise ValueError(
                "Expected `targets` to have the same shape as `predictions` "
                f"without its last dimension, but got {targets.shape=} and "
                f"{predictions.shape=}."
            )

        endpoint_1, endpoint_2 = predictions.unbind(dim=-1)
        interval_product = (targets - endpoint_1) * (targets - endpoint_2)
        corrected_level = self.coverage_level + 2 * self.width_weight
        loss = torch.maximum(
            corrected_level * interval_product,
            (corrected_level - 1) * interval_product,
        )
        loss += self.width_weight * (endpoint_2 - endpoint_1).square() / 2

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
