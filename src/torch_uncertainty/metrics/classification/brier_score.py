from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat


class BrierScore(Metric):
    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    values: list[Tensor]
    total: Tensor

    def __init__(
        self,
        num_classes: int,
        top_class: bool = False,
        reduction: Literal["mean", "sum", "none"] | None = "mean",
        **kwargs,
    ) -> None:
        r"""Compute the Brier score.

        The Brier Score measures the mean squared difference between predicted
        probabilities and actual target values. It is used to evaluate the
        accuracy of probabilistic predictions, where a lower score indicates
        better calibration and prediction quality.

        Args:
            num_classes (int): Number of classes.
            top_class (bool, optional): If True, computes the Brier score for the
                top predicted class only. Defaults to ``False``.
            reduction (str, optional): Determines how to reduce the score across the
                batch dimension:

                - ``'mean'`` [default]: Averages the score across samples.
                - ``'sum'``: Sums the score across samples.
                - ``'none'`` or ``None``: Returns the score for each sample.

            kwargs: Additional keyword arguments, see `Advanced metric settings
                <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

        Inputs:
            - :attr:`probs`: :math:`(B, C)` or :math:`(B, N, C)`
                Predicted probabilities for each class.
            - :attr:`target`: :math:`(B)` or :math:`(B, C)`
                Ground truth class labels or one-hot encoded targets.

            where:
                :math:`B` is the batch size,
                :math:`C` is the number of classes,
                :math:`N` is the number of estimators.

        Note:
            If :attr:`probs` is a 3D tensor, the metric computes the mean of
            the Brier score over the estimators, as:
            :math:`t = \frac{1}{N} \sum_{i=0}^{N-1} BrierScore(probs[:,i,:], target)`.

        Warning:
            Ensure that the probabilities in :attr:`probs` are normalized to sum
            to one before passing them to the metric.

        Raises:
            ValueError: If :attr:`reduction` is not one of ``'mean'``, ``'sum'``,
                ``'none'`` or ``None``.

        Examples:
            >>> from torch_uncertainty.metrics.classification.brier_score import BrierScore
            # Example 1: Binary Classification
            >>> probs = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
            >>> target = torch.tensor([0, 1])
            >>> metric = BrierScore(num_classes=2)
            >>> metric.update(probs, target)
            >>> score = metric.compute()
            >>> print(score)
            tensor(0.1299)
            # Example 2: Multi-Class Classification
            >>> probs = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
            >>> target = torch.tensor([0, 2])
            >>> metric = BrierScore(num_classes=3, reduction="mean")
            >>> metric.update(probs, target)
            >>> score = metric.compute()
            >>> print(score)
            tensor(0.5199)

        References:
            [1] `Wikipedia entry for the Brier score
            <https://en.wikipedia.org/wiki/Brier_score>`_.
        """
        super().__init__(**kwargs)

        allowed_reduction = ("sum", "mean", "none", None)
        if reduction not in allowed_reduction:
            raise ValueError(
                "Expected argument `reduction` to be one of ",
                f"{allowed_reduction} but got {reduction}",
            )

        self.num_classes = num_classes
        self.top_class = top_class
        self.reduction = reduction
        self.num_estimators = 1

        if self.reduction in ["mean", "sum"]:
            self.add_state("values", default=torch.tensor(0.0), dist_reduce_fx="sum")
        else:
            self.add_state("values", default=[], dist_reduce_fx="cat")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, probs: Tensor, target: Tensor) -> None:
        """Update the current Brier score with a new tensor of probabilities.

        Args:
            probs (Tensor): A probability tensor of shape
                (batch, num_estimators, num_classes) or
                (batch, num_classes)
            target (Tensor): A tensor of ground truth labels of shape
                (batch, num_classes) or (batch)
        """
        if target.ndim == 1 and self.num_classes > 1:
            target = F.one_hot(target, self.num_classes)

        if probs.ndim <= 2:
            batch_size = probs.size(0)
        elif probs.ndim == 3:
            batch_size = probs.size(0)
            self.num_estimators = probs.size(1)
            target = target.unsqueeze(1).repeat(1, self.num_estimators, 1)
        else:
            raise ValueError(
                f"Expected `probs` to be of shape (batch, num_classes) or "
                f"(batch, num_estimators, num_classes) but got {probs.shape}"
            )

        if self.top_class:
            probs, indices = probs.max(dim=-1)
            target = target.gather(-1, indices.unsqueeze(-1)).squeeze(-1)
            brier_score = F.mse_loss(probs, target, reduction="none")
        else:
            brier_score = F.mse_loss(probs, target, reduction="none").sum(dim=-1)

        if self.reduction is None or self.reduction == "none":
            self.values.append(brier_score)
        else:
            self.values += brier_score.sum()
            self.total += batch_size

    def compute(self) -> Tensor:
        """Compute the final Brier score based on inputs passed to ``update``.

        Returns:
            Tensor: The final value(s) for the Brier score
        """
        values = dim_zero_cat(self.values)
        if self.reduction == "sum":
            return values.sum(dim=-1) / self.num_estimators
        if self.reduction == "mean":
            return values.sum(dim=-1) / self.total / self.num_estimators
        return values
