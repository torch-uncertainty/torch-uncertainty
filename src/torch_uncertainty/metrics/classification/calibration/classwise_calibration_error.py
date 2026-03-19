from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat

from .calibration_error import TUBinaryCalibrationError


class ClasswiseCalibrationError(Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    probs: list[Tensor]
    targets: list[Tensor]

    def __init__(
        self,
        num_classes: int,
        num_bins: int = 15,
        norm: Literal["l1", "l2", "max"] = "l1",
        reduction: Literal["mean", "sum", "none"] | None = "mean",
        **kwargs,
    ) -> None:
        r"""Compute the Classwise Expected Calibration Error (ECE).

        The Classwise ECE measures the expected calibration error for each class
        independently in a one-vs-all manner, and then reduces the scores.
        It is used to evaluate the calibration of individual classes, where a
        lower score indicates better calibration quality.

        Args:
            num_classes (int): Number of classes.
            num_bins (int, optional): Number of calibration bins. Defaults to ``15``.
            norm (Literal["l1", "l2", "max"]): Norm used to compute the ECE (e.g., ``'l1'``,
                ``'l2'``, ``'max'``). Defaults to ``'l1'``.
            reduction (Literal["mean", "sum", "none"] | None): Determines how to reduce the score across the
                classes:

                - ``'mean'`` [default]: Averages the ECE across classes.
                - ``'sum'``: Sums the ECE across classes.
                - ``'none'`` or ``None``: Returns the ECE for each class.

            kwargs: Additional keyword arguments, see `Advanced metric settings
                <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

        Inputs:
            - :attr:`probs`: :math:`(B, C)`
                Predicted probabilities for each class.
            - :attr:`target`: :math:`(B)` or :math:`(B, C)`
                Ground truth class labels or one-hot encoded targets.

            where:
                :math:`B` is the batch size,
                :math:`C` is the number of classes.

        Warning:
            Ensure that the probabilities in :attr:`probs` are normalized to sum
            to one before passing them to the metric.

        Raises:
            ValueError: If :attr:`reduction` is not one of ``'mean'``, ``'sum'``,
                ``'none'`` or ``None``.
            ValueError: If :attr:`norm` is not one of ``'l1'``, ``'l2'``, or ``'max'``.

        References:
            [1] `Kull et al. Beyond temperature scaling: Obtaining well-calibrated multiclass probabilities with Dirichlet calibration. In NeurIPS, 2019
            <https://arxiv.org/abs/1910.12656>`_.

        Examples:
            >>> from torch_uncertainty.metrics.classification import ClasswiseECE
            # Example: Multi-Class Classification
            >>> probs = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
            >>> target = torch.tensor([0, 2])
            >>> metric = ClasswiseECE(num_classes=3, reduction="mean")
            >>> metric.update(probs, target)
            >>> score = metric.compute()
            >>> print(score)
            tensor(...)
        """
        super().__init__(**kwargs)

        allowed_reduction = ("sum", "mean", "none", None)
        if reduction not in allowed_reduction:
            raise ValueError(
                "Expected argument `reduction` to be one of ",
                f"{allowed_reduction} but got {reduction}",
            )

        allowed_norm = ("l1", "l2", "max")
        if norm not in allowed_norm:
            raise ValueError(
                "Expected argument `norm` to be one of ",
                f"{allowed_norm} but got {norm}",
            )

        self.num_classes = num_classes
        self.num_bins = num_bins
        self.norm = norm
        self.reduction = reduction

        # Store all probabilities and targets to compute binning at the compute step
        self.add_state("probs", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, probs: Tensor, target: Tensor) -> None:
        """Update the state with a new tensor of probabilities.

        Args:
            probs (Tensor): A probability tensor of shape (batch, num_classes).
            target (Tensor): A tensor of ground truth labels of shape
                (batch, num_classes) or (batch).
        """
        if target.ndim == 1 and self.num_classes > 1:
            target = F.one_hot(target, self.num_classes)

        if probs.ndim != 2:
            raise ValueError(
                f"Expected `probs` to be of shape (batch, num_classes) but got {probs.shape}"
            )

        self.probs.append(probs)
        self.targets.append(target)

    def compute(self) -> Tensor:
        """Compute the final Classwise ECE based on inputs passed to ``update``.

        Returns:
            Tensor: The final value(s) for the Classwise ECE.
        """
        probs = dim_zero_cat(self.probs)
        targets = dim_zero_cat(self.targets)

        metric = TUBinaryCalibrationError(
            n_bins=self.num_bins,
            norm=self.norm,
        )
        calibration_errors = []
        for c in range(self.num_classes):
            class_probs = probs[..., c]
            class_targets = targets[..., c]
            calibration_errors.append(metric(class_probs, class_targets))

        calibration_errors = torch.stack(calibration_errors)
        if self.reduction == "sum":
            return calibration_errors.sum(dim=-1)
        if self.reduction == "mean":
            return calibration_errors.mean(dim=-1)
        return calibration_errors
