from typing import Literal

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat


class VariationRatio(Metric):
    full_state_update = False
    is_differentiable = True
    higher_is_better = False

    probs: list[Tensor]

    def __init__(
        self,
        probabilistic: bool = True,
        reduction: Literal["mean", "sum", "none"] | None = "mean",
        **kwargs,
    ) -> None:
        r"""Compute the Variation Ratio.

        The Variation Ratio is a measure of the uncertainty or disagreement among
        predictions from multiple estimators. It is defined as the proportion of
        predicted class labels that are not the chosen (most frequent) class.

        Args:
            probabilistic (bool, optional): Whether to use probabilistic predictions. Defaults to True.
            reduction (Literal["mean", "sum", "none", None], optional): Determines how to reduce over the batch dimension:

                - ``'mean'`` [default]: Averages score across samples
                - ``'sum'``: Sum score across samples
                - ``'none'`` or ``None``: Returns score per sample

            kwargs: Additional keyword arguments, see `Advanced metric settings <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

        Inputs:
            - :attr:`probs`: :math:`(B, N, C)`

              where :math:`B` is the batch size, :math:`C` is the number of classes
              and :math:`N` is the number of estimators.

        Note:
            A higher variation ratio indicates higher uncertainty or disagreement
            among the estimators.

        Warning:
            Metric `VariationRatio` will save all predictions in buffer. For large
            datasets this may lead to large memory footprint.

        Raises:
            ValueError:
                If :attr:`reduction` is not one of ``'mean'``, ``'sum'``,
                ``'none'`` or ``None``.

        Example:

        .. code-block:: python

            from torch_uncertainty.metrics.classification import VariationRatio

            probs = torch.tensor(
                [
                    [[0.7, 0.3], [0.6, 0.4], [0.8, 0.2]],  # Example 1, 3 estimators
                    [[0.4, 0.6], [0.5, 0.5], [0.3, 0.7]],  # Example 2, 3 estimators
                ]
            )

            vr = VariationRatio(probabilistic=True, reduction="mean")
            vr.update(probs)
            result = vr.compute()
            print(result)
            # output: tensor(0.4500)
        """
        super().__init__(**kwargs)
        allowed_reduction = ("sum", "mean", "none", None)
        if reduction not in allowed_reduction:
            raise ValueError(
                "Expected argument `reduction` to be one of ",
                f"{allowed_reduction} but got {reduction}",
            )

        self.probabilistic = probabilistic
        self.reduction = reduction

        self.add_state("probs", [], dist_reduce_fx="cat")

        rank_zero_warn(
            "Metric `VariationRatio` will save all predictions in buffer. For "
            "large datasets this may lead to large memory footprint."
        )

    def update(self, probs: Tensor) -> None:
        # store data as (example, estimator, class)
        self.probs.append(probs.transpose(0, 1))

    def compute(self) -> Tensor:
        r"""Computes the variation ratio which amounts to the proportion of
        predicted class labels which are not the chosen class.

        Returns:
            Tensor: Mean disagreement between estimators.
        """
        probs_per_est = dim_zero_cat(self.probs)
        n_estimators = probs_per_est.shape[1]
        probs = probs_per_est.mean(dim=1)

        # best class for example
        max_classes = probs.argmax(dim=-1)

        if self.probabilistic:
            probs_per_est = probs_per_est.permute((0, 2, 1))
            variation_ratio = 1 - probs_per_est[
                torch.arange(probs_per_est.size(0)), max_classes
            ].mean(dim=1)
        else:
            # best class for (example, estimator)
            max_classes_per_est = probs_per_est.argmax(dim=-1)
            variation_ratio = (
                1
                - torch.sum(max_classes_per_est == max_classes.unsqueeze(1), dim=-1) / n_estimators
            )

        if self.reduction == "mean":
            variation_ratio = variation_ratio.mean()
        elif self.reduction == "sum":
            variation_ratio = variation_ratio.sum()
        else:  # None
            pass

        return variation_ratio
