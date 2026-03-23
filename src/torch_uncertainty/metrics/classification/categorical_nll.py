from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat


class CategoricalNLL(Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    values: list[Tensor]
    total: Tensor

    def __init__(
        self,
        reduction: Literal["mean", "sum", "none"] | None = "mean",
        **kwargs: Any,
    ) -> None:
        r"""Computes the Negative Log-Likelihood (NLL) metric for classification tasks.

        This metric evaluates the performance of a probabilistic classification model by
        calculating the negative log likelihood of the predicted probabilities. For a batch
        of size :math:`B` with :math:`C` classes, the negative log likelihood is defined as:

        .. math::

            \ell(p, y) = -\frac{1}{B} \sum_{i=1}^B \log(p_{i, y_i})

        where :math:`p_{i, y_i}` is the predicted probability for the true class :math:`y_i`
        of sample :math:`i`.

        Args:
            reduction (str, optional): Determines how to reduce the computed loss over
                the batch dimension:

                - ``'mean'`` [default]: Averages the loss across samples in the batch.
                - ``'sum'``: Sums the loss across samples in the batch.
                - ``'none'`` or ``None``: Returns the loss for each sample without reducing.

            kwargs: Additional keyword arguments as described in `Advanced Metric Settings <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

        Inputs:
            - :attr:`probs`: :math:`(B, C)`
                A Tensor containing the predicted probabilities for `C` classes, where each
                row corresponds to a sample in the batch.
            - :attr:`target`: :math:`(B,)`
                A Tensor containing the ground truth labels as integers in the range :math:`[0, C-1]`.

        Note:
            Ensure that the probabilities in :attr:`probs` are normalized to sum to one:

            .. math::

                \sum_{c=1}^C p_{i, c} = 1 \quad \forall i \in [1, B].

        Warning:
            If `reduction` is not one of ``'mean'``, ``'sum'``, ``'none'``, or ``None``, a
            :class:`ValueError` will be raised.

        Example:

        .. code-block:: python

            from torch_uncertainty.metrics.classification.categorical_nll import (
                CategoricalNLL,
            )

            metric = CategoricalNLL(reduction="mean")
            probs = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
            target = torch.tensor([0, 1])
            metric.update(probs, target)
            print(metric.compute())
            # Output: tensor(0.4338)
        """
        super().__init__(**kwargs)

        allowed_reduction = ("sum", "mean", "none", None)
        if reduction not in allowed_reduction:
            raise ValueError(
                f"Expected argument `reduction` to be one of {allowed_reduction} "
                f"but got {reduction}"
            )

        self.reduction = reduction

        if self.reduction in ["mean", "sum"]:
            self.add_state(
                "values",
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )
        else:
            self.add_state("values", default=[], dist_reduce_fx="cat")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, probs: Tensor, target: Tensor) -> None:
        r"""Update state with prediction probabilities and targets.

        Args:
            probs (Tensor): Probabilities from the model.
            target (Tensor): Ground truth labels.

        For each sample :math:`i`, the negative log likelihood is computed as:

        .. math::
            \ell_i = -\log(p_{i, y_i}),

        where :math:`p_{i, y_i}` is the predicted probability for the true class :math:`y_i`.
        """
        if probs.ndim == 1:
            probs = torch.stack([1 - probs, probs], dim=-1)

        if self.reduction is None or self.reduction == "none":
            self.values.append(F.nll_loss(torch.log(probs), target, reduction="none"))
        else:
            self.values += F.nll_loss(torch.log(probs), target, reduction="sum")
            self.total += target.size(0)

    def compute(self) -> Tensor:
        """Computes the final NLL score based on the accumulated state.

        Returns:
            Tensor: A scalar if `reduction` is `'mean'` or `'sum'`; otherwise, a tensor
            of shape :math:`(B,)` if `reduction` is `'none'`.
        """
        values = dim_zero_cat(self.values)

        if self.reduction == "sum":
            return values.sum(dim=-1)
        if self.reduction == "mean":
            return values.sum(dim=-1) / self.total
        # reduction is None or "none"
        return values
