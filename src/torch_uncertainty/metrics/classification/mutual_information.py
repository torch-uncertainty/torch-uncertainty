from typing import Any, Literal

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat


class MutualInformation(Metric):
    is_differentiable = False
    higher_is_better = None
    full_state_update = False

    values: list[Tensor]
    total: Tensor

    def __init__(
        self,
        reduction: Literal["mean", "sum", "none"] | None = "mean",
        **kwargs: Any,
    ) -> None:
        r"""Compute the Mutual Information Metric.

        The Mutual Information Metric estimates the epistemic uncertainty of an
        ensemble of estimators.

        Args:
            reduction (str, optional): Determines how to reduce over the :math:`B`/batch dimension:

                - ``'mean'`` [default]: Averages score across samples
                - ``'sum'``: Sum score across samples
                - ``'none'`` or ``None``: Returns score per sample

            kwargs: Additional keyword arguments, see `Advanced metric settings <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

        Inputs:
            - :attr:`probs`: :math:`(B, N, C)`

              where :math:`B` is the batch size, :math:`N` is the number of
              estimators, and :math:`C` is the number of classes.

        Note:
            A higher mutual information can be interpreted as a higher
            epistemic uncertainty. The Mutual Information is also
            computationally equivalent to the Generalized Jensen-Shannon
            Divergence (GJSD).

            The implementation of the mutual information clamps results to zero
            to avoid negative values that could appear due to numerical
            instabilities.

        Warning:
            Make sure that the probabilities in :attr:`probs` are normalized to
            sum to one.

        Raises:
            ValueError:
                If :attr:`reduction` is not one of ``'mean'``, ``'sum'``,
                ``'none'`` or ``None``.
        """
        super().__init__(**kwargs)

        allowed_reduction = ("sum", "mean", "none", None)
        if reduction not in allowed_reduction:
            raise ValueError(
                "Expected argument `reduction` to be one of ",
                f"{allowed_reduction} but got {reduction}",
            )
        self.reduction = reduction

        if self.reduction in ["mean", "sum"]:
            self.add_state("values", default=torch.tensor(0.0), dist_reduce_fx="sum")
        else:
            self.add_state("values", default=[], dist_reduce_fx="cat")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, probs: torch.Tensor) -> None:
        r"""Update the current mutual information with a new tensor of
        probabilities.

        Args:
            probs (torch.Tensor): Likelihoods from the ensemble of shape
                :math:`(B, N, C)`, where :math:`B` is the batch size,
                :math:`N` is the number of estimators and :math:`C` is the
                number of classes.
        """
        batch_size = probs.size(0)

        ens_probs = probs.mean(dim=1)
        entropy_mean = torch.special.entr(ens_probs).sum(dim=-1)
        mean_entropy = torch.special.entr(probs).sum(dim=-1).mean(dim=1)

        mutual_information = entropy_mean - mean_entropy

        if self.reduction is None or self.reduction == "none":
            self.values.append(mutual_information)
        else:
            self.values += mutual_information.sum()
            self.total += batch_size

    def compute(self) -> torch.Tensor:
        r"""Computes Mutual Information based on inputs passed in to ``update``
        previously.
        """
        values = torch.clamp(dim_zero_cat(self.values), min=0)
        if self.reduction == "sum":
            return values.sum(dim=-1)
        if self.reduction == "mean":
            return values.sum(dim=-1) / self.total
        # reduction is None or "none"
        return values
