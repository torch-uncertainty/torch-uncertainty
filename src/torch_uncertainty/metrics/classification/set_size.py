from typing import Literal

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.utilities.data import dim_zero_cat


class SetSize(Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    sizes: list[Tensor] | Tensor
    total: Tensor

    def __init__(
        self,
        reduction: Literal["mean", "sum", "none"] | None = "mean",
        **kwargs,
    ) -> None:
        """Set size to compute the efficiency of conformal prediction methods.

        Args:
            reduction (str, optional): Determines how to reduce over the
                :math:`B`/batch dimension:

                - ``'mean'`` [default]: Averages score across samples
                - ``'sum'``: Sum score across samples
                - ``'none'`` or ``None``: Returns score per sample

            kwargs: Additional keyword arguments, see `Advanced metric settings
                <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.
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
            self.add_state("sizes", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        else:
            self.add_state("sizes", default=[], dist_reduce_fx="cat")
        self.add_state("total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor | None = None) -> None:
        """Update the metric state with predictions and targets.

        Args:
            preds (torch.Tensor): predicted sets tensor of shape (B, C), where B is the batch size
                and C is the number of classes.
            targets (torch.Tensor): For API consistency
        """
        batch_size = preds.size(0)
        pred_sizes = preds.bool().sum(-1)

        if self.reduction is None or self.reduction == "none":
            self.sizes.append(pred_sizes)
        else:
            self.sizes += pred_sizes.sum()
            self.total += batch_size

    def compute(self) -> Tensor:
        """Compute the mean set size.

        Returns:
            Tensor: The coverage rate.
        """
        values = dim_zero_cat(self.sizes)
        if self.reduction == "sum":
            return values
        if self.reduction == "mean":
            return _safe_divide(values, self.total)
        return values
