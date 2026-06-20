from typing import Literal, cast

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
        r"""Average prediction-set size — the standard *efficiency* metric for conformal
        prediction methods.

        For a set-valued predictor :math:`\mathcal{C}(X) \subseteq \{1, \dots, C\}`,

        .. math::
            \text{SetSize} = \frac{1}{N} \sum_{i=1}^{N} |\mathcal{C}(x_i)|.

        Smaller sets are more informative, hence ``higher_is_better = False``. Set size
        is typically reported jointly with the empirical
        :class:`~torch_uncertainty.metrics.classification.CoverageRate`: a useful
        conformal predictor achieves the target coverage with as small a set as
        possible.

        Args:
            reduction: Determines how to reduce over the :math:`B`/batch dimension:

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
            preds: Predicted sets tensor of shape ``(B, C)``, where ``B`` is the
                batch size and ``C`` is the number of classes.
            targets: Unused. Kept for API consistency. Defaults to ``None``.
        """
        batch_size = preds.size(0)
        pred_sizes = preds.bool().sum(-1)

        if self.reduction is None or self.reduction == "none":
            sizes = cast("list[Tensor]", self.sizes)
            sizes.append(pred_sizes)
        else:
            self.sizes += pred_sizes.sum()
            self.total += batch_size

    def compute(self) -> Tensor:
        """Compute the set size.

        Returns:
            Tensor: The set size according to the selected reduction.
        """
        values = dim_zero_cat(self.sizes)
        if self.reduction == "sum":
            return values
        if self.reduction == "mean":
            return _safe_divide(values, self.total)
        return values
