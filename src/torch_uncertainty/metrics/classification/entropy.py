from typing import Any, Literal

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat


class Entropy(Metric):
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
        r"""The Shannon Entropy Metric to estimate the confidence of a single model
        or the mean confidence across estimators.

        Args:
            reduction (str, optional): Determines how to reduce over the
                :math:`B`/batch dimension:

                - ``'mean'`` [default]: Averages score across samples
                - ``'sum'``: Sum score across samples
                - ``'none'`` or ``None``: Returns score per sample

            kwargs: Additional keyword arguments, see `Advanced metric settings
                <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

        Inputs:
            - ``probs``: :math:`(B, C)` or :math:`(B, N, C)`

            where :math:`B` is the batch size, :math:`C` is the number of classes
            and :math:`N` is the number of estimators.

        Note:
            A higher entropy means a lower confidence.

        Raises:
            ValueError:
                If :attr:`reduction` is not one of ``'mean'``, ``'sum'``,
                ``'none'`` or ``None``.

        Example:

        .. code-block:: python

            from torch_uncertainty.metrics.classification import Entropy

            probs = torch.tensor(
                [
                    [[0.7, 0.3], [0.6, 0.4], [0.8, 0.2]],  # Example 1, 3 estimators
                    [[0.4, 0.6], [0.5, 0.5], [0.3, 0.7]],  # Example 2, 3 estimators
                ]
            )
            metric = Entropy(reduction="mean")
            metric.update(probs)
            result = metric.compute()
            print(result)  # Mean entropy value across samples
            # tensor(0.6269)

            # Using single-estimator probabilities
            probs = torch.tensor(
                [
                    [0.7, 0.3],  # Example 1
                    [0.4, 0.6],  # Example 2
                ]
            )
            metric = Entropy(reduction=None)
            metric.update(probs)
            result = metric.compute()
            print(result)  # Per-sample entropy values
            # tensor([0.6109, 0.6730])
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
        """Update the current entropy with a new tensor of probabilities.

        Args:
            probs (torch.Tensor): Probabilities from the model.
        """
        batch_size = probs.size(0)
        entropy = torch.special.entr(probs).sum(dim=-1)

        if len(probs.shape) == 3:
            entropy = entropy.mean(dim=1)

        if self.reduction is None or self.reduction == "none":
            self.values.append(entropy)
        else:
            self.values += entropy.sum()
            self.total += batch_size

    def compute(self) -> torch.Tensor:
        """Computes Entropy based on inputs passed in to ``update``
        previously.
        """
        values = dim_zero_cat(self.values)
        if self.reduction == "sum":
            return values.sum(dim=-1)
        if self.reduction == "mean":
            return values.sum(dim=-1) / self.total
        # reduction is None or "none"
        return values
