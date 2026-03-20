import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat


class ThresholdAccuracy(Metric):
    values: Tensor
    total: Tensor
    def __init__(self, power: int, lmbda: float = 1.25, **kwargs) -> None:
        r"""Computes the Threshold Accuracy metric, also referred to as d1, d2, or d3.

        This metric evaluates the percentage of predictions that fall within a
        specified threshold of their corresponding target values. The threshold
        is determined based on the maximum ratio between predictions and targets
        (or its inverse), raised to a specified power.

        Args:
            power: The power to raise the threshold to. Often in [1, 2, 3].
            lmbda: The threshold to compare the max of ratio of predictions
                to targets and its inverse to. Defaults to ``1.25.``
            kwargs: Additional keyword arguments, see `Advanced metric settings
                <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

        Example:

        .. code-block:: python

            from torch_uncertainty.metrics.regression import ThresholdAccuracy
            import torch

            # Initialize the metric with power=2 and lambda=1.25
            threshold_accuracy = ThresholdAccuracy(power=2, lmbda=1.25)

            # Example predictions and targets
            preds = torch.tensor([2.0, 3.0, 5.0, 8.0, 20.0])
            target = torch.tensor([2.1, 2.5, 4.5, 10.0, 10.0])

            # Update the metric state
            threshold_accuracy.update(preds, target)

            # Compute the Threshold Accuracy
            result = threshold_accuracy.compute()
            print(f"Threshold Accuracy: {result.item():.2f}")
            # Output: Threshold Accuracy: 0.80
        """
        super().__init__(**kwargs)
        if power < 0:
            raise ValueError(f"Power must be greater than or equal to 0. Got {power}.")
        self.power = power
        if lmbda < 1:
            raise ValueError(f"Lambda must be greater than or equal to 1. Got {lmbda}.")
        self.lmbda = lmbda
        self.add_state("values", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        self.values += torch.sum(torch.max(preds / target, target / preds) < self.lmbda**self.power)
        self.total += target.size(0)

    def compute(self) -> Tensor:
        """Compute the Threshold Accuracy."""
        values = dim_zero_cat(self.values)
        return values / self.total
