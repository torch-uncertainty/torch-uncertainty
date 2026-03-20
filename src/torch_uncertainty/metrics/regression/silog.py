from typing import Any

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat


class SILog(Metric):
    log_dists: Tensor
    sq_log_dists: Tensor
    def __init__(self, sqrt: bool = False, lmbda: float = 1.0, **kwargs: Any) -> None:
        r"""Computes The Scale-Invariant Logarithmic Loss metric.

        The Scale-Invariant Logarithmic Loss (SILog), a metric designed for depth estimation tasks.

        .. math:: \text{SILog} = \frac{1}{N} \sum_{i=1}^{N} \left(\log(y_i) - \log(\hat{y_i})\right)^2 - \left(\frac{1}{N} \sum_{i=1}^{N} \log(y_i) \right)^2,

        where :math:`N` is the batch size, :math:`y_i` is a tensor of target
        values and :math:`\hat{y_i}` is a tensor of prediction.
        Return the square root of SILog by setting :attr:`sqrt` to `True`.

        This metric evaluates the scale-invariant error between predicted and target values in log-space. It accounts for
        both the variance of the error and the mean log difference between predictions and targets. By setting the
        :attr:`sqrt` argument to `True`, the metric computes the square root of the SILog value.

        Args:
            sqrt: If `True`, return the square root of the metric. Defaults to ``False.``
            lmbda: The regularization parameter on the variance of error. Defaults to ``1.0.``
            kwargs: Additional keyword arguments, see `Advanced metric settings <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

        Reference:
            [1] `Depth Map Prediction from a Single Image using a Multi-Scale Deep Network, NeurIPS 2014
            <https://papers.nips.cc/paper_files/paper/2014/hash/7bccfde7714a1ebadf06c5f4cea752c1-Abstract.html>`_.

            [2] `From big to small: Multi-scale local planar guidance for monocular depth estimation
            <https://arxiv.org/abs/1907.10326>`_.

        Example:

        .. code-block:: python

            from torch_uncertainty.metrics.regression import SILog
            import torch

            # Initialize the SILog metric with sqrt=True
            silog_metric = SILog(sqrt=True, lmbda=1.0)

            # Example predictions and targets
            preds = torch.tensor([1.5, 2.0, 3.5, 5.0])
            target = torch.tensor([1.4, 2.2, 3.3, 5.2])

            # Update the metric state
            silog_metric.update(preds, target)

            # Compute the Scale-Invariant Logarithmic Loss
            result = silog_metric.compute()
            print(f"SILog: {result.item():.4f}")
            # Output: SILog: 0.0686

        """
        super().__init__(**kwargs)
        self.sqrt = sqrt
        self.lmbda = lmbda
        self.add_state(
            "log_dists",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "sq_log_dists",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.

        Args:
            pred (Tensor): A prediction tensor of shape (batch)
            target (Tensor): A tensor of ground truth labels of shape (batch)
        """
        self.log_dists += torch.sum(pred.log() - target.log())
        self.sq_log_dists += torch.sum((pred.log() - target.log()) ** 2)
        self.total += target.size(0)

    def compute(self) -> Tensor:
        """Compute the Scale-Invariant Logarithmic Loss."""
        log_dists = dim_zero_cat(self.log_dists)
        sq_log_dists = dim_zero_cat(self.sq_log_dists)
        out = sq_log_dists / self.total - self.lmbda * log_dists**2 / (self.total * self.total)
        if self.sqrt:
            return torch.sqrt(out)
        return out
