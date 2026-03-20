from typing import Any

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.classification import BinaryAveragePrecision


class SegmentationBinaryAveragePrecision(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    binary_aupr: Tensor
    total: Tensor
    def __init__(
        self,
        thresholds: int | list[float] | Tensor | None = None,
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        """SegmentationBinaryAveragePrecision computes the Average Precision (AP) for binary segmentation tasks.
        It aggregates the mean AP across batches and computes the average AP over all batches processed.
        """
        super().__init__(**kwargs)
        self.aupr_metric = BinaryAveragePrecision(
            thresholds=thresholds, ignore_index=ignore_index, validate_args=validate_args, **kwargs
        )
        self.add_state("binary_aupr", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        batch_size = preds.size(0)
        aupr = self.aupr_metric(preds, target)
        self.binary_aupr += aupr * batch_size
        self.total += batch_size

    def compute(self) -> Tensor:
        if self.total == 0:
            return torch.tensor(0.0, device=self.binary_aupr.device)
        return self.binary_aupr / self.total
