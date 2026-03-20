from typing import Any

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.classification import BinaryAUROC


class SegmentationBinaryAUROC(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    binary_auroc: Tensor
    total: Tensor

    def __init__(
        self,
        max_fpr: float | None = None,
        thresholds: int | list[float] | Tensor | None = None,
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        """SegmentationBinaryAUROC computes the Area Under the Receiver Operating Characteristic Curve (AUROC)
        for binary segmentation tasks. It aggregates the AUROC across batches and computes the average AUROC
        over all batches processed.
        """
        super().__init__(**kwargs)
        self.auroc_metric = BinaryAUROC(
            max_fpr=max_fpr,
            thresholds=thresholds,
            ignore_index=ignore_index,
            validate_args=validate_args,
            **kwargs,
        )
        self.add_state("binary_auroc", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        batch_size = preds.size(0)
        auroc = self.auroc_metric(preds, target)
        self.binary_auroc += auroc * batch_size
        self.total += batch_size

    def compute(self) -> Tensor:
        if self.total == 0:
            return torch.tensor(0.0, device=self.binary_auroc.device)
        return self.binary_auroc / self.total
