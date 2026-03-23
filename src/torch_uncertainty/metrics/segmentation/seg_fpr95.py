import torch
from torch import Tensor
from torchmetrics import Metric

from torch_uncertainty.metrics import FPR95


class SegmentationFPR95(Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    fpr95: Tensor
    total: Tensor

    def __init__(self, pos_label: int, **kwargs) -> None:
        """FPR95 metric for segmentation tasks.
        Compute the mean FPR95 per batch across all batches.

        Args:
            pos_label (int): The positive label in the segmentation OOD detection task.
            **kwargs: Additional keyword arguments for the FPR95 metric.
        """
        super().__init__(**kwargs)
        self.fpr95_metric = FPR95(pos_label, **kwargs)
        self.add_state("fpr95", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        batch_size = preds.size(0)
        self.fpr95 += self.fpr95_metric(preds, target) * batch_size
        self.total += batch_size

    def compute(self) -> Tensor:
        if self.total == 0:
            return torch.tensor(torch.nan, device=self.fpr95.device)
        return self.fpr95 / self.total
