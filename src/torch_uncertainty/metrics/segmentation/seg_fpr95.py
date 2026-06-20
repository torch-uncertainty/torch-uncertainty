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
        r"""Image-averaged FPR@95 TPR for dense binary segmentation tasks.

        For each image, a per-pixel False Positive Rate at 95% True Positive Rate is
        computed (see :class:`~torch_uncertainty.metrics.classification.FPR95`) from
        the pixel scores and binary OOD labels. The metric is then averaged over the
        :math:`B` images of the test set:

        .. math::
            \text{FPR95} = \frac{1}{B} \sum_{b=1}^{B} \text{FPR95}_b.

        Image-wise averaging is the convention used in the dense OOD-detection
        literature.

        Args:
            pos_label: The positive label in the segmentation OOD detection task
                (typically ``1`` for OOD pixels).
            kwargs: Additional keyword arguments for the underlying
                :class:`~torch_uncertainty.metrics.classification.FPR95` metric.
        """
        super().__init__(**kwargs)
        self.fpr95_metric = FPR95(pos_label, **kwargs)
        self.add_state("fpr95", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # pyrefly: ignore[bad-override]
        batch_size = preds.size(0)
        self.fpr95 += self.fpr95_metric(preds, target) * batch_size
        self.total += batch_size

    def compute(self) -> Tensor:
        if self.total == 0:
            return torch.tensor(torch.nan, device=self.fpr95.device)
        return self.fpr95 / self.total
