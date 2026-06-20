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
        r"""Image-averaged binary AUROC for dense binary segmentation tasks.

        At each image, a per-pixel binary AUROC is computed from the pixel scores
        :math:`s_{ij}` and binary labels :math:`y_{ij} \in \{0, 1\}`:

        .. math::
            \text{AUROC}_b = \int_0^1 \text{TPR}_b\!\left(\text{FPR}_b^{-1}(u)\right) \mathrm{d}u,

        where TPR and FPR are computed by sweeping a threshold over the pixel-level
        scores of image :math:`b`. The final metric is the average over all images:

        .. math::
            \text{AUROC} = \frac{1}{B} \sum_{b=1}^{B} \text{AUROC}_b.

        This image-wise averaging is the convention used in the dense OOD-detection
        literature (e.g., MUAD) and behaves better than computing AUROC over the
        flattened set of all pixels when image sizes or OOD prevalences vary.

        Args:
            max_fpr: If set, computes the partial AUROC up to this FPR value
                (passed to :class:`~torchmetrics.classification.BinaryAUROC`).
            thresholds: Optional explicit thresholds to use when computing the ROC.
            ignore_index: Optional label value to ignore.
            validate_args: Whether to validate input arguments.
            kwargs: Additional keyword arguments, see `Advanced metric settings
                <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.
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

    def update(self, preds: Tensor, target: Tensor) -> None:  # pyrefly: ignore[bad-override]
        batch_size = preds.size(0)
        auroc = self.auroc_metric(preds, target)
        self.binary_auroc += auroc * batch_size
        self.total += batch_size

    def compute(self) -> Tensor:
        if self.total == 0:
            return torch.tensor(0.0, device=self.binary_auroc.device)
        return self.binary_auroc / self.total
