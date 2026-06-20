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
        r"""Image-averaged binary Average Precision for dense segmentation tasks.

        Per-image Average Precision summarises the precision-recall curve obtained by
        sweeping a threshold over the pixel scores of image :math:`b`:

        .. math::
            \text{AP}_b = \sum_{k} \left( R_b(k) - R_b(k-1) \right) P_b(k),

        where :math:`P_b(k)` and :math:`R_b(k)` are the precision and recall at the
        :math:`k`-th threshold. The final metric is averaged over all :math:`B` images:

        .. math::
            \text{AP} = \frac{1}{B} \sum_{b=1}^{B} \text{AP}_b.

        As for :class:`SegmentationBinaryAUROC`, image-wise averaging is the convention
        used in the dense OOD-detection literature.

        Args:
            thresholds: Optional explicit thresholds for the PR curve, see
                :class:`~torchmetrics.classification.BinaryAveragePrecision`.
            ignore_index: Optional label value to ignore.
            validate_args: Whether to validate input arguments.
            kwargs: Additional keyword arguments, see `Advanced metric settings
                <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.
        """
        super().__init__(**kwargs)
        self.aupr_metric = BinaryAveragePrecision(
            thresholds=thresholds, ignore_index=ignore_index, validate_args=validate_args, **kwargs
        )
        self.add_state("binary_aupr", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # pyrefly: ignore[bad-override]
        batch_size = preds.size(0)
        aupr = self.aupr_metric(preds, target)
        self.binary_aupr += aupr * batch_size
        self.total += batch_size

    def compute(self) -> Tensor:
        if self.total == 0:
            return torch.tensor(0.0, device=self.binary_aupr.device)
        return self.binary_aupr / self.total
