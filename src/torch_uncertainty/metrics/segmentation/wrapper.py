import torch
from einops import rearrange
from torch import Tensor
from torchmetrics import Metric


class SegmentationMetric(Metric):
    def __init__(self, metric: Metric, subsampling_rate: float | None = None):
        super().__init__()
        if subsampling_rate is not None and (subsampling_rate <= 0.0 or subsampling_rate > 1.0):
            raise ValueError(
                f"subsampling_rate must be in the range (0.0, 1.0], but got {subsampling_rate}"
            )

        self.metric = metric
        self.subsampling_rate = subsampling_rate

    def update(self, preds: Tensor, target: Tensor, ignore_mask: Tensor | None = None):
        """Standard update method for segmentation metrics.

        Args:
            preds: logits or probabilities of shape (N, C, H, W).
            target: ground truth labels of shape (N, H, W) or (N, 1, H, W).
            ignore_mask: mask to ignore certain pixels in the computation
                of size (N, H, W) or (N, 1, H, W). Defaults to ``None``.
        """
        if target.ndim == 4:
            if target.size(1) == 1:
                target = target.squeeze(1)
            else:
                raise ValueError(
                    f"Expected target to have shape (N, H, W) or (N, 1, H, W), but got {target.shape}"
                )

        keep_mask = torch.ones_like(target, dtype=torch.bool)

        if ignore_mask is not None:
            if ignore_mask.ndim == 4:
                if ignore_mask.size(1) == 1:
                    ignore_mask = ignore_mask.squeeze(1)
                else:
                    raise ValueError(
                        f"Expected ignore_mask to have shape (N, H, W) or (N, 1, H, W), but got {ignore_mask.shape}"
                    )
            keep_mask &= ~ignore_mask

        if self.subsampling_rate is not None:
            # Subsample the pixels to speed up the computation of the metric
            total_size = target.numel()
            subsample_size = int(total_size * self.subsampling_rate)
            subsample_indices = torch.randperm(total_size, device=target.device)[:subsample_size]
            subsample_mask = torch.zeros_like(target, dtype=torch.bool).view(-1)
            subsample_mask[subsample_indices] = True
            subsample_mask = subsample_mask.view_as(target)
            keep_mask &= subsample_mask

        if keep_mask.all():
            # If all pixels are kept, we can directly use the metric without masking
            preds = rearrange(preds, "b c h w -> (b h w) c")
            target = target.flatten()
            self.metric.update(preds, target)
        else:
            preds = rearrange(preds, "b c h w -> b h w c")[keep_mask]
            target = target[keep_mask]
            self.metric.update(preds, target)

    def compute(self):
        return self.metric.compute()

    def reset(self):
        self.metric.reset()
