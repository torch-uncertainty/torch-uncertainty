import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities import rank_zero_warn


class PAvPU(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    accurate_certain: Tensor
    accurate_uncertain: Tensor
    inaccurate_certain: Tensor
    inaccurate_uncertain: Tensor

    def __init__(
        self, patch_size: int, acc_threshold: float = 0.5, unc_threshold: float = 0.5
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.acc_threshold = acc_threshold
        self.unc_threshold = unc_threshold

        self.add_state("accurate_certain", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("accurate_uncertain", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("inaccurate_certain", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("inaccurate_uncertain", default=torch.tensor(0), dist_reduce_fx="sum")

    @staticmethod
    def _squeeze_channel_dim(x: Tensor, name: str) -> Tensor:
        if x.ndim == 4:
            if x.size(1) == 1:
                return x.squeeze(1)
            raise ValueError(
                f"Expected {name} to have shape (N, H, W) or (N, 1, H, W), but got {x.shape}"
            )
        return x

    def _validate_shapes(
        self, preds: Tensor, target: Tensor, ignore_mask: Tensor | None
    ) -> tuple[Tensor, Tensor, Tensor]:
        if preds.ndim != 4:
            raise ValueError(f"Expected preds to have shape (N, C, H, W), but got {preds.shape}.")

        target = self._squeeze_channel_dim(target, "target")
        if preds.shape[0] != target.shape[0] or preds.shape[-2:] != target.shape[-2:]:
            raise ValueError(
                "Expected preds and target to have matching batch and spatial dimensions, "
                f"but got preds={preds.shape} and target={target.shape}."
            )

        if ignore_mask is None:
            valid = torch.ones_like(target, dtype=torch.bool)
        else:
            ignore_mask = self._squeeze_channel_dim(ignore_mask, "ignore_mask")
            if ignore_mask.shape != target.shape:
                raise ValueError(
                    "Expected ignore_mask to have shape (N, H, W) matching target, "
                    f"but got ignore_mask={ignore_mask.shape} and target={target.shape}."
                )
            valid = ~ignore_mask.bool()

        return preds, target, valid

    def update(self, preds: Tensor, target: Tensor, ignore_mask: Tensor | None = None) -> None:
        """Update the metric with new predictions and targets.

        Args:
            preds: Tensor of shape (N, C, H, W) containing the predicted probabilities or logits.
            target: Tensor of shape (N, H, W) or (N, 1, H, W) containing the ground truth labels.
            ignore_mask: Optional tensor of shape (N, H, W) or (N, 1, H, W) indicating which pixels to ignore.
        """
        preds, target, valid = self._validate_shapes(preds, target, ignore_mask)

        if not torch.all((preds >= 0) & (preds <= 1)):
            preds = F.softmax(preds, dim=1)
            rank_zero_warn(
                "Preds do not seem to be probabilities in the range [0, 1]."
                " Applying softmax to convert logits to probabilities."
            )

        hard_preds = preds.argmax(dim=1)
        acc_map = (hard_preds == target).float()
        unc_map = 1 - preds.max(dim=1).values

        ps = self.patch_size
        valid_f = valid.float().unsqueeze(1)
        # Compute patch statistics
        pooled_valid = F.avg_pool2d(
            valid_f,
            kernel_size=ps,
            stride=ps,
            ceil_mode=True,
            count_include_pad=False,
        )
        pooled_acc = F.avg_pool2d(
            (acc_map * valid.float()).unsqueeze(1),
            kernel_size=ps,
            stride=ps,
            ceil_mode=True,
            count_include_pad=False,
        )
        pooled_unc = F.avg_pool2d(
            (unc_map * valid.float()).unsqueeze(1),
            kernel_size=ps,
            stride=ps,
            ceil_mode=True,
            count_include_pad=False,
        )

        valid_patches = pooled_valid > 0
        patch_acc = pooled_acc / pooled_valid.clamp_min(1e-12)
        patch_unc = pooled_unc / pooled_valid.clamp_min(1e-12)

        accurate = patch_acc > self.acc_threshold
        uncertain = patch_unc > self.unc_threshold

        self.accurate_certain += (valid_patches & accurate & ~uncertain).sum()
        self.accurate_uncertain += (valid_patches & accurate & uncertain).sum()
        self.inaccurate_certain += (valid_patches & (~accurate) & ~uncertain).sum()
        self.inaccurate_uncertain += (valid_patches & (~accurate) & uncertain).sum()

    def compute(self) -> Tensor:
        """Compute the final PAvPU metric.

        Returns:
            Tensor containing the computed PAvPU value.
        """
        n_ac = self.accurate_certain
        n_au = self.accurate_uncertain
        n_ic = self.inaccurate_certain
        n_iu = self.inaccurate_uncertain
        return (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu)
