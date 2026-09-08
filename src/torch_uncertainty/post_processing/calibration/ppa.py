import logging
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from torch_uncertainty.post_processing import PostProcessing

from .utils import _determine_dimensionality, _extract_data


class PPAScaler(PostProcessing):
    r"""Parameterized Probability Adjustment (PPA) calibration.

    PPA sharpens a probability vector :math:`\mathbf{p}` towards
    :math:`\mathbf{p}_0`, which distributes a mass of one uniformly over the
    classes tied for the highest probability:

    .. math::
        \mathbf{p}_{\mathrm{PPA}} = r\mathbf{p}_0 + (1-r)\mathbf{p}.

    The adjustment :math:`r \in [0, 1]` is fitted on the calibration set by
    minimising the Brier score. PPA was introduced to correct the tendency of
    random forests of probability estimation trees to produce probabilities
    biased towards the uniform distribution.

    Args:
        model: Model to calibrate. Defaults to ``None``.
        eps: Small value for stability when converting probabilities back to
            logits. Defaults to ``1e-6``.
        device: Device to use for tensor operations. Defaults to ``None``.

    References:
        [1] `Boström, H. (2008). Calibrating Random Forests. ICMLA 2008
        <https://doi.org/10.1109/ICMLA.2008.107>`_.
        [2] `Shaker, M. H., & Hüllermeier, E. (2025). Random Forest
        Calibration. Knowledge-Based Systems
        <https://doi.org/10.1016/j.knosys.2025.114143>`_.
    """

    num_classes: int
    adjustment: Tensor

    def __init__(
        self,
        model: nn.Module | None = None,
        eps: float = 1e-6,
        device: Literal["cpu", "cuda"] | torch.device | None = None,
    ) -> None:
        super().__init__(model)
        if not 0 < eps < 0.5:
            raise ValueError(f"eps must be strictly between 0 and 0.5. Got {eps}.")

        self.eps = eps
        self.device = device
        self.register_buffer("adjustment", torch.tensor(0.0, device=device))

    @staticmethod
    def _target_distribution(probs: Tensor) -> Tensor:
        """Return the uniform distribution over the most probable classes."""
        maxima = probs.amax(dim=-1, keepdim=True)
        max_mask = probs == maxima
        return max_mask / max_mask.sum(dim=-1, keepdim=True)

    def fit(self, dataloader: DataLoader, progress: bool = True) -> None:
        """Fit the PPA adjustment on calibration data.

        Args:
            dataloader: Dataloader providing the calibration data.
            progress: Whether to show a progress bar. Defaults to ``True``.
        """
        if self.model is None or isinstance(self.model, nn.Identity):  # coverage: ignore
            logging.warning(
                "model is None. Fitting post_processing method on the dataloader's data directly."
            )
            self.model = nn.Identity()

        all_logits, all_labels = _extract_data(
            dataloader=dataloader, model=self.model, device=self.device, progress=progress
        )
        self.num_classes, probs, labels = _determine_dimensionality(all_logits, all_labels)

        if self.num_classes == 1:
            probs = torch.stack((1 - probs, probs), dim=-1)
            num_label_classes = 2
        else:
            num_label_classes = self.num_classes

        labels_one_hot = F.one_hot(labels.long(), num_label_classes).to(probs.dtype)
        target_probs = self._target_distribution(probs)
        direction = target_probs - probs
        denominator = direction.square().sum()

        # Closed-form least-squares solution for the Brier objective.
        if denominator == 0:
            adjustment = torch.zeros_like(denominator)
        else:
            numerator = (direction * (labels_one_hot - probs)).sum()
            adjustment = (numerator / denominator).clamp(0, 1)

        self.adjustment.copy_(adjustment)
        self.trained = True

    @torch.no_grad()
    def forward(self, inputs: Tensor) -> Tensor:
        """Apply PPA and return calibrated logits."""
        if self.model is None:  # coverage: ignore
            raise ValueError("Provide a model before calling forward.")
        if not self.trained:
            logging.warning("Scaler not trained. Returning raw predictions.")
            return self.model(inputs)

        logits = self.model(inputs)
        if self.num_classes == 1:
            positive_probs = torch.sigmoid(logits).flatten()
            probs = torch.stack((1 - positive_probs, positive_probs), dim=-1)
        else:
            probs = torch.softmax(logits, dim=-1)

        target_probs = self._target_distribution(probs)
        calibrated_probs = self.adjustment * target_probs + (1 - self.adjustment) * probs
        calibrated_probs = calibrated_probs.clamp(self.eps, 1 - self.eps)

        if self.num_classes == 1:
            return torch.logit(calibrated_probs[:, 1], eps=self.eps).view_as(logits)
        return torch.log(calibrated_probs)
