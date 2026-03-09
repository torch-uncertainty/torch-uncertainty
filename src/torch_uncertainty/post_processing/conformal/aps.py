from typing import Literal

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .abstract import Conformal


class ConformalClsAPS(Conformal):
    def __init__(
        self,
        alpha: float,
        model: nn.Module | None = None,
        randomized: bool = True,
        ts_init_val: float = 1,
        ts_lr: float = 0.1,
        ts_max_iter: int = 100,
        enable_ts: bool = True,
        device: Literal["cpu", "cuda"] | torch.device | None = None,
    ) -> None:
        r"""Conformal prediction with APS scores.

        Args:
            alpha (float): The confidence level meaning we allow :math:`1-\alpha` error.
            model (nn.Module | None): Trained classification model. Defaults to ``None``.
            randomized (bool): Whether to use randomized smoothing in APS. Defaults to ``True``.
            ts_init_val (float, optional): Initial value for the temperature.
                Defaults to ``1.0``.
            ts_lr (float, optional): Learning rate for the temperature scaling optimizer. Defaults to ``0.1``.
            ts_max_iter (int, optional): Maximum number of iterations for the temperature scaling
                optimizer. Defaults to ``100``.
            enable_ts (bool): Whether to scale the logits. Defaults to ``False``.
            device (Literal["cpu", "cuda"] | torch.device | None, optional): device.
                Defaults to ``None``.

        Warning:
            This implementation only works in the multiclass setting. Raise an issue if binary is needed.

        Reference:
            - TODO:

        Code inspired by TorchCP.
        """
        super().__init__(
            alpha=alpha,
            model=model,
            ts_init_val=ts_init_val,
            ts_lr=ts_lr,
            ts_max_iter=ts_max_iter,
            enable_ts=enable_ts,
            device=device,
        )
        self.randomized = randomized

    def model_forward(self, inputs: Tensor) -> Tensor:
        """Apply the model and return the scores."""
        self.model.eval()
        return self.model(inputs.to(self.device)).softmax(-1)

    def _sort_sum(self, probs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Sort probabilities and compute cumulative sums."""
        ordered, indices = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(ordered, dim=-1)
        return indices, ordered, cumsum

    def _calculate_all_labels(self, probs: Tensor) -> Tensor:
        """Calculate APS scores for all labels."""
        indices, ordered, cumsum = self._sort_sum(probs)
        if self.randomized:
            noise = torch.rand(probs.shape, device=probs.device)
        else:
            noise = torch.zeros_like(probs)

        ordered_scores = cumsum - ordered * noise
        _, sorted_indices = torch.sort(indices, descending=False, dim=-1)
        return ordered_scores.gather(dim=-1, index=sorted_indices)

    def _calculate_single_label(self, probs: Tensor, label: Tensor) -> Tensor:
        """Calculate APS score for a single label."""
        indices, ordered, cumsum = self._sort_sum(probs)
        if self.randomized:
            noise = torch.rand(indices.shape[0], device=probs.device)
        else:
            noise = torch.zeros(indices.shape[0], device=probs.device)

        idx = torch.where(indices == label.view(-1, 1))
        return cumsum[idx] - noise * ordered[idx]

    @torch.no_grad()
    def fit(self, dataloader: DataLoader) -> None:
        """Calibrate the APS threshold q_hat on a calibration set."""
        if self.enable_ts:
            self.model.fit(dataloader=dataloader)

        aps_scores = []
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            probs = self.model_forward(inputs)
            scores = self._calculate_single_label(probs, labels)
            aps_scores.append(scores)

        self.q_hat = torch.quantile(torch.cat(aps_scores), 1 - self.alpha).item()

    @torch.no_grad()
    def conformal(self, inputs: Tensor) -> Tensor:
        """Compute the prediction set for each input."""
        probs = self.model_forward(inputs)
        pred_set = self._calculate_all_labels(probs) <= self.quantile
        return pred_set.float() / pred_set.sum(dim=1, keepdim=True)
