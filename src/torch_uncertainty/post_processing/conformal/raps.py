from typing import Literal

import torch
from torch import Tensor, nn

from .aps import ConformalClsAPS


class ConformalClsRAPS(ConformalClsAPS):
    def __init__(
        self,
        alpha: float,
        model: nn.Module | None = None,
        randomized: bool = True,
        penalty: float = 0.1,
        regularization_rank: int = 1,
        ts_init_val: float = 1.0,
        ts_lr: float = 0.1,
        ts_max_iter: int = 100,
        enable_ts: bool = False,
        device: Literal["cpu", "cuda"] | torch.device | None = None,
    ) -> None:
        r"""Conformal prediction with RAPS scores.

        Args:
            alpha (float): The confidence level meaning we allow :math:`1-\alpha` error.
            model (nn.Module | None): Trained classification model. Defaults to ``None``.
            randomized (bool): Whether to use randomized smoothing in RAPS. Defaults to ``True``.
            penalty (float): Regularization weight. Defaults to ``0.1``.
            regularization_rank (int): Rank threshold for regularization. Defaults to ``1``.
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
            randomized=randomized,
            ts_init_val=ts_init_val,
            ts_lr=ts_lr,
            ts_max_iter=ts_max_iter,
            enable_ts=enable_ts,
            device=device,
        )
        if penalty < 0:
            raise ValueError(f"penalty should be non-negative. Got {penalty}.")

        if not isinstance(regularization_rank, int):
            raise TypeError(f"regularization_rank should be an integer. Got {regularization_rank}.")

        if regularization_rank < 0:
            raise ValueError(
                f"regularization_rank should be non-negative. Got {regularization_rank}."
            )

        self.penalty = penalty
        self.regularization_rank = regularization_rank

    def _calculate_all_labels(self, probs: Tensor) -> Tensor:
        indices, ordered, cumsum = self._sort_sum(probs)
        if self.randomized:
            noise = torch.rand(probs.shape, device=probs.device)
        else:
            noise = torch.zeros_like(probs)
        reg = torch.maximum(
            self.penalty
            * (
                torch.arange(1, probs.shape[-1] + 1, device=probs.device) - self.regularization_rank
            ),
            torch.tensor(0, device=probs.device),
        )
        ordered_scores = cumsum - ordered * noise + reg
        _, sorted_indices = torch.sort(indices, descending=False, dim=-1)
        return ordered_scores.gather(dim=-1, index=sorted_indices)

    def _calculate_single_label(self, probs: Tensor, label: Tensor) -> Tensor:
        indices, ordered, cumsum = self._sort_sum(probs)
        if self.randomized:
            noise = torch.rand(indices.shape[0], device=probs.device)
        else:
            noise = torch.zeros(indices.shape[0], device=probs.device)
        idx = torch.where(indices == label.view(-1, 1))
        reg = torch.maximum(
            self.penalty * (idx[1] + 1 - self.regularization_rank), torch.tensor(0).to(probs.device)
        )
        return cumsum[idx] - noise * ordered[idx] + reg
