import logging
from typing import Literal

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .scaler import Scaler


class TemperatureScaler(Scaler):
    def __init__(
        self,
        model: nn.Module | None = None,
        init_temperature: float | Tensor = 1,
        lr: float = 0.1,
        max_iter: int = 100,
        eps: float = 1e-8,
        device: Literal["cpu", "cuda"] | torch.device | None = None,
    ) -> None:
        """Temperature scaling post-processing for calibrated probabilities.

        Args:
            model (nn.Module): Model to calibrate.
            init_temperature (float | Tensor, optional): Initial value for the temperature. Defaults to ``1``.
            lr (float, optional): Learning rate for the optimizer. Defaults to ``0.1``.
            max_iter (int, optional): Maximum number of iterations for the optimizer. Defaults to ``100``.
            eps (float): Small value for stability. Defaults to ``1e-8``.
            device (Optional[Literal["cpu", "cuda"]], optional): Device to use for optimization. Defaults to ``None``.

        References:
            [1] `On calibration of modern neural networks. In ICML 2017
            <https://arxiv.org/abs/1706.04599>`_.

        Warning:
            If the model is binary, we will by default apply the sigmoid before transposing the prediction to the
            corresponding 2-class logits.

        Note:
            The Scaler will log an error if the temperature after fitting is negative.
        """
        super().__init__(model=model, lr=lr, max_iter=max_iter, eps=eps, device=device)

        if init_temperature <= 0:
            raise ValueError(f"Initial temperature value must be positive. Got {init_temperature}.")

        self.set_temperature(init_temperature)

    def fit(
        self,
        dataloader: DataLoader,
        save_logits: bool = False,
        progress: bool = True,
    ) -> None:
        super().fit(dataloader=dataloader, save_logits=save_logits, progress=progress)
        if self.inv_temp.item() <= 0:  # coverage: ignore
            logging.error(
                "TemperatureScaler converged to a negative temperature %.3f.", 1 / self.inv_temp
            )

    def set_temperature(self, val: float | Tensor) -> None:
        """Set the temperature to a fixed value.

        Args:
            val (float | Tensor): Temperature value.
        """
        if val <= 0:
            raise ValueError(f"Temperature value must be strictly positive. Got {val}.")

        self.inv_temp = nn.Parameter(torch.ones(1, device=self.device) / val, requires_grad=True)
        self.trained = False

    def _scale(self, logits: Tensor) -> Tensor:
        return self.inv_temp * logits

    @property
    def inv_temperature(self) -> list:
        return [self.inv_temp]

    @property
    def temperature(self) -> list:
        return [1 / self.inv_temp]
