from typing import Literal

import torch
from torch import Tensor, nn

from .scaler import Scaler


class VectorScaler(Scaler):
    def __init__(
        self,
        num_classes: int,
        model: nn.Module | None = None,
        init_temperature: float = 1,
        lr: float = 0.1,
        max_iter: int = 200,
        eps: float = 1e-8,
        device: Literal["cpu", "cuda"] | torch.device | None = None,
    ) -> None:
        """Vector scaling post-processing for calibrated probabilities.

        Args:
            model (nn.Module): Model to calibrate.
            num_classes (int): Number of classes.
            init_temperature (float, optional): Initial value for the weights. Defaults to ``1``.
            lr (float, optional): Learning rate for the optimizer. Defaults to ``0.1``.
            max_iter (int, optional): Maximum number of iterations for the optimizer. Defaults to ``100``.
            eps (float): Small value for stability. Defaults to ``1e-8``.
            device (Optional[Literal["cpu", "cuda"]], optional): Device to use for optimization. Defaults to ``None``.

        References:
            [1] `On calibration of modern neural networks. In ICML 2017
            <https://arxiv.org/abs/1706.04599>`_.

        Warning:
            If the model is binary, we will by default apply the sigmoid before transposing the prediction to the
            2-class case.
        """
        super().__init__(model=model, lr=lr, max_iter=max_iter, eps=eps, device=device)

        if not isinstance(num_classes, int):
            raise TypeError(f"num_classes must be an integer. Got {num_classes}.")
        if num_classes <= 0:
            raise ValueError(f"The number of classes must be positive. Got {num_classes}.")
        self.num_classes = num_classes

        self.set_temperature(init_temperature)

    def set_temperature(self, val: float | Tensor) -> None:
        """Set the temperature vector to a given value.

        Args:
            val (float | Tensor): Weight temperature vector, or float.
        """
        if isinstance(val, float | int) or (isinstance(val, Tensor) and val.size == 1):
            if val <= 0:
                raise ValueError(f"Temperature value must be strictly positive. Got {val}")
            self.inv_temp = nn.Parameter(
                torch.ones(self.num_classes, device=self.device) / val,
                requires_grad=True,
            )
        elif isinstance(val, Tensor):
            if torch.any(val <= 0):
                raise ValueError(f"Temperature value must be strictly positive. Got {val}")
            self.inv_temp = nn.Parameter(
                val.to(device=self.device),
                requires_grad=True,
            )
        else:
            raise ValueError(f"val should be a float or a Tensor. Got {val}.")
        self.trained = False

    def _scale(self, logits: torch.Tensor) -> torch.Tensor:
        return self.inv_temp * logits

    @property
    def inv_temperature(self) -> list:
        return [self.inv_temp]

    @property
    def temperature(self) -> list:
        return [1 / self.inv_temp]
