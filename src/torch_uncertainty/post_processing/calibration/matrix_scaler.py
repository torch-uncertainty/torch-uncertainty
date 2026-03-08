from typing import Literal

import torch
from torch import Tensor, device, nn
from torch.nn.functional import linear

from .scaler import Scaler


class MatrixScaler(Scaler):
    def __init__(
        self,
        num_classes: int,
        model: nn.Module | None = None,
        init_temperature_weight: float = 1,
        init_temperature_bias: float | None = None,
        lr: float = 0.1,
        max_iter: int = 200,
        eps: float = 1e-8,
        device: Literal["cpu", "cuda"] | device | None = None,
    ) -> None:
        """Matrix scaling post-processing for calibrated probabilities.

        Args:
            num_classes (int): Number of classes.
            model (nn.Module | None): Model to calibrate. Defaults to ``None``.
            init_temperature_weight (float, optional): Initial value for the weights. Defaults to ``1``.
            init_temperature_bias (float | None, optional): Initial value for the bias. The inverse bias will be
                set to the ``0`` vector if set to ``None``. Defaults to ``None``.
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

        self.set_temperature(init_temperature_weight, init_temperature_bias)

    def set_temperature(self, val_weight: float | Tensor, val_bias: float | Tensor | None) -> None:
        """Set the temperature matrix to a given value.

        Args:
            val_weight (float | Tensor): Weight temperature value.
            val_bias (float | Tensor): Bias temperature value.
        """
        eye = torch.eye(self.num_classes, device=self.device)
        self.inv_temperature_weight = nn.Parameter(
            eye / val_weight,
            requires_grad=True,
        )
        if val_bias is None:
            bias = torch.zeros(self.num_classes, device=self.device)
        else:
            bias = torch.ones(self.num_classes, device=self.device) / val_bias

        self.inv_temperature_bias = nn.Parameter(
            bias,
            requires_grad=True,
        )
        self.trained = False

    def _scale(self, logits: Tensor) -> Tensor:
        return linear(logits, self.inv_temperature_weight, self.inv_temperature_bias)

    @property
    def inv_temperature(self) -> list:
        return [self.inv_temperature_weight, self.inv_temperature_bias]

    @property
    def temperature(self) -> list:
        return [torch.inverse(self.inv_temperature_weight), 1 / self.inv_temperature_bias]
