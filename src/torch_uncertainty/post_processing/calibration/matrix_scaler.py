from typing import Literal

import torch
from torch import Tensor, device, nn
from torch.nn.functional import linear

from .scaler import Scaler
from .utils import _check_classes


class MatrixScaler(Scaler):
    def __init__(
        self,
        num_classes: int,
        model: nn.Module | None = None,
        init_weight_temperature: float | Tensor = 1,
        init_bias_temperature: float | Tensor | None = None,
        lr: float = 0.1,
        max_iter: int = 200,
        eps: float = 1e-8,
        device: Literal["cpu", "cuda"] | device | None = None,
    ) -> None:
        r"""Matrix scaling post-processing for calibrated probabilities.

        Generalises temperature and vector scaling by applying a full affine
        transformation to the logits before the softmax:

        .. math::
            \tilde{\mathbf{p}}(\mathbf{x}) = \mathrm{softmax}\!\left(\mathbf{W} \mathbf{z}(\mathbf{x}) + \mathbf{b}\right),

        where :math:`\mathbf{W} \in \mathbb{R}^{C \times C}` and
        :math:`\mathbf{b} \in \mathbb{R}^C` are fit by minimising the cross-entropy on
        a held-out calibration set. Matrix scaling has :math:`C^2 + C` parameters and
        can therefore overfit on small calibration sets — consider :class:`VectorScaler`
        or :class:`DirichletScaler` when calibration data is scarce.

        Args:
            num_classes: Number of classes :math:`C`.
            model: Model to calibrate. Defaults to ``None``.
            init_weight_temperature: Initial value for the weight matrix (used as
                :math:`1/T \cdot \mathbf{I}`). Defaults to ``1``.
            init_bias_temperature: Initial value for the bias. The inverse bias will be
                set to the ``0`` vector if set to ``None``. Defaults to ``None``.
            lr: Learning rate for the optimizer. Defaults to ``0.1``.
            max_iter: Maximum number of iterations for the optimizer. Defaults to ``200``.
            eps: Small value for stability. Defaults to ``1e-8``.
            device: Device to use for optimization. Defaults to ``None``.

        References:
            [1] `Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration
            of modern neural networks. ICML 2017 <https://arxiv.org/abs/1706.04599>`_.

        Warning:
            For binary models, a sigmoid is applied before the prediction is transposed
            to the 2-class case.
        """
        super().__init__(model=model, lr=lr, max_iter=max_iter, eps=eps, device=device)

        _check_classes(num_classes)
        self.num_classes = num_classes
        self.set_temperature(init_weight_temperature, init_bias_temperature)

    def set_temperature(self, val_weight: float | Tensor, val_bias: float | Tensor | None) -> None:
        """Set the temperature matrix to a given value.

        Args:
            val_weight: Weight temperature value.
            val_bias: Bias temperature value.
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
