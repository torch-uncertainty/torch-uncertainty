from typing import Literal

import torch
from torch import Tensor, nn

from .scaler import Scaler
from .utils import _check_classes


class VectorScaler(Scaler):
    def __init__(
        self,
        num_classes: int,
        model: nn.Module | None = None,
        init_temperature: float | Tensor = 1,
        lr: float = 0.1,
        max_iter: int = 200,
        eps: float = 1e-8,
        device: Literal["cpu", "cuda"] | torch.device | None = None,
    ) -> None:
        r"""Vector scaling post-processing for calibrated probabilities.

        Generalises temperature scaling by learning a per-class temperature vector
        :math:`\mathbf{T} \in \mathbb{R}^C_{>0}`:

        .. math::
            \tilde{\mathbf{p}}(\mathbf{x}) = \mathrm{softmax}\!\left(\mathbf{z}(\mathbf{x}) \oslash \mathbf{T}\right),

        where :math:`\oslash` is element-wise division. The :math:`C` temperatures are
        fit jointly by minimising the cross-entropy on a held-out calibration set.

        Args:
            num_classes: Number of classes :math:`C`.
            model: Model to calibrate.
            init_temperature: Initial value for the per-class temperature. A scalar
                broadcasts to all classes. Defaults to ``1``.
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
        self.set_temperature(init_temperature)

    def set_temperature(self, val: float | Tensor) -> None:
        """Set the temperature vector to a given value.

        Args:
            val: Weight temperature vector, or float.
        """
        if isinstance(val, float | int) or (isinstance(val, Tensor) and val.size == 1):
            if val <= 0:
                raise ValueError(f"Temperature value must be strictly positive. Got {val}.")
            self.inv_temp = nn.Parameter(
                torch.ones(self.num_classes, device=self.device) / val,
                requires_grad=True,
            )
        elif isinstance(val, Tensor):
            if torch.any(val <= 0):
                raise ValueError(f"Temperature value must be strictly positive. Got {val}.")
            self.inv_temp = nn.Parameter(
                val.to(dtype=torch.float32, device=self.device),
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
