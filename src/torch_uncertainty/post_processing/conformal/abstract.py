from abc import abstractmethod
from typing import Literal

import torch
from torch import Tensor, nn

from torch_uncertainty.post_processing import TemperatureScaler
from torch_uncertainty.post_processing.abstract import PostProcessing


class Conformal(PostProcessing):
    """Conformal base class.

    Warning:
        This implementation only works in the multiclass setting. Raise an issue if binary is needed.
    """

    q_hat: Tensor | None = None

    def __init__(
        self,
        alpha: float,
        model: nn.Module | None,
        ts_init_val: float,
        ts_lr: float,
        ts_max_iter: int,
        enable_ts: bool,
        device: Literal["cpu", "cuda"] | torch.device | None,
    ) -> None:
        super().__init__(model=model)
        self.alpha = alpha
        self.enable_ts = enable_ts
        if enable_ts:
            self.model = TemperatureScaler(
                model=model,
                init_temperature=ts_init_val,
                lr=ts_lr,
                max_iter=ts_max_iter,
                device=device,
            )
        else:
            self.model = model
        self.device = device or "cpu"

    def set_model(self, model: nn.Module | None) -> None:
        if self.enable_ts:
            self.model.set_model(model=model.eval())
        else:
            self.model = model

    def model_forward(self, inputs: Tensor) -> Tensor:
        """Apply the model and return the scores."""
        self.model.eval()
        return self.model(inputs.to(self.device)).softmax(-1)

    @abstractmethod
    def conformal(self, inputs: Tensor) -> Tensor: ...

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conformal(inputs)

    @property
    def quantile(self) -> Tensor:
        if self.q_hat is None:
            raise RuntimeError("Quantile q_hat is not set. Run `.fit()` first.")
        return self.q_hat

    @property
    def temperature(self) -> float:
        if self.enable_ts:
            return self.model.temperature[0].item()
        raise RuntimeError("Cannot return temperature when enable_ts is False.")
