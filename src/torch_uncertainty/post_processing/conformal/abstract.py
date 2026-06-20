from abc import abstractmethod
from typing import Literal

import torch
from torch import Tensor, nn

from torch_uncertainty.post_processing import TemperatureScaler
from torch_uncertainty.post_processing.abstract import PostProcessing


class Conformal(PostProcessing):
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
        r"""Abstract base class for split-conformal classification predictors.

        Builds prediction sets :math:`\mathcal{C}(\mathbf{x}) \subseteq \{1, \dots, C\}`
        with the *marginal coverage guarantee*

        .. math::
            \mathbb{P}\!\left[ Y \in \mathcal{C}(X) \right] \geq 1 - \alpha,

        provided that the calibration and test points are exchangeable. At fit time,
        non-conformity scores are computed on a held-out calibration set and the
        empirical :math:`(1 - \alpha)`-quantile :math:`\hat{q}` is stored in
        :attr:`q_hat`. At test time, the prediction set is built by including all
        classes whose conformal score is below :math:`\hat{q}`. See :class:`ConformalClsTHR`,
        :class:`ConformalClsAPS`, and :class:`ConformalClsRAPS` for concrete scores.

        Args:
            alpha: Target mis-coverage level :math:`\alpha \in (0, 1)`. A smaller
                :math:`\alpha` yields larger prediction sets.
            model: Underlying classifier.
            ts_init_val: Initial temperature used when :attr:`enable_ts` is ``True``.
            ts_lr: Learning rate for the temperature optimizer.
            ts_max_iter: Maximum number of iterations for the temperature optimizer.
            enable_ts: If ``True``, wraps the model in a :class:`TemperatureScaler`
                fit on the calibration set before computing the conformal scores.
            device: Device to run the post-processing on.

        Warning:
            This implementation only works in the multiclass setting. Raise an issue
            if binary support is needed.
        """
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

    def set_model(self, model: nn.Module) -> None:
        if self.enable_ts:
            assert self.model is not None
            self.model.set_model(model=model.eval())
        else:
            self.model = model

    def model_forward(self, inputs: Tensor) -> Tensor:
        """Apply the model and return the scores."""
        assert self.model is not None
        self.model.eval()
        return self.model(inputs.to(self.device)).softmax(-1)

    @abstractmethod
    def conformal(self, inputs: Tensor) -> Tensor:
        """Apply the conformal prediction rule to the inputs."""

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conformal(inputs)

    @property
    def quantile(self) -> Tensor:
        if self.q_hat is None:
            raise RuntimeError("Quantile q_hat is not set. Run `.fit()` first.")
        return self.q_hat

    @property
    def temperature(self) -> float:
        if self.enable_ts and (self.model is not None):
            return self.model.temperature[0].item()
        raise RuntimeError("Cannot return temperature when enable_ts is False.")
