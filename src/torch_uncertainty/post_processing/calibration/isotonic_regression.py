import logging
from importlib import util
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from torch_uncertainty.post_processing import PostProcessing

from .utils import _determine_dimensionality, _extract_data

if util.find_spec("sklearn"):
    from sklearn.isotonic import IsotonicRegression

    sklearn_installed = True
else:  # coverage: ignore
    sklearn_installed = False


class IsotonicRegressionScaler(PostProcessing):
    ir_models: list[IsotonicRegression]
    num_classes: int

    def __init__(
        self,
        model: nn.Module | None = None,
        eps: float = 1e-6,
        device: Literal["cpu", "cuda"] | torch.device | None = None,
    ) -> None:
        r"""Isotonic Regression post-processing for calibrated probabilities
        (Zadrozny & Elkan, 2002).

        A non-parametric calibration method that fits a piecewise-constant,
        monotonically non-decreasing mapping :math:`f` from the uncalibrated
        probabilities to the calibrated ones by minimising the mean squared error:

        .. math::
            \min_{f \text{ non-decreasing}} \sum_{i=1}^{N} \left( y_i - f(\hat{p}_i) \right)^2.

        Compared to :class:`HistogramBinningScaler`, the bins are not pre-defined and
        the resulting mapping is smoother. Multi-class calibration is handled with a
        one-vs-rest approach: one isotonic regressor is fit per class and the
        calibrated probabilities are renormalised to sum to one.

        Args:
            model: Model to calibrate. Defaults to ``None``.
            eps: Small value for stability when converting probs back to logits.
                Defaults to ``1e-6``.
            device: Device to use for tensor operations. Defaults to ``None``.

        References:
            [1] `Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into
            accurate multiclass probability estimates. KDD 2002
            <https://dl.acm.org/doi/10.1145/775047.775151>`_.

        Note:
            This implementation uses scikit-learn's
            :class:`~sklearn.isotonic.IsotonicRegression` as the underlying solver.

        Warning:
            Isotonic regression requires a sufficient amount of calibration data
            to avoid overfitting the step function, especially in multi-class
            scenarios.
        """
        if not sklearn_installed:
            raise ImportError(
                "The scikit-learn library is not installed. Please install "
                "torch_uncertainty with the others option: pip install -U torch_uncertainty[others]"
            )
        super().__init__(model)
        self.eps = eps
        self.device = device
        self.ir_models: list[IsotonicRegression] = []

    def fit(
        self,
        dataloader: DataLoader,
        progress: bool = True,
    ) -> None:
        """Fit the isotonic regression models to the calibration data.

        For binary classification, a single isotonic regressor is fit.
        For multiclass classification, a One-vs-Rest (OvR) approach is used:
        one regressor is trained per class to predict the probability of that
        class versus all others.

        Args:
            dataloader: Dataloader providing the calibration data (logits and targets).
            progress: Whether to show a progress bar during data extraction. Defaults to ``True``.
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
        probs, labels = probs.numpy(), labels.numpy()
        self.ir_models = []

        # Fit Isotonic Regression
        if self.num_classes == 1:
            ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            ir.fit(probs, labels)
            self.ir_models.append(ir)
        else:
            # One-vs-Rest for multi-class
            for c in range(self.num_classes):
                ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                c_labels = (labels == c).astype(int)
                ir.fit(probs[:, c], c_labels)
                self.ir_models.append(ir)
        self.trained = True

    @torch.no_grad()
    def forward(self, inputs: Tensor) -> Tensor:
        """Apply the fitted Isotonic Regression and return calibrated logits.

        The forward pass transforms the input logits into probabilities,
        applies the isotonic mapping, and then converts the resulting
        probabilities back into the logit space for compatibility with
        downstream loss functions or metrics.

        Args:
            inputs: Input logits to be calibrated.

        Returns:
            Tensor: Calibrated logits.
        """
        if self.model is None:  # coverage: ignore
            raise ValueError("Provide a model before calling forward.")
        if not self.trained:
            logging.warning("Scaler not trained. Returning raw predictions.")
            return self.model(inputs)

        logits = self.model(inputs)

        # Binary case
        if self.num_classes == 1:
            probs = torch.sigmoid(logits).cpu().flatten().numpy()
            calib_probs = self.ir_models[0].predict(probs)
            calib_probs = torch.from_numpy(calib_probs).to(logits.device).view_as(logits)
        else:
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            calib_probs = np.zeros_like(probs)
            for c in range(self.num_classes):
                calib_probs[:, c] = self.ir_models[c].predict(probs[:, c])

            # Normalize so multiclass probabilities sum to 1
            calib_probs = calib_probs / (calib_probs.sum(axis=-1, keepdims=True) + 1e-12)
            calib_probs = torch.from_numpy(calib_probs).to(logits.device)

        # Convert calibrated probabilities back to pseudo-logits
        calib_probs = calib_probs.clamp(self.eps, 1 - self.eps)
        if self.num_classes == 1:
            return torch.logit(calib_probs, eps=self.eps)
        return torch.log(calib_probs)
