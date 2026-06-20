import warnings
from typing import Literal, cast

import torch
from torch import Tensor
from torch.distributions import Distribution, Independent
from torchmetrics.classification import BinaryCalibrationError
from torchmetrics.functional.classification.calibration_error import (
    _binning_bucketize,
)
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.plot import _PLOT_OUT_TYPE

from torch_uncertainty.metrics.classification.calibration.calibration_error import reliability_chart


class QuantileCalibrationError(BinaryCalibrationError):
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False
    not_implemented_error: bool = False

    def __init__(
        self,
        num_bins: int = 15,
        norm: Literal["l1", "l2", "max"] = "l1",
        ignore_index=None,
        validate_args=True,
        **kwargs,
    ) -> None:
        r"""Quantile Calibration Error for regression tasks.

        For each confidence level :math:`\alpha \in (0, 1)`, a well-calibrated
        probabilistic regressor should ensure that a fraction :math:`\alpha` of the
        ground-truth targets lies inside the centered :math:`\alpha`-credible interval
        of the predicted distribution :math:`p_\theta(\cdot \mid x)`. Concretely, let

        .. math::
            \hat{c}(\alpha) = \frac{1}{N} \sum_{i=1}^{N}
            \mathbf{1}\!\left[ y_i \in
            \left[ F^{-1}_{\theta, x_i}\!\left(\tfrac{1 - \alpha}{2}\right),
                   F^{-1}_{\theta, x_i}\!\left(\tfrac{1 + \alpha}{2}\right) \right]
            \right],

        where :math:`F^{-1}_{\theta, x_i}` is the predictive inverse CDF (computed via
        the distribution's ``icdf`` method). The Quantile Calibration Error then
        aggregates the gap :math:`|\hat{c}(\alpha) - \alpha|` over a regular grid of
        :attr:`num_bins` confidence levels using the chosen :attr:`norm` — the regression
        counterpart of the :class:`~torch_uncertainty.metrics.classification.CalibrationError`.

        Args:
            num_bins: Number of bins to use for calibration. Defaults to ``15``.
            norm: Norm to use for calibration error computation. Defaults to ``"l1"``.
            ignore_index: Index to ignore during calibration. Defaults to ``None``.
            validate_args: Whether to validate the input arguments. Defaults to ``True``.
            kwargs: Additional keyword arguments, see `Advanced metric settings
              <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.
        """
        super().__init__(num_bins, norm, ignore_index, validate_args, **kwargs)
        self.conf_intervals = torch.linspace(0.05, 0.95, self.n_bins + 1)

    def update(  # pyrefly: ignore[bad-override]
        self,
        dist: Distribution,
        target: Tensor,
        padding_mask: Tensor | None = None,
    ) -> None:
        """Update the metric with new predictions and targets.

        Args:
            dist: The predicted distribution.
            target: The ground truth values.
            padding_mask: A mask to ignore certain values. Defaults to ``None``.
        """
        reduce_event_dims = False
        if isinstance(dist, Independent):
            iid_dist = dist.base_dist
            reduce_event_dims = True
        else:
            iid_dist = dist

        try:
            iid_dist.icdf((1 - self.conf_intervals[0]) / 2)

        except NotImplementedError:
            warnings.warn(
                "The distribution does not support the `icdf()` method. "
                "This metric will therefore return `nan` values. "
                "Please use a distribution that implements `icdf()`.",
                UserWarning,
                stacklevel=2,
            )
            self.not_implemented_error = True
            return

        confidences = self.conf_intervals.expand(*dist.batch_shape, -1)
        correct_mask = torch.empty_like(confidences)

        for i, conf in enumerate(self.conf_intervals):
            b_min = iid_dist.icdf((1 - conf) / 2)
            bound_log_prob = iid_dist.log_prob(b_min)
            target_log_prob = dist.log_prob(target)
            if reduce_event_dims:
                indep_dist = cast("Independent", dist)
                bound_log_prob = bound_log_prob.sum(
                    dim=list(range(-indep_dist.reinterpreted_batch_ndims, 0))
                )

            correct_mask[..., i] = (bound_log_prob <= target_log_prob).float()

        if padding_mask is not None:
            confidences = confidences[~padding_mask]
            correct_mask = correct_mask[~padding_mask]

        super().update(confidences.flatten(), correct_mask.flatten())

    def compute(self) -> Tensor:
        """Compute the quantile calibration error.

        Returns:
            Tensor: The quantile calibration error.

        Warning:
            If the distribution does not support ``icdf()``, this returns ``nan`` values.
        """
        if self.not_implemented_error:
            return torch.tensor(float("nan"))
        return super().compute()

    def plot(self) -> _PLOT_OUT_TYPE:  # pyrefly: ignore[bad-override]
        """Plot the quantile calibration reliability diagram.

        Raises:
            NotImplementedError: If the distribution does not support ``icdf()``.
        """
        if self.not_implemented_error:
            raise NotImplementedError(
                "The distribution does not support the `icdf()` method. "
                "This metric will therefore return `nan` values. "
                "Please use a distribution that implements `icdf()`."
            )

        confidences = dim_zero_cat(self.confidences)
        accuracies = dim_zero_cat(self.accuracies)

        bin_boundaries = torch.linspace(
            0,
            1,
            self.n_bins + 1,
            dtype=torch.float,
            device=confidences.device,
        )

        with torch.no_grad():
            acc_bin, conf_bin, prop_bin = _binning_bucketize(
                confidences, accuracies, bin_boundaries
            )

        np_acc_bin = acc_bin.cpu().numpy()
        np_conf_bin = conf_bin.cpu().numpy()
        np_prop_bin = prop_bin.cpu().numpy()
        np_bin_boundaries = bin_boundaries.cpu().numpy()

        return reliability_chart(
            accuracies=accuracies.cpu().numpy(),
            confidences=confidences.cpu().numpy(),
            bin_accuracies=np_acc_bin,
            bin_confidences=np_conf_bin,
            bin_sizes=np_prop_bin,
            bins=np_bin_boundaries,
        )
