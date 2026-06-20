import math
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.compute import _auc_compute
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.plot import _AX_TYPE


class AURC(Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    scores: list[Tensor]
    errors: list[Tensor]

    def __init__(self, **kwargs) -> None:
        r"""Calculate Area Under the Risk-Coverage curve.

        The Area Under the Risk-Coverage curve (AURC) is the main metric for
        Selective Classification (SC) performance assessment. It evaluates the
        quality of uncertainty estimates by measuring the ability to
        discriminate between correct and incorrect predictions based on their
        rank (and not their values in contrast with calibration).

        Let :math:`\sigma` be the permutation sorting the :math:`N` samples by
        descending top-class confidence, so that
        :math:`\hat{p}_{\sigma(1)} \geq \cdots \geq \hat{p}_{\sigma(N)}`.
        The error rate at coverage :math:`\kappa = k/N` is

        .. math::

            r\!\left(\tfrac{k}{N}\right) = \frac{1}{k}
            \sum_{i=1}^{k} \mathbf{1}\!\left[\hat{y}_{\sigma(i)} \neq y_{\sigma(i)}\right]

        and the AURC is

        .. math::

            \text{AURC} = \int_0^1 r(\kappa)\,\mathrm{d}\kappa
            \approx \frac{1}{N} \sum_{k=1}^{N} r\!\left(\tfrac{k}{N}\right)

        As input to ``forward`` and ``update`` the metric accepts the following input:

        - **preds** (:class:`~torch.Tensor`): A float tensor of shape
          ``(N, ...)`` containing probabilities for each observation.
        - **target** (:class:`~torch.Tensor`): An int tensor of shape
          ``(N, ...)`` containing ground-truth labels.

        As output to ``forward`` and ``compute`` the metric returns the
        following output:

        - **Aurc** (:class:`~torch.Tensor`): A scalar tensor containing the
          area under the risk-coverage curve

        Args:
            kwargs: Additional keyword arguments.

        Example:
            >>> from torch_uncertainty.metrics.classification import AURC
            >>> aurc = AURC()
            >>> probs = torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.8, 0.2]])
            >>> targets = torch.tensor([0, 1, 0])
            >>> aurc.update(probs, targets)
            >>> result = aurc.compute()
            >>> print(result)
            tensor(0.0833)  # Example output

        References:
            [1] `Geifman & El-Yaniv. Selective classification for deep neural networks. In NeurIPS, 2017 <https://papers.nips.cc/paper_files/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf>`_.
        """
        super().__init__(**kwargs)
        self.add_state("scores", default=[], dist_reduce_fx="cat")
        self.add_state("errors", default=[], dist_reduce_fx="cat")

    def update(self, probs: Tensor, targets: Tensor) -> None:  # pyrefly: ignore[bad-override]
        """Store the scores and their associated errors for later computation.

        Args:
            probs: The predicted probabilities of shape :math:`(N, C)`.
            targets: The ground truth labels of shape :math:`(N,)`.
        """
        if probs.ndim == 1:
            probs = torch.stack([1 - probs, probs], dim=-1)
        self.scores.append(probs.max(-1).values)
        self.errors.append((probs.argmax(-1) != targets) * 1.0)

    def partial_compute(self) -> Tensor:
        """Compute the error and optimal error rates for the RC curve.

        Returns:
            Tensor: The error rates and the optimal/oracle error
                rates.
        """
        scores = dim_zero_cat(self.scores)
        errors = dim_zero_cat(self.errors)
        return _aurc_rejection_rate_compute(scores, errors)

    def compute(self) -> Tensor:
        r"""Compute the Area Under the Risk-Coverage curve (AURC).

        Note:
            Normalize the AURC as if its support was between 0 and 1.
            This has an impact on the AURC when the number of samples is small.

        Returns:
            Tensor: The AURC.
        """
        error_rates = self.partial_compute()
        num_samples = error_rates.size(0)
        if num_samples < 2:
            return torch.tensor([float("nan")], device=self.device)
        # There is no error rate associated to 0 coverage: starting at 1
        cov = torch.arange(1, num_samples + 1, device=self.device) / num_samples
        return _auc_compute(cov, error_rates) / (1 - 1 / num_samples)

    def plot(  # pyrefly: ignore[bad-override]
        self,
        ax: _AX_TYPE | None = None,
        plot_value: bool = True,
        name: str | None = None,
    ) -> tuple[plt.Figure | None, plt.Axes]:
        """Plot the risk-cov. curve corresponding to the inputs passed to
        ``update``.

        Args:
            ax: A matplotlib axis object. If provided, the plot is added to this axis. Defaults to ``None``.
            plot_value: Whether to print the AURC value on the plot. Defaults to ``True``.
            name: Name of the model. Defaults to ``None``.

        Returns:
            tuple[Figure | None, Axes]: Figure object and axes object.
        """
        fig, ax = plt.subplots(figsize=(6, 6)) if ax is None else (None, ax)
        ax = cast("plt.Axes", ax)

        # Computation of AURC
        error_rates = self.partial_compute().cpu().flip(0)
        num_samples = error_rates.size(0)

        x = torch.arange(1, num_samples + 1) / num_samples
        aurc = _auc_compute(x, error_rates).cpu().item()

        # reduce plot size
        plot_xs = np.arange(0.01, 100 + 0.01, 0.01)
        xs = np.arange(start=1, stop=num_samples + 1) / num_samples

        rejection_rates = np.interp(plot_xs, xs, x * 100)
        error_rates = np.interp(plot_xs, xs, error_rates)

        # plot
        ax.plot(
            100 - rejection_rates,
            error_rates * 100,
            label="Model" if name is None else name,
        )

        if plot_value:
            ax.text(
                0.02,
                0.95,
                f"AURC={aurc:.2%}",
                color="black",
                ha="left",
                va="bottom",
                transform=ax.transAxes,
            )
        plt.grid(True, linestyle="--", alpha=0.7, zorder=0)
        ax.set_xlabel("Coverage (%)", fontsize=16)
        ax.set_ylabel("Risk - Error Rate (%)", fontsize=16)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, min(100, np.ceil(error_rates.max() * 100)))
        ax.legend(loc="upper right")
        return fig, ax


def _aurc_rejection_rate_compute(
    scores: Tensor,
    errors: Tensor,
) -> Tensor:
    """Compute the cumulative error rates for a given set of scores and errors.

    Args:
        scores: uncertainty scores of shape :math:`(B,)`
        errors: binary errors of shape :math:`(B,)`
    """
    errors = errors[scores.argsort(descending=True)]
    return errors.cumsum(dim=-1) / torch.arange(
        1, scores.size(0) + 1, dtype=scores.dtype, device=scores.device
    )


class AUGRC(AURC):
    def __init__(self, **kwargs) -> None:
        r"""Calculate The Area Under the Generalized Risk-Coverage curve (AUGRC).

        The Area Under the Generalized Risk-Coverage curve (AUGRC) for selective classification
        performance assessment. Unlike AURC, it weights the error rate at each coverage level
        by the coverage itself, reducing the influence of the most confident samples and
        making the metric more sensitive to performance at intermediate coverage levels.

        Using the same notation as :class:`AURC`, the AUGRC is defined as

        .. math::

            \text{AUGRC} = \int_0^1 \kappa \cdot r(\kappa)\,\mathrm{d}\kappa
            \approx \frac{1}{N} \sum_{k=1}^{N} \frac{k}{N} \cdot r\!\left(\tfrac{k}{N}\right)

        As input to ``forward`` and ``update`` the metric accepts the following input:

        - **preds** (:class:`~torch.Tensor`): A float tensor of shape
          ``(N, ...)`` containing probabilities for each observation.
        - **target** (:class:`~torch.Tensor`): An int tensor of shape
          ``(N, ...)`` containing ground-truth labels.

        As output to ``forward`` and ``compute`` the metric returns the
        following output:

        - **Augrc** (:class:`~torch.Tensor`): A scalar tensor containing the
          area under the risk-coverage curve

        Args:
            kwargs: Additional keyword arguments.

        References:
            [1] `Traub et al. Overcoming Common Flaws in the Evaluation of Selective Classification Systems <https://arxiv.org/pdf/2407.01032>`_.

        .. seealso::
            - :class:`~torch_uncertainty.metrics.classification.AURC` : Parent class, the AURC metric
        """
        super().__init__(**kwargs)

    def compute(self) -> Tensor:
        """Normalize the AUGRC as if its support was between 0 and 1. This has an
        impact on the AUGRC when the number of samples is small.

        Returns:
            Tensor: The AUGRC.
        """
        error_rates = self.partial_compute()
        num_samples = error_rates.size(0)
        if num_samples < 2:
            return torch.tensor([float("nan")], device=self.device)
        cov = torch.arange(1, num_samples + 1, device=self.device) / num_samples
        return _auc_compute(cov, error_rates * cov) / (1 - 1 / num_samples)

    def plot(
        self,
        ax: _AX_TYPE | None = None,
        plot_value: bool = True,
        name: str | None = None,
    ) -> tuple[plt.Figure | None, plt.Axes]:
        """Plot the generalized risk-cov. curve corresponding to the inputs passed to
        ``update``.

        Args:
            ax: A matplotlib axis object. If provided, the plot is added to this axis. Defaults to ``None``.
            plot_value: Whether to print the AURC value on the plot. Defaults to ``True``.
            name: Name of the model. Defaults to ``None``.

        Returns:
            tuple[Figure | None, Axes]: Figure object and axes object.
        """
        fig, ax = plt.subplots(figsize=(6, 6)) if ax is None else (None, ax)
        ax = cast("plt.Axes", ax)

        # Computation of AUGRC
        error_rates = self.partial_compute().cpu().flip(0)
        num_samples = error_rates.size(0)
        cov = torch.arange(num_samples) / num_samples

        augrc = _auc_compute(cov, error_rates * cov).cpu().item()

        # reduce plot size
        plot_covs = np.arange(0.01, 100 + 0.01, 0.01)
        covs = np.arange(start=1, stop=num_samples + 1) / num_samples

        rejection_rates = np.interp(plot_covs, covs, cov * 100)
        error_rates = np.interp(plot_covs, covs, error_rates * covs[::-1] * 100)

        # plot
        ax.plot(
            100 - rejection_rates,
            error_rates,
            label="Model" if name is None else name,
        )

        if plot_value:
            ax.text(
                0.02,
                0.95,
                f"AUGRC={augrc:.2%}",
                color="black",
                ha="left",
                va="bottom",
                transform=ax.transAxes,
            )
        plt.grid(True, linestyle="--", alpha=0.7, zorder=0)
        ax.set_xlabel("Coverage (%)", fontsize=16)
        ax.set_ylabel("Generalized Risk (%)", fontsize=16)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, min(100, np.ceil(error_rates.max() * 100)))
        ax.legend(loc="upper right")
        return fig, ax


class CovAtxRisk(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    scores: list[Tensor]
    errors: list[Tensor]

    def __init__(self, risk_threshold: float, **kwargs) -> None:
        r"""Provide coverage at x Risk.

        If there are multiple coverage values corresponding to the given risk,
        i.e., the risk(coverage) is not monotonic, the coverage at x risk is
        the maximum coverage value corresponding to the given risk. If no
        there is no coverage value corresponding to the given risk, this metric returns ``float("nan")``.

        Args:
            risk_threshold: The risk threshold at which to compute the coverage.
            kwargs: Additional arguments to pass to the metric class.

        Example:

        .. code-block:: python

            from torch_uncertainty.metrics.classification import CovAtxRisk

            # Define a more diverse dataset with probabilities and targets
            probs = torch.tensor(
                [
                    [0.9, 0.1],  # Correct prediction (confidence 0.9)
                    [0.6, 0.4],  # Incorrect prediction (confidence 0.6)
                    [0.8, 0.2],  # Correct prediction (confidence 0.8)
                    [0.5, 0.5],  # Incorrect prediction (confidence 0.5)
                    [0.7, 0.3],  # Correct prediction (confidence 0.7)
                ]
            )
            targets = torch.tensor([0, 1, 0, 1, 0])  # Ground truth labels

            # Instantiate the CovAtxRisk metric with a risk threshold
            metric = CovAtxRisk(risk_threshold=0.3)
            metric.update(probs, targets)
            coverage_at_risk = metric.compute()

            print(f"Coverage at risk: {coverage_at_risk.item()}")
            # tensor(0.800000011920929)

        """
        super().__init__(**kwargs)
        self.add_state("scores", default=[], dist_reduce_fx="cat")
        self.add_state("errors", default=[], dist_reduce_fx="cat")
        _risk_coverage_checks(risk_threshold)
        self.risk_threshold = risk_threshold

    def update(self, probs: Tensor, targets: Tensor) -> None:  # pyrefly: ignore[bad-override]
        """Store the scores and their associated errors for later computation.

        Args:
            probs: The predicted probabilities of shape :math:`(N, C)`.
            targets: The ground truth labels of shape :math:`(N,)`.
        """
        if probs.ndim == 1:
            probs = torch.stack([1 - probs, probs], dim=-1)
        self.scores.append(probs.max(-1).values)
        self.errors.append((probs.argmax(-1) != targets) * 1.0)

    def compute(self) -> Tensor:
        """Compute the coverage at x Risk.

        Returns:
            Tensor: The coverage at x risk.
        """
        scores = dim_zero_cat(self.scores)
        errors = dim_zero_cat(self.errors)
        num_samples = scores.size(0)
        if num_samples < 1:
            return torch.tensor([float("nan")], device=self.device)
        error_rates = _aurc_rejection_rate_compute(scores, errors)
        admissible_risks = (error_rates > self.risk_threshold) * 1
        max_cov_at_risk = admissible_risks.flip(0).argmin()

        # check if max_cov_at_risk is really admissible, if not return nan
        risk = admissible_risks[max_cov_at_risk]
        if risk > self.risk_threshold:
            return torch.tensor([float("nan")], device=self.device)
        return 1 - max_cov_at_risk / num_samples


class CovAt5Risk(CovAtxRisk):
    def __init__(self, **kwargs) -> None:
        r"""Provide coverage at 5% Risk.

        If there are multiple coverage values corresponding to 5% risk, the
        coverage at 5% risk is the maximum coverage value corresponding to 5%
        risk. If there is no coverage value corresponding to the given risk,
        this metric returns ``float("nan")``.

        This is a specific case of the more general :class:`CovAtxRisk` metric, where the risk level is fixed at 5%.

        .. seealso::
            - :class:`CovAtxRisk` : Parent class, the CovAtxRisk metric
        """
        super().__init__(risk_threshold=0.05, **kwargs)


class RiskAtxCov(Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    scores: list[Tensor]
    errors: list[Tensor]

    def __init__(self, cov_threshold: float, **kwargs) -> None:
        r"""Compute the risk at a specified coverage threshold.

        This metric calculates the error rate (risk) at a given coverage level.
        The coverage threshold determines the fraction of samples considered,
        sorted by model confidence. The metric is useful in evaluating the
        trade-off between coverage and risk in predictive models.

        Args:
            cov_threshold: The coverage threshold at which to compute the risk.
            kwargs: Additional arguments to pass to the metric class.

        Example:

        .. code-block:: python

            from torch_uncertainty.metrics.classification import RiskAtxCov

            # Initialize the metric with a coverage threshold of 0.5 (50%)
            metric = RiskAtxCov(cov_threshold=0.5)

            # Simulated predicted probabilities (N samples, C classes)
            predicted_probs = torch.tensor(
                [
                    [0.9, 0.1],  # Correct (class 0)
                    [0.7, 0.3],  # Incorrect (class 1)
                    [0.95, 0.05],  # Correct (class 0)
                    [0.8, 0.2],  # Incorrect (class 1)
                    [0.6, 0.4],  # Correct (class 0)
                    [0.3, 0.7],  # Correct (class 1)
                    [0.85, 0.15],  # Incorrect (class 1)
                    [0.2, 0.8],  # Correct (class 1)
                ]
            )

            # Simulated ground truth labels
            ground_truth = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

            # Update the metric with the probabilities and labels
            metric.update(predicted_probs, ground_truth)

            # Compute the risk at the specified coverage threshold
            risk_at_cov = metric.compute()

            # Output the result
            print(f"Risk at coverage threshold: {risk_at_cov.item():.2f}")

            # output : Risk at coverage threshold: 0.25
        """
        super().__init__(**kwargs)
        self.add_state("scores", default=[], dist_reduce_fx="cat")
        self.add_state("errors", default=[], dist_reduce_fx="cat")
        _risk_coverage_checks(cov_threshold)
        self.cov_threshold = cov_threshold

    def update(self, probs: Tensor, targets: Tensor) -> None:  # pyrefly: ignore[bad-override]
        """Store the scores and their associated errors for later computation.

        Args:
            probs: The predicted probabilities of shape :math:`(N, C)`.
            targets: The ground truth labels of shape :math:`(N,)`.
        """
        if probs.ndim == 1:
            probs = torch.stack([1 - probs, probs], dim=-1)
        self.scores.append(probs.max(-1).values)
        self.errors.append((probs.argmax(-1) != targets) * 1.0)

    def compute(self) -> Tensor:
        """Compute the risk at given coverage.

        Returns:
            Tensor: The risk at given coverage.
        """
        scores = dim_zero_cat(self.scores)
        errors = dim_zero_cat(self.errors)
        error_rates = _aurc_rejection_rate_compute(scores, errors)
        return error_rates[math.ceil(scores.size(0) * self.cov_threshold) - 1]


class RiskAt80Cov(RiskAtxCov):
    def __init__(self, **kwargs) -> None:
        r"""Compute the risk at 80% coverage.

        The risk at 80% coverage is the error rate among the 80% most confident
        predictions. Samples are ranked by descending top-class confidence and
        the error rate is computed over the top 80%.

        This is a specific case of the more general :class:`RiskAtxCov` metric,
        where the coverage level is fixed at 80%.

        .. seealso::
            - :class:`RiskAtxCov` : Parent class, the RiskAtxCov metric
        """
        super().__init__(cov_threshold=0.8, **kwargs)


def _risk_coverage_checks(threshold: float) -> None:
    if not isinstance(threshold, float):
        raise TypeError(f"Expected threshold to be of type float, but got {type(threshold)}")
    if threshold < 0 or threshold > 1:
        raise ValueError(f"Threshold should be in the range [0, 1], but got {threshold}.")
