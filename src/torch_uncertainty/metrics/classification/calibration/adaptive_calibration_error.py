from typing import Any, Literal

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.classification.calibration_error import (
    _binary_calibration_error_arg_validation,
    _multiclass_calibration_error_arg_validation,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.enums import ClassificationTaskNoMultilabel


def _equal_binning_bucketize(
    confidences: Tensor, accuracies: Tensor, num_bins: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute bins for the adaptive calibration error.

    Args:
        confidences: The confidence (i.e. predicted prob) of the top-1 prediction.
        accuracies: 1.0 if the top-1 prediction was correct, 0.0 otherwise.
        num_bins: Number of bins to use when computing adaptive calibration error.

    Returns:
        tuple with binned accuracy, binned confidence and binned probabilities
    """
    confidences, indices = torch.sort(confidences)
    accuracies = accuracies[indices]
    acc_bin, conf_bin = (
        list(accuracies.tensor_split(num_bins)),
        list(confidences.tensor_split(num_bins)),
    )
    count_bin = torch.as_tensor(
        [len(cb) for cb in conf_bin],
        dtype=confidences.dtype,
        device=confidences.device,
    )
    return (
        pad_sequence(acc_bin, batch_first=True).sum(1) / count_bin,
        pad_sequence(conf_bin, batch_first=True).sum(1) / count_bin,
        torch.as_tensor(count_bin) / len(confidences),
    )


def _equal_binning_bucketize_with_bounds(
    confidences: Tensor, accuracies: Tensor, num_bins: int
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Compute adaptive bins and their confidence bounds.

    Returns:
        tuple of (acc_bin, conf_bin, prop_bin, bin_lowers, bin_uppers) where
        bin_lowers/bin_uppers are the min/max confidence in each bin.
    """
    confidences_sorted, indices = torch.sort(confidences)
    accuracies_sorted = accuracies[indices]

    conf_splits = list(confidences_sorted.tensor_split(num_bins))
    acc_splits = list(accuracies_sorted.tensor_split(num_bins))

    # Drop empty splits (can occur when num_bins > len(confidences))
    pairs = [(a, c) for a, c in zip(acc_splits, conf_splits, strict=True) if len(c) > 0]
    acc_splits = [a for a, _ in pairs]
    conf_splits = [c for _, c in pairs]

    count_bin = torch.as_tensor(
        [len(cb) for cb in conf_splits],
        dtype=confidences.dtype,
        device=confidences.device,
    )
    bin_lowers = torch.stack([cb[0] for cb in conf_splits])
    bin_uppers = torch.stack([cb[-1] for cb in conf_splits])

    return (
        pad_sequence(acc_splits, batch_first=True).sum(1) / count_bin,
        pad_sequence(conf_splits, batch_first=True).sum(1) / count_bin,
        count_bin / len(confidences),
        bin_lowers,
        bin_uppers,
    )


def _ace_compute(
    confidences: Tensor,
    accuracies: Tensor,
    num_bins: int,
    norm: Literal["l1", "l2", "max"] = "l1",
    debias: bool = False,
) -> Tensor:
    """Compute the adaptive calibration error given the provided number of bins and norm.

    Args:
        confidences: The confidence (i.e. predicted prob) of the top-1 prediction.
        accuracies: 1.0 if the top-1 prediction was correct, 0.0 otherwise.
        num_bins: Number of bins to use when computing adaptive calibration error.
        norm: Norm function to use when computing calibration error. Defaults to ``"l1"``.
        debias: Apply debiasing to L2 norm computation as in `Verified Uncertainty Calibration`.
            Defaults to ``False``.

    Returns:
        Tensor: Adaptive Calibration error scalar.
    """
    with torch.no_grad():
        acc_bin, conf_bin, prop_bin = _equal_binning_bucketize(confidences, accuracies, num_bins)

    if norm == "l1":
        return torch.sum(torch.abs(acc_bin - conf_bin) * prop_bin)
    if norm == "l2":
        ace = torch.sum(torch.pow(acc_bin - conf_bin, 2) * prop_bin)
        if debias:  # coverage: ignore
            debias_bins = (acc_bin * (acc_bin - 1) * prop_bin) / (
                prop_bin * accuracies.size()[0] - 1
            )
            ace += torch.sum(
                torch.nan_to_num(debias_bins)
            )  # replace nans with zeros if nothing appeared in a bin
        return torch.sqrt(ace) if ace > 0 else torch.tensor(0)
    if norm == "max":
        return torch.max(torch.abs(acc_bin - conf_bin))
    raise ValueError(f"Unexpected norm. Got {norm}.")


def _adaptive_reliability_diagram_subplot(
    ax,
    bin_accuracies: np.ndarray,
    bin_confidences: np.ndarray,
    bin_sizes: np.ndarray,
    bin_lowers: np.ndarray,
    bin_uppers: np.ndarray,
    norm: str | None,
    ace_value: float | None,
    title: str = "Adaptive Reliability Diagram",
    xlabel: str = "Top-class Confidence (%)",
    ylabel: str = "Success Rate (%)",
) -> None:
    widths = (bin_uppers - bin_lowers) * 100
    centers = (bin_lowers + bin_uppers) / 2 * 100

    # Normalize alpha by max bin size so equal-count bins appear fully opaque
    max_size = bin_sizes.max()
    alphas = 0.2 + 0.8 * (bin_sizes / max_size if max_size > 0 else np.ones_like(bin_sizes))

    colors = np.zeros((len(bin_sizes), 4))
    colors[:, 0] = 240 / 255.0
    colors[:, 1] = 60 / 255.0
    colors[:, 2] = 60 / 255.0
    colors[:, 3] = alphas

    gap_plt = ax.bar(
        centers,
        np.abs(bin_accuracies - bin_confidences) * 100,
        bottom=np.minimum(bin_accuracies, bin_confidences) * 100,
        width=widths,
        edgecolor=colors,
        color=colors,
        linewidth=1,
        label="Gap",
    )

    acc_plt = ax.bar(
        centers,
        0,
        bottom=bin_accuracies * 100,
        width=widths,
        edgecolor="black",
        color="black",
        alpha=1.0,
        linewidth=2,
        label="Accuracy",
    )

    ax.set_aspect("equal")
    ax.plot([0, 100], [0, 100], linestyle="--", color="gray")

    if norm is not None and ace_value is not None:
        ax.text(
            0.98,
            0.02,
            f"ACE (norm: {norm}) = {ace_value:.02%}",
            color="black",
            ha="right",
            va="bottom",
            transform=ax.transAxes,
        )

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, linestyle="--", zorder=0)
    ax.legend(handles=[gap_plt, acc_plt])


def _adaptive_confidence_histogram_subplot(
    ax,
    accuracies: np.ndarray,
    confidences: np.ndarray,
    bin_lowers: np.ndarray,
    bin_uppers: np.ndarray,
    title: str = "",
    xlabel: str = "Top-class Confidence (%)",
    ylabel: str = "Density (%)",
) -> None:
    sns.kdeplot(
        confidences * 100,
        linewidth=2,
        ax=ax,
        fill=True,
        alpha=0.5,
    )

    # Draw all unique bin boundaries to reveal the adaptive bin structure
    all_bounds = np.unique(np.concatenate([bin_lowers, bin_uppers])) * 100
    for boundary in all_bounds:
        ax.axvline(x=boundary, color="steelblue", linestyle=":", alpha=0.6, linewidth=1.0)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, None)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    avg_acc = np.mean(accuracies)
    avg_conf = np.mean(confidences)

    acc_plt = ax.axvline(x=avg_acc * 100, ls="solid", lw=2, c="black", label="Accuracy")
    conf_plt = ax.axvline(x=avg_conf * 100, ls="dotted", lw=2, c="#444", label="Avg. confidence")
    ax.grid(True, alpha=0.3, linestyle="--", zorder=0)
    ax.legend(handles=[acc_plt, conf_plt], loc="upper left")


def adaptive_reliability_chart(
    accuracies: np.ndarray,
    confidences: np.ndarray,
    bin_accuracies: np.ndarray,
    bin_confidences: np.ndarray,
    bin_sizes: np.ndarray,
    bin_lowers: np.ndarray,
    bin_uppers: np.ndarray,
    norm: str | None = None,
    ace_value: float | None = None,
    title: str = "Adaptive Reliability Diagram",
    rd_xlabel: str = "Top-class Confidence (%)",
    rd_ylabel: str = "Success Rate (%)",
    ch_xlabel: str = "Top-class Confidence (%)",
    ch_ylabel: str = "Density (%)",
    figsize: tuple[float, float] = (6.0, 6.0),
    dpi: int = 150,
) -> tuple[object, object]:
    """Build an Adaptive Reliability Diagram with variable-width bins.

    Unlike the standard reliability diagram, bin widths reflect the actual
    span of confidence values in each equal-count bin, making clustered
    predictions visually distinct from spread-out ones.
    """
    figsize = (figsize[0], figsize[0] * 1.4)

    fig, ax = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=figsize,
        dpi=dpi,
        gridspec_kw={"height_ratios": [4, 1]},
    )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0)

    _adaptive_reliability_diagram_subplot(
        ax[0],
        bin_accuracies,
        bin_confidences,
        bin_sizes,
        bin_lowers,
        bin_uppers,
        norm=norm,
        ace_value=ace_value,
        title=title,
        xlabel=rd_xlabel,
        ylabel=rd_ylabel,
    )

    _adaptive_confidence_histogram_subplot(
        ax[1],
        accuracies,
        confidences,
        bin_lowers,
        bin_uppers,
        title="",
        xlabel=ch_xlabel,
        ylabel=ch_ylabel,
    )
    ax[1].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    return fig, ax


def _adaptive_custom_plot(
    self,
    title: str = "Adaptive Reliability Diagram",
    rd_xlabel: str = "Top-class Confidence (%)",
    rd_ylabel: str = "Success Rate (%)",
    ch_xlabel: str = "Top-class Confidence (%)",
    ch_ylabel: str = "Density (%)",
) -> tuple[object, object]:
    confidences = dim_zero_cat(self.confidences)
    accuracies = dim_zero_cat(self.accuracies)

    ace_value = _ace_compute(confidences, accuracies, num_bins=self.n_bins, norm=self.norm).item()
    with torch.no_grad():
        acc_bin, conf_bin, prop_bin, bin_lowers, bin_uppers = _equal_binning_bucketize_with_bounds(
            confidences, accuracies, self.n_bins
        )

    return adaptive_reliability_chart(
        accuracies=accuracies.cpu().numpy(),
        confidences=confidences.cpu().numpy(),
        bin_accuracies=acc_bin.cpu().numpy(),
        bin_confidences=conf_bin.cpu().numpy(),
        bin_sizes=prop_bin.cpu().numpy(),
        bin_lowers=bin_lowers.cpu().numpy(),
        bin_uppers=bin_uppers.cpu().numpy(),
        title=title,
        rd_xlabel=rd_xlabel,
        rd_ylabel=rd_ylabel,
        ch_xlabel=ch_xlabel,
        ch_ylabel=ch_ylabel,
        ace_value=ace_value,
        norm=self.norm,
    )


class BinaryAdaptiveCalibrationError(Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    confidences: list[Tensor]
    accuracies: list[Tensor]

    def __init__(
        self,
        n_bins: int = 10,
        norm: Literal["l1", "l2", "max"] = "l1",
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        r"""Adaptive Top-label Calibration Error for binary tasks.

        Args:
            n_bins: Number of bins to use when computing the calibration error. Defaults to ``10``.
            norm: Norm function to use when computing calibration error. Defaults to ``"l1"``.
            ignore_index: Index to ignore during calculations. Defaults to ``None``.
            validate_args: Whether to validate input arguments. Defaults to ``True``.
            kwargs: Additional keyword arguments passed to the parent metric.
        """
        super().__init__(**kwargs)
        if ignore_index is not None:  # coverage: ignore
            raise ValueError("ignore_index is not supported for multiclass tasks.")

        if validate_args:
            _binary_calibration_error_arg_validation(n_bins, norm, ignore_index)
        self.n_bins = n_bins
        self.norm = norm

        self.add_state("confidences", [], dist_reduce_fx="cat")
        self.add_state("accuracies", [], dist_reduce_fx="cat")

    def update(self, probs: Tensor, targets: Tensor) -> None:  # pyrefly: ignore[bad-override]
        """Update metric states with predictions and targets."""
        confidences, preds = torch.max(probs, 1 - probs), torch.round(probs)
        accuracies = preds == targets
        self.confidences.append(confidences.float())
        self.accuracies.append(accuracies.float())

    def compute(self) -> Tensor:
        """Compute metric."""
        confidences = dim_zero_cat(self.confidences)
        accuracies = dim_zero_cat(self.accuracies)
        return _ace_compute(confidences, accuracies, self.n_bins, norm=self.norm)

    def plot(
        self,
        title: str = "Adaptive Reliability Diagram",
        rd_xlabel: str = "Top-class Confidence (%)",
        rd_ylabel: str = "Success Rate (%)",
        ch_xlabel: str = "Top-class Confidence (%)",
        ch_ylabel: str = "Density (%)",
    ) -> tuple[object, object]:
        return _adaptive_custom_plot(self, title, rd_xlabel, rd_ylabel, ch_xlabel, ch_ylabel)


class MulticlassAdaptiveCalibrationError(Metric):
    is_differentiable: bool | None = False
    higher_is_better: bool | None = False
    full_state_update: bool | None = False

    confidences: list[Tensor]
    accuracies: list[Tensor]

    def __init__(
        self,
        num_classes: int,
        n_bins: int = 10,
        norm: Literal["l1", "l2", "max"] = "l1",
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        r"""Adaptive Top-label Calibration Error for multiclass tasks.

        Args:
            num_classes: Number of classes.
            n_bins: Number of bins to use when computing the calibration error. Defaults to ``10``.
            norm: Norm function to use when computing calibration error. Defaults to ``"l1"``.
            ignore_index: Index to ignore during calculations. Defaults to ``None``.
            validate_args: Whether to validate input arguments. Defaults to ``True``.
            kwargs: Additional keyword arguments passed to the parent metric.
        """
        super().__init__(**kwargs)
        if ignore_index is not None:  # coverage: ignore
            raise ValueError("ignore_index is not supported for multiclass tasks.")

        if validate_args:
            _multiclass_calibration_error_arg_validation(num_classes, n_bins, norm, ignore_index)
        self.n_bins = n_bins
        self.norm = norm

        self.add_state("confidences", [], dist_reduce_fx="cat")
        self.add_state("accuracies", [], dist_reduce_fx="cat")

    def update(self, probs: Tensor, targets: Tensor) -> None:  # pyrefly: ignore[bad-override]
        """Update metric states with predictions and targets."""
        confidences, preds = torch.max(probs, 1)
        accuracies = preds == targets
        self.confidences.append(confidences.float())
        self.accuracies.append(accuracies.float())

    def compute(self) -> Tensor:
        """Compute metric."""
        confidences = dim_zero_cat(self.confidences)
        accuracies = dim_zero_cat(self.accuracies)
        return _ace_compute(confidences, accuracies, self.n_bins, norm=self.norm)

    def plot(
        self,
        title: str = "Adaptive Reliability Diagram",
        rd_xlabel: str = "Top-class Confidence (%)",
        rd_ylabel: str = "Success Rate (%)",
        ch_xlabel: str = "Top-class Confidence (%)",
        ch_ylabel: str = "Density (%)",
    ) -> tuple[object, object]:
        return _adaptive_custom_plot(self, title, rd_xlabel, rd_ylabel, ch_xlabel, ch_ylabel)


class AdaptiveCalibrationError:
    def __new__(
        cls,
        task: Literal["binary", "multiclass"],
        num_bins: int = 10,
        norm: Literal["l1", "l2", "max"] = "l1",
        num_classes: int | None = None,
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        r"""Computes the Adaptive Top-label Calibration Error (ACE) for classification tasks.

        The Adaptive Calibration Error is a metric designed to measure the calibration
        of predicted probabilities by dividing the probability space into bins that adapt
        to the distribution of predicted probabilities. Unlike uniform binning, adaptive binning
        ensures a more balanced representation of predictions across bins.

        Given top-class confidences :math:`\hat{p}_i` and accuracies
        :math:`a_i = \mathbf{1}[\hat{y}_i = y_i]`, the :math:`N` samples are sorted by
        confidence and split into :math:`M` bins :math:`B_1, \dots, B_M` each containing
        (approximately) the same number of samples. Three norms are available:

        **Adaptive Calibration Error (ACE):**

        .. math::

            \text{ACE} = \sum_{m=1}^{M} \frac{|B_m|}{N}
            \left| \operatorname{acc}(B_m) - \operatorname{conf}(B_m) \right|

        **Maximum Adaptive Calibration Error (MACE):**

        .. math::

            \text{MACE} = \max_{m} \left| \operatorname{acc}(B_m) -
            \operatorname{conf}(B_m) \right|

        **Root Mean Square Adaptive Calibration Error (RMACE):**

        .. math::

            \text{RMACE} = \sqrt{\sum_{m=1}^{M} \frac{|B_m|}{N}
            \left( \operatorname{acc}(B_m) - \operatorname{conf}(B_m) \right)^2}

        where :math:`\operatorname{acc}(B_m) = \tfrac{1}{|B_m|}\sum_{i \in B_m} a_i`
        is the fraction of correct predictions in bin :math:`m`,
        :math:`\operatorname{conf}(B_m) = \tfrac{1}{|B_m|}\sum_{i \in B_m} \hat{p}_i`
        is the mean predicted confidence in bin :math:`m`, and :math:`|B_m|/N` is
        the fraction of total samples in bin :math:`m`.

        This metric is particularly useful for datasets or models where predictions are
        concentrated in certain regions of the probability space.

        Args:
            task: Specifies the task type, either ``"binary"`` or ``"multiclass"``.
            num_bins: Number of bins to divide the probability space. Defaults to ``10``.
            norm: Specifies the type of norm to use: ``"l1"``, ``"l2"``, or ``"max"``.
                Defaults to ``"l1"``.
            num_classes: Number of classes for ``"multiclass"`` tasks. Required when task is ``"multiclass"``.
            ignore_index: Index to ignore during calculations. Defaults to ``None``.
            validate_args: Whether to validate input arguments. Defaults to ``True``.
            kwargs: Additional keyword arguments passed to the metric.

        Example:

            .. code-block:: python

                from torch_uncertainty.metrics.classification.adaptive_calibration_error import (
                    AdaptiveCalibrationError,
                )

                # Binary classification example
                predicted_probs = torch.tensor([0.95, 0.85, 0.15, 0.05])
                true_labels = torch.tensor([1, 1, 0, 0])

                metric = AdaptiveCalibrationError(
                    task="binary",
                    num_bins=5,
                    norm="l1",
                )

                calibration_error = metric(predicted_probs, true_labels)
                print(f"Calibration Error (Binary): {calibration_error}")
                # Output : Calibration Error (Binary): 0.1


        Note:
            - Adaptive binning adjusts the size of bins to ensure a more uniform distribution of samples across bins.
            - If `task="multiclass"`, `num_classes` must be provided; otherwise, a :class:`TypeError` will be raised.

        Warning:
            - Ensure that `num_classes` matches the actual number of classes in the dataset for multiclass tasks.

        References:
            [1] `Nixon et al., Measuring calibration in deep learning, CVPR Workshops, 2019
            <https://arxiv.org/abs/1904.01685>`_.

        .. seealso::
            - :class:`CalibrationError` for a metric that uses uniform binning.
        """
        task_enum = ClassificationTaskNoMultilabel.from_str(task)
        kwargs.update(
            {
                "n_bins": num_bins,
                "norm": norm,
                "ignore_index": ignore_index,
                "validate_args": validate_args,
            }
        )
        if task_enum == ClassificationTaskNoMultilabel.BINARY:
            return BinaryAdaptiveCalibrationError(**kwargs)
        # task is ClassificationTaskNoMultilabel.MULTICLASS
        if not isinstance(num_classes, int):
            raise TypeError(
                f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`"
            )
        return MulticlassAdaptiveCalibrationError(num_classes, **kwargs)
