from typing import Any, Literal

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
import torch
from torchmetrics.classification.calibration_error import (
    BinaryCalibrationError,
    MulticlassCalibrationError,
)
from torchmetrics.functional.classification.calibration_error import (
    _binning_bucketize,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.enums import ClassificationTaskNoMultilabel

from .adaptive_calibration_error import AdaptiveCalibrationError


def _reliability_diagram_subplot(
    ax,
    accuracies: np.ndarray,
    confidences: np.ndarray,
    bin_sizes: np.ndarray,
    bins: np.ndarray,
    title: str = "Reliability Diagram",
    xlabel: str = "Top-class Confidence (%)",
    ylabel: str = "Success Rate (%)",
) -> None:
    widths = 1.0 / len(bin_sizes)
    positions = bins + widths / 2.0
    alphas = 0.2 + 0.8 * bin_sizes

    colors = np.zeros((len(bin_sizes), 4))
    colors[:, 0] = 240 / 255.0
    colors[:, 1] = 60 / 255.0
    colors[:, 2] = 60 / 255.0
    colors[:, 3] = alphas

    gap_plt = ax.bar(
        positions * 100,
        np.abs(accuracies - confidences) * 100,
        bottom=np.minimum(accuracies, confidences) * 100,
        width=widths * 100,
        edgecolor=colors,
        color=colors,
        linewidth=1,
        label="Gap",
    )

    acc_plt = ax.bar(
        positions * 100,
        0,
        bottom=accuracies * 100,
        width=widths * 100,
        edgecolor="black",
        color="black",
        alpha=1.0,
        linewidth=2,
        label="Accuracy",
    )

    ax.set_aspect("equal")
    ax.plot([0, 100], [0, 100], linestyle="--", color="gray")

    gaps = np.abs(accuracies - confidences)
    ece = np.sum(gaps * bin_sizes) / np.sum(bin_sizes)

    ax.text(
        0.98,
        0.02,
        f"ECE={ece:.02%}",
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


def _confidence_histogram_subplot(
    ax,
    accuracies: np.ndarray,
    confidences: np.ndarray,
    title: str = "Examples per bin",
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

    ax.set_xlim(0, 100)
    ax.set_ylim(0, None)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    avg_acc = np.mean(accuracies)
    avg_conf = np.mean(confidences)

    acc_plt = ax.axvline(
        x=avg_acc * 100,
        ls="solid",
        lw=2,
        c="black",
        label="Accuracy",
    )
    conf_plt = ax.axvline(
        x=avg_conf * 100,
        ls="dotted",
        lw=2,
        c="#444",
        label="Avg. confidence",
    )
    ax.grid(True, alpha=0.3, linestyle="--", zorder=0)
    ax.legend(handles=[acc_plt, conf_plt], loc="upper left")


def reliability_chart(
    accuracies: np.ndarray,
    confidences: np.ndarray,
    bin_accuracies: np.ndarray,
    bin_confidences: np.ndarray,
    bin_sizes: np.ndarray,
    bins: np.ndarray,
    title: str = "Reliability Diagram",
    rd_xlabel: str = "Top-class Confidence (%)",
    rd_ylabel: str = "Success Rate (%)",
    ch_xlabel: str = "Top-class Confidence (%)",
    ch_ylabel: str = "Density (%)",
    figsize: tuple[int, int] = (6, 6),
    dpi: int = 150,
) -> tuple[object, object]:
    """Builds Reliability Diagram
    `Source <https://github.com/hollance/reliability-diagrams>`_.
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

    # reliability diagram subplot
    _reliability_diagram_subplot(
        ax[0],
        bin_accuracies,
        bin_confidences,
        bin_sizes,
        bins,
        title=title,
        xlabel=rd_xlabel,
        ylabel=rd_ylabel,
    )

    # confidence histogram subplot
    _confidence_histogram_subplot(
        ax[1], accuracies, confidences, title="", xlabel=ch_xlabel, ylabel=ch_ylabel
    )
    ax[1].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    return fig, ax


def custom_plot(
    self,
    title: str = "Reliability Diagram",
    rd_xlabel: str = "Top-class Confidence (%)",
    rd_ylabel: str = "Success Rate (%)",
    ch_xlabel: str = "Top-class Confidence (%)",
    ch_ylabel: str = "Density (%)",
) -> tuple[object, object]:
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
        acc_bin, conf_bin, prop_bin = _binning_bucketize(confidences, accuracies, bin_boundaries)

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
        title=title,
        rd_xlabel=rd_xlabel,
        rd_ylabel=rd_ylabel,
        ch_xlabel=ch_xlabel,
        ch_ylabel=ch_ylabel,
    )


# overwrite the plot method of the original metrics
BinaryCalibrationError.plot = custom_plot
MulticlassCalibrationError.plot = custom_plot


class CalibrationError:
    def __new__(  # type: ignore[misc]
        cls,
        task: Literal["binary", "multiclass"],
        adaptive: bool = False,
        num_bins: int = 10,
        norm: Literal["l1", "l2", "max"] = "l1",
        num_classes: int | None = None,
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        r"""Computes the Calibration Error for classification tasks.

        This metric evaluates how well a model's predicted probabilities align with
        the actual ground truth probabilities. Calibration is crucial in assessing
        the reliability of probabilistic predictions, especially for downstream
        decision-making tasks.

        Three norms are available for measuring calibration error:

        **Expected Calibration Error (ECE):**

        .. math::

            \text{ECE} = \sum_{i=1}^N b_i \lvert p_i - c_i \rvert

        **Maximum Calibration Error (MCE):**

        .. math::

            \text{MCE} = \max_{i} \lvert p_i - c_i \rvert

        **Root Mean Square Calibration Error (RMSCE):**

        .. math::

            \text{RMSCE} = \sqrt{\sum_{i=1}^N b_i (p_i - c_i)^2}

        Here:
            - :math:`p_i` is the accuracy of bin :math:`i` (fraction of correct predictions).
            - :math:`c_i` is the mean predicted confidence in bin :math:`i`.
            - :math:`b_i` is the fraction of total samples falling into bin :math:`i`.

        Bins are constructed either uniformly in the range :math:`[0, 1]` or adaptively
        (if `adaptive=True`).

        Args:
            task (str): Specifies the task type, either ``"binary"`` or ``"multiclass"``.
            adaptive (bool, optional): Whether to use adaptive binning. Defaults to ``False``.
            num_bins (int, optional): Number of bins to divide the probability space. Defaults to ``10``.
            norm (str, optional): Specifies the type of norm to use: ``"l1"``, ``"l2"``, or ``"max"``.
                Defaults to ``"l1"``.
            num_classes (int, optional): Number of classes for ``"multiclass"`` tasks. Required when task
                is ``"multiclass"``.
            ignore_index (int, optional): Index to ignore during calculations. Defaults to ``None``.
            validate_args (bool, optional): Whether to validate input arguments. Defaults to ``True``.
            **kwargs: Additional keyword arguments for the metric.

        Example:

        .. code-block:: python

            from torch_uncertainty.metrics.classification.calibration_error import (
                CalibrationError,
            )

            # Example for binary classification
            predicted_probs = torch.tensor([0.9, 0.8, 0.3, 0.2])
            true_labels = torch.tensor([1, 1, 0, 0])

            metric = CalibrationError(
                task="binary",
                num_bins=5,
                norm="l1",
                adaptive=False,
            )

            calibration_error = metric(predicted_probs, true_labels)
            print(f"Calibration Error: {calibration_error}")
            # Output: Calibration Error: 0.199

        Note:
            Bins are either uniformly distributed in :math:`[0, 1]` or adaptively sized
            (if `adaptive=True`).

        Warning:
            If `task="multiclass"`, `num_classes` must be an integer; otherwise, a :class:`TypeError`
            is raised.

        References:
            [1] `Naeini et al. Obtaining well calibrated probabilities using Bayesian binning. In AAAI, 2015
            <https://ojs.aaai.org/index.php/AAAI/article/view/9602>`_.

        .. seealso::
            See `CalibrationError <https://torchmetrics.readthedocs.io/en/stable/classification/calibration_error.html>`_
            for details. Our version of the metric is a wrapper around the original metric providing a plotting functionality.
        """
        if kwargs.get("n_bins") is not None:
            raise ValueError("`n_bins` does not exist in TorchUncertainty, use `num_bins`.")
        if adaptive:
            return AdaptiveCalibrationError(
                task=task,
                num_bins=num_bins,
                norm=norm,
                num_classes=num_classes,
                ignore_index=ignore_index,
                validate_args=validate_args,
                **kwargs,
            )
        task = ClassificationTaskNoMultilabel.from_str(task)
        kwargs.update(
            {
                "n_bins": num_bins,
                "norm": norm,
                "ignore_index": ignore_index,
                "validate_args": validate_args,
            }
        )
        if task == ClassificationTaskNoMultilabel.BINARY:
            return BinaryCalibrationError(**kwargs)
        #  task is ClassificationTaskNoMultilabel.MULTICLASS
        if not isinstance(num_classes, int):
            raise TypeError(
                f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`"
            )
        return MulticlassCalibrationError(num_classes, **kwargs)
