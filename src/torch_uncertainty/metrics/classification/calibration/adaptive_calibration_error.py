from typing import Any, Literal

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
        confidences: The confidence (i.e. predicted prob) of the top1
            prediction.
        accuracies: 1.0 if the top-1 prediction was correct, 0.0 otherwise.
        num_bins: Number of bins to use when computing adaptive calibration
            error.

    Returns:
        tuple with binned accuracy, binned confidence and binned probabilities
    """
    confidences, indices = torch.sort(confidences)
    accuracies = accuracies[indices]
    acc_bin, conf_bin = (
        accuracies.tensor_split(num_bins),
        confidences.tensor_split(num_bins),
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


def _ace_compute(
    confidences: Tensor,
    accuracies: Tensor,
    num_bins: int,
    norm: Literal["l1", "l2", "max"] = "l1",
    debias: bool = False,
) -> Tensor:
    """Compute the adaptive calibration error given the provided number of bins
        and norm.

    Args:
        confidences: The confidence (i.e. predicted prob) of the top1
            prediction.
        accuracies: 1.0 if the top-1 prediction was correct, 0.0 otherwise.
        num_bins: Number of bins to use when computing adaptive calibration
            error.
        norm: Norm function to use when computing calibration error. Defaults
            to "l1".
        debias: Apply debiasing to L2 norm computation as in
            `Verified Uncertainty Calibration`. Defaults to False.

    Returns:
        Tensor: Adaptive Calibration error scalar.
    """
    with torch.no_grad():
        acc_bin, conf_bin, prop_bin = _equal_binning_bucketize(confidences, accuracies, num_bins)

    if norm == "l1":
        return torch.sum(torch.abs(acc_bin - conf_bin) * prop_bin)
    if norm == "max":
        ace = torch.max(torch.abs(acc_bin - conf_bin))
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
    return ace


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
        r"""Adaptive Top-label Calibration Error for binary tasks."""
        super().__init__(**kwargs)
        if ignore_index is not None:  # coverage: ignore
            raise ValueError("ignore_index is not supported for multiclass tasks.")

        if validate_args:
            _binary_calibration_error_arg_validation(n_bins, norm, ignore_index)
        self.n_bins = n_bins
        self.norm = norm

        self.add_state("confidences", [], dist_reduce_fx="cat")
        self.add_state("accuracies", [], dist_reduce_fx="cat")

    def update(self, probs: Tensor, targets: Tensor) -> None:
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


class MulticlassAdaptiveCalibrationError(Metric):
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False

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
        r"""Adaptive Top-label Calibration Error for multiclass tasks."""
        super().__init__(**kwargs)
        if ignore_index is not None:  # coverage: ignore
            raise ValueError("ignore_index is not supported for multiclass tasks.")

        if validate_args:
            _multiclass_calibration_error_arg_validation(num_classes, n_bins, norm, ignore_index)
        self.n_bins = n_bins
        self.norm = norm

        self.add_state("confidences", [], dist_reduce_fx="cat")
        self.add_state("accuracies", [], dist_reduce_fx="cat")

    def update(self, probs: Tensor, targets: Tensor) -> None:
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

        This metric is particularly useful for datasets or models where predictions are
        concentrated in certain regions of the probability space.

        Args:
            task (str): Specifies the task type, either ``"binary"`` or ``"multiclass"``.
            num_bins (int, optional): Number of bins to divide the probability space. Defaults to ``10``.
            norm (str, optional): Specifies the type of norm to use: ``"l1"``, ``"l2"``, or ``"max"``. Defaults to ``"l1"``.
            num_classes (int, optional): Number of classes for ``"multiclass"`` tasks. Required when task is ``"multiclass"``.
            ignore_index (int, optional): Index to ignore during calculations. Defaults to ``None``.
            validate_args (bool, optional): Whether to validate input arguments. Defaults to ``True``.
            **kwargs (Any): Additional keyword arguments passed to the metric.

        Example:

            .. code-block:: python

                from torch_uncertainty.metrics.classification.adaptive_calibration_error import (
                    AdaptiveCalibrationError,
                )

                # Binary classification example
                predicted_probs = torch.tensor([0.95, 0.85, 0.15, 0.05])
                true_labels = torch.tensor([1, 1, 0, 0])

                metric = CalibrationError(
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
            - See `:class:`CalibrationError` for a metric that uses uniform binning.
        """
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
            return BinaryAdaptiveCalibrationError(**kwargs)
        # task is ClassificationTaskNoMultilabel.MULTICLASS
        if not isinstance(num_classes, int):
            raise TypeError(
                f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`"
            )
        return MulticlassAdaptiveCalibrationError(num_classes, **kwargs)
