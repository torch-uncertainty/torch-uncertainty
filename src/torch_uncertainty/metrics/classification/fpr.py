import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat


class FPRx(Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    confidences: list[Tensor]
    targets: list[Tensor]

    def __init__(self, recall_level: float, pos_label: int, **kwargs) -> None:
        r"""Compute the False Positive Rate at x% Recall.

        The False Positive Rate at x% Recall (FPR@x) is a metric used in tasks like anomaly detection, out-of-distribution (OOD) detection, and binary classification. It measures the proportion of false positives (normal samples misclassified as anomalies) when the model achieves a specified recall level for the positive class (e.g., anomalies or OOD samples).

        Args:
            recall_level (float): The recall level at which to compute the FPR.
            pos_label (int): The positive label.
            kwargs: Additional arguments to pass to the metric class.

        Reference:
            Improved from https://github.com/hendrycks/anomaly-seg and translated to torch.

        Example:
            .. code-block:: python

                from torch_uncertainty.metrics.classification import FPRx

                # Initialize the metric with 95% recall and positive label as 1 (e.g., OOD)
                metric = FPRx(recall_level=0.95, pos_label=1)

                # Simulated model predictions (confidence scores) and ground-truth labels
                confidences = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.4, 0.2, 0.1])
                targets = torch.tensor([1, 0, 1, 0, 0, 1, 0])  # 1: OOD, 0: In-Distribution

                # Update the metric with predictions and labels
                metric.update(confidences, targets)

                # Compute FPR at 95% recall
                result = metric.compute()
                print(f"FPR at 95% Recall: {result.item()}")
                # output : FPR at 95% Recall: 0.75
        """
        super().__init__(**kwargs)

        if recall_level < 0 or recall_level > 1:
            raise ValueError(f"Recall level must be between 0 and 1. Got {recall_level}.")
        self.recall_level = recall_level
        self.pos_label = pos_label
        self.add_state("confidences", [], dist_reduce_fx="cat")
        self.add_state("targets", [], dist_reduce_fx="cat")

        rank_zero_warn(
            f"Metric `FPR{int(recall_level * 100)}` will save all targets and predictions"
            " in buffer. For large datasets this may lead to large memory"
            " footprint."
        )

    def update(self, confidences: Tensor, target: Tensor) -> None:
        """Update the metric state.

        Args:
            confidences (Tensor): The confidence scores.
            target (Tensor): The target labels, 0 if ID, 1 if OOD.
        """
        self.confidences.append(confidences)
        self.targets.append(target)

    def compute(self) -> Tensor:
        """Compute the False Positive Rate at x% Recall.

        Returns:
            Tensor: The value of the FPRx.
        """
        confidences = dim_zero_cat(self.confidences)
        targets = dim_zero_cat(self.targets)

        # map examples and labels to OOD first
        indx = torch.argsort(targets, descending=True)
        examples = confidences[indx]
        labels = torch.zeros_like(targets, dtype=torch.bool, device=self.device)
        labels[: torch.count_nonzero(targets)] = True

        # sort examples and labels by decreasing confidence
        desc_scores_indx = torch.argsort(examples, descending=True)
        examples = examples[desc_scores_indx]
        labels = labels[desc_scores_indx]

        # Get the indices of the distinct values
        distinct_value_indices = torch.where(torch.diff(examples))[0]
        threshold_idxs = torch.cat(
            [
                distinct_value_indices,
                torch.tensor([labels.shape[0] - 1], dtype=torch.long, device=self.device),
            ]
        )

        # accumulate the true positives with decreasing threshold
        true_pos = torch.cumsum(labels, dim=0)[threshold_idxs]
        false_pos = 1 + threshold_idxs - true_pos  # add one because of zero-based indexing

        # check that there is at least one OOD example
        if true_pos[-1] == 0:
            return torch.tensor([torch.nan], device=self.device)

        recall = true_pos / true_pos[-1]

        last_ind = torch.searchsorted(true_pos, true_pos[-1])
        recall = torch.cat(
            [
                recall[: last_ind + 1].flip(0),
                torch.tensor([1.0], device=self.device),
            ]
        )
        false_pos = torch.cat(
            [
                false_pos[: last_ind + 1].flip(0),
                torch.tensor([0.0], device=self.device),
            ]
        )
        cutoff = torch.argmin(torch.abs(recall - self.recall_level))
        return false_pos[cutoff] / (~labels).sum()


class FPR95(FPRx):
    def __init__(self, pos_label: int, **kwargs) -> None:
        r"""Compute the False Positive Rate at 95% Recall.

        This is a specific case of the more general FPRx metric, where the recall level is fixed at 95%.

        Args:
            pos_label (int): The positive label (e.g., 1 for OOD samples).
            kwargs: Additional arguments to pass to the FPRx metric class.

        .. seealso::
            - :class:`FPRx` - The base metric that allows customization of the recall level.
        """
        super().__init__(recall_level=0.95, pos_label=pos_label, **kwargs)
