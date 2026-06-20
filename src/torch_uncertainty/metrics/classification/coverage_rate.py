import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.utilities.data import _bincount


class CoverageRate(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    correct: Tensor
    total: Tensor

    def __init__(
        self,
        num_classes: int | None = None,
        average: str = "micro",
        validate_args: bool = True,
        **kwargs,
    ) -> None:
        r"""Empirical coverage rate metric.

        For a prediction set :math:`\mathcal{C}(X)` returned by a conformal predictor
        (or any set-valued predictor), the coverage rate is the fraction of test points
        whose ground-truth label is contained in the predicted set:

        .. math::
            \text{Coverage} = \frac{1}{N} \sum_{i=1}^{N}
            \mathbf{1}\!\left[ y_i \in \mathcal{C}(x_i) \right].

        With ``average="macro"``, the per-class coverage rates are averaged uniformly:

        .. math::
            \text{Coverage}_{\text{macro}} = \frac{1}{C} \sum_{c=1}^{C}
            \frac{\sum_{i:\, y_i = c} \mathbf{1}\!\left[ y_i \in \mathcal{C}(x_i) \right]}
                 {\sum_{i:\, y_i = c} 1}.

        Args:
            num_classes: Number of classes. Defaults to ``None``.
            average: Defines the reduction that is applied over labels.  Defaults to ``"macro"``.
                Should be one of the following:

                - ``'macro'``: Compute the metric for each class separately and find their
                  unweighted mean. This does not take label imbalance into account.
                - ``'micro'``: Sum statistics across over all labels.

            validate_args: Whether to validate the arguments. Defaults to ``True``.
            kwargs: Additional keyword arguments, see `Advanced metric settings
                <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

        Raises:
            ValueError: If `num_classes` is `None` and `average` is not `micro`.
            ValueError: If `num_classes` is not an integer larger than 1.
            ValueError: If `average` is not one of `macro` or `micro`.
        """
        super().__init__(**kwargs)

        if validate_args:
            if num_classes is None and average != "micro":
                raise ValueError(
                    f"Argument `num_classes` can only be `None` for `average='micro'`, but got `average={average}`."
                )
            if num_classes is not None and (not isinstance(num_classes, int) or num_classes < 2):
                raise ValueError(
                    f"Expected argument `num_classes` to be an integer larger than 1, but got {num_classes}"
                )
            if average not in ["macro", "micro"]:
                raise ValueError("average must be either 'macro' or 'micro'.")

        self.num_classes = num_classes
        self.average = average
        self.validate_args = validate_args

        size = 1 if (average == "micro" or num_classes is None) else num_classes

        self.add_state("correct", torch.zeros(size, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("total", torch.zeros(size, dtype=torch.float), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update the metric state with predictions and targets.

        Args:
            preds: Predicted sets tensor of shape (B, C), where B is the batch size
                and C is the number of classes.
            target: Target labels tensor of shape (B,).
        """
        batch_size = preds.size(0)
        target = target.long()

        covered = preds[torch.arange(batch_size), target]  # (B,)

        if self.average == "micro":
            self.correct += covered.bool().sum()
            self.total += batch_size

        else:
            self.correct += _bincount(target[covered.bool()], self.num_classes)
            self.total += _bincount(target, self.num_classes)

    def compute(self) -> Tensor:
        """Compute the coverage rate.

        Returns:
            Tensor: The coverage rate.
        """
        if self.average == "micro":
            return _safe_divide(self.correct, self.total)
        return _safe_divide(self.correct, self.total).mean()
