from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat


class Disagreement(Metric):
    is_differentiable = False
    higher_is_better = None
    full_state_update = False

    values: list[Tensor]
    total: Tensor

    def __init__(
        self,
        reduction: Literal["mean", "sum", "none"] | None = "mean",
        **kwargs: Any,
    ) -> None:
        r"""Calculate the Disagreement Metric.

        The Disagreement Metric estimates the confidence of an ensemble of
        estimators. Given the predicted classes :math:`\hat{y}_{b,n} =
        \arg\max_c \hat{p}_{b,n,c}` for sample :math:`b` and estimator
        :math:`n`, the disagreement is the fraction of estimator pairs that
        predict different classes:

        .. math::

            \text{Disagreement} = \frac{2}{N(N-1)}
            \sum_{1 \le n < m \le N} \mathbf{1}\!\left[
            \hat{y}_{b,n} \neq \hat{y}_{b,m} \right]

        where :math:`N` is the number of estimators. Equivalently, this equals

        .. math::

            \text{Disagreement} = 1 - \frac{1}{\binom{N}{2}}
            \sum_{c=1}^{C} \binom{n_c}{2}

        where :math:`n_c = \sum_{n=1}^{N} \mathbf{1}[\hat{y}_{b,n} = c]`
        is the number of estimators predicting class :math:`c`.

        Args:
            reduction: Determines how to reduce over the :math:`B`/batch dimension:

                - ``'mean'`` [default]: Averages score across samples
                - ``'sum'``: Sum score across samples
                - ``'none'`` or ``None``: Returns score per sample

            kwargs: Additional keyword arguments, see `Advanced metric settings <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

        Inputs:
            - :attr:`probs`: :math:`(B, N, C)`

            where :math:`B` is the batch size, :math:`C` is the number of classes and :math:`N` is the number of estimators.

        Note:
            A higher disagreement means a lower confidence.

        Warning:
            Make sure that the probabilities in :attr:`probs` are normalized to sum
            to one.

        Raises:
            ValueError:
                If :attr:`reduction` is not one of ``'mean'``, ``'sum'``,
                ``'none'`` or ``None``.

        Example:

        .. code-block:: python

            from torch_uncertainty.metrics.classification import Disagreement

            probs = torch.tensor(
                [
                    [[0.7, 0.3], [0.6, 0.4], [0.8, 0.2]],  # Example 1, 3 estimators
                    [[0.4, 0.6], [0.5, 0.5], [0.3, 0.7]],  # Example 2, 3 estimators
                ]
            )

            ds = Disagreement(reduction="mean")
            ds.update(probs)
            result = ds.compute()
            print(result)
            # output: tensor(0.3333)
        """
        super().__init__(**kwargs)

        allowed_reduction = ("sum", "mean", "none", None)
        if reduction not in allowed_reduction:
            raise ValueError(
                "Expected argument `reduction` to be one of ",
                f"{allowed_reduction} but got {reduction}",
            )

        self.reduction = reduction

        if self.reduction in ["mean", "sum"]:
            self.add_state(
                "values",
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )
        else:
            self.add_state("values", default=[], dist_reduce_fx="cat")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def _compute_disagreement(self, preds: Tensor) -> Tensor:
        num_estimators = preds.size(-1)
        counts = torch.sum(F.one_hot(preds), dim=1)
        max_counts = num_estimators * (num_estimators - 1) / 2
        return 1 - (counts * (counts - 1) / 2).sum(dim=1) / max_counts

    def update(self, probs: Tensor) -> None:  # pyrefly: ignore[bad-override]
        """Update state with prediction probabilities and targets.

        Args:
            probs: Probabilities from the model.
        """
        preds = probs.argmax(dim=-1)
        if self.reduction is None or self.reduction == "none":
            self.values.append(self._compute_disagreement(preds))
        else:
            self.values += self._compute_disagreement(preds).sum(dim=-1)
            self.total += probs.size(0)

    def compute(self) -> Tensor:
        """Compute Disagreement based on inputs passed in to ``update``."""
        values = dim_zero_cat(self.values)
        if self.reduction == "sum":
            return values.sum(dim=-1)
        if self.reduction == "mean":
            return values.sum(dim=-1) / self.total
        # reduction is None or "none"
        return values
