from torch import Tensor, distributions
from torchmetrics.utilities.data import dim_zero_cat

from torch_uncertainty.metrics import CategoricalNLL


class DistributionNLL(CategoricalNLL):
    r"""Negative Log-Likelihood under a predictive distribution.

    Evaluates a probabilistic regression model by computing the negative log-likelihood
    of the targets under the model's predictive distribution
    :math:`p_\theta(y \mid x)`:

    .. math::
        \text{NLL} = -\frac{1}{N} \sum_{i=1}^{N} \log p_\theta(y_i \mid x_i).

    For multi-variate targets, the underlying distribution is typically wrapped in a
    :class:`torch.distributions.Independent` so that ``log_prob`` correctly sums the
    log-density over the event dimensions.

    Args:
        reduction: How to reduce the per-sample losses (``"mean"``, ``"sum"``,
            ``"none"`` or ``None``). Defaults to ``"mean"``.
        kwargs: Additional keyword arguments, see `Advanced metric settings
            <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

    Inputs:
        - :attr:`dist`: a :class:`torch.distributions.Distribution` over the targets.
        - :attr:`target`: ground-truth targets of compatible shape.
        - :attr:`padding_mask`: optional boolean mask of positions to ignore (``True`` for padding).
    """

    def update(  # pyrefly: ignore[bad-override]
        self,
        dist: distributions.Distribution,
        target: Tensor,
        padding_mask: Tensor | None = None,
    ) -> None:
        """Update state with the predicted distributions and the targets.

        Args:
            dist: Predicted distributions.
            target: Ground truth labels.
            padding_mask: Optional padding mask. Sets the loss to 0 for padded values. Defaults to
                ``None``.
        """
        nlog_prob = -dist.log_prob(target)
        if padding_mask is not None:
            nlog_prob = nlog_prob.masked_fill(padding_mask, float("nan"))
        if self.reduction is None or self.reduction == "none":
            self.values.append(nlog_prob)
        else:
            self.values += nlog_prob.nansum()
            self.total += padding_mask.sum() if padding_mask is not None else target.numel()

    def compute(self) -> Tensor:
        """Compute NLL based on inputs passed to ``update``."""
        values = dim_zero_cat(self.values)

        if self.reduction == "sum":
            return values.nansum()
        if self.reduction == "mean":
            return values.nansum() / self.total
        # reduction is None or "none"
        return values
