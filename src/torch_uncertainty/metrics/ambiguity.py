from typing import Any, Literal

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat


class Ambiguity(Metric):
    is_differentiable = False
    higher_is_better = None
    full_state_update = False

    values: list[Tensor]
    total: Tensor

    def __init__(
        self,
        reduction: Literal["mean", "sum", "none"] | None = "mean",
        relative: bool = False,
        **kwargs: Any,
    ) -> None:
        r"""Compute the ambiguity of an ensemble of regression models.

        For predictions :math:`f_m(x_b)` from :math:`M` ensemble members, the
        ambiguity is the population variance around the ensemble prediction
        :math:`\bar f(x_b)`:

        .. math::

            A(x_b) = \frac{1}{M}\sum_{m=1}^M
            \left(f_m(x_b) - \bar f(x_b)\right)^2.

        If :attr:`relative` is ``True``, the ambiguity is normalized elementwise
        by the squared ensemble prediction:

        .. math::

            A_{\mathrm{rel}}(x_b) = \frac{A(x_b)}
            {\max\!\left(\bar f(x_b)^2, \epsilon\right)},

        where :math:`\epsilon` is the machine epsilon of the prediction dtype.
        This is the squared coefficient of variation of the ensemble predictions.
        For vector-valued predictions, the metric averages over the output
        dimensions to produce one value per sample. :attr:`relative` is intended
        for regression settings only.

        Args:
            reduction: Determines how to reduce over the batch dimension:

                - ``"mean"`` [default]: Average score across samples.
                - ``"sum"``: Sum score across samples.
                - ``"none"`` or ``None``: Return one score per sample.

            relative: If ``True``, normalize the ambiguity by the squared
                ensemble prediction. Defaults to ``False``.
            kwargs: Additional keyword arguments, see `Advanced metric settings
                <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

        Inputs:
            - :attr:`preds`: :math:`(B, M, ...)` floating-point ensemble
              predictions, where :math:`B` is the batch size and :math:`M` is
              the number of ensemble members.

        Reference:
            `Neural Network Ensembles, Cross Validation, and Active Learning,
            NeurIPS 1994
            <https://proceedings.neurips.cc/paper/1994/hash/b8c37e33defde51cf91e1e03e51657da-Abstract.html>`_.

        Raises:
            ValueError: If :attr:`reduction` is invalid, :attr:`preds` is not a
                floating-point tensor of at least two dimensions, or fewer than
                two ensemble members are provided.
        """
        super().__init__(**kwargs)

        allowed_reduction = ("sum", "mean", "none", None)
        if reduction not in allowed_reduction:
            raise ValueError(
                "Expected argument `reduction` to be one of "
                f"{allowed_reduction} but got {reduction}."
            )

        self.reduction = reduction
        self.relative = relative

        if reduction in ("mean", "sum"):
            self.add_state("values", default=torch.tensor(0.0), dist_reduce_fx="sum")
        else:
            self.add_state("values", default=[], dist_reduce_fx="cat")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def _compute_ambiguity(self, preds: Tensor) -> Tensor:
        ensemble_preds = preds.mean(dim=1, keepdim=True)
        ambiguity = (preds - ensemble_preds).square().mean(dim=1)
        if self.relative:
            scale = ensemble_preds.squeeze(1).square().clamp_min(torch.finfo(preds.dtype).eps)
            ambiguity = ambiguity / scale
        if ambiguity.ndim > 1:
            ambiguity = ambiguity.flatten(start_dim=1).mean(dim=1)
        return ambiguity

    def update(self, preds: Tensor) -> None:  # pyrefly: ignore[bad-override]
        """Update the metric state with ensemble predictions.

        Args:
            preds: Floating-point ensemble predictions of shape :math:`(B, M, ...)`.
        """
        if preds.ndim < 2:
            raise ValueError(
                "Expected `preds` to have shape (batch, estimators, ...), "
                f"but got {tuple(preds.shape)}."
            )
        if preds.size(1) < 2:
            raise ValueError("Expected at least two ensemble members.")
        if not preds.is_floating_point():
            raise ValueError("Expected `preds` to be a floating-point tensor.")

        ambiguity = self._compute_ambiguity(preds)
        if self.reduction is None or self.reduction == "none":
            self.values.append(ambiguity)
        else:
            self.values += ambiguity.sum()
            self.total += ambiguity.numel()

    def compute(self) -> Tensor:
        """Compute ambiguity from the inputs passed to :meth:`update`."""
        values = dim_zero_cat(self.values)
        if self.reduction == "sum":
            return values.sum()
        if self.reduction == "mean":
            return values.sum() / self.total
        return values
