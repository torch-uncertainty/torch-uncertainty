from importlib import util
from typing import Literal

from torch import Tensor, nn
from torch.utils.data import DataLoader

from .abstract import PostProcessing

if util.find_spec("laplace"):
    from laplace import Laplace

    laplace_installed = True
else:  # coverage: ignore
    laplace_installed = False


class LaplaceApprox(PostProcessing):
    def __init__(
        self,
        task: Literal["classification", "regression"],
        model: nn.Module | None = None,
        weight_subset: str = "last_layer",
        hessian_struct: str = "kron",
        pred_type: Literal["glm", "nn"] = "glm",
        link_approx: Literal["mc", "probit", "bridge", "bridge_norm"] = "probit",
        optimize_prior_precision: bool = True,
    ) -> None:
        r"""Laplace approximation for post-hoc Bayesian uncertainty estimation.

        Fits a Gaussian posterior :math:`\mathcal{N}(\boldsymbol{\theta}_\text{MAP},
        \mathbf{H}^{-1})` around the MAP estimate of a trained network, where
        :math:`\mathbf{H}` is (an approximation of) the Hessian of the negative
        log-posterior, computed on the calibration set. Predictions are then obtained
        by marginalising over the posterior — analytically for regression (and the
        ``"probit"`` / ``"bridge"`` classification approximations), or by Monte Carlo
        for ``"mc"``.

        This class is a thin wrapper around the
        `laplace-torch <https://github.com/aleximmer/Laplace>`_ library.

        Args:
            task: Task type. Either ``"classification"`` or ``"regression"``.
            model: Model to be converted.
            weight_subset: Subset of weights to be considered (e.g. ``"last_layer"``
                or ``"all"``). Defaults to ``"last_layer"``.
            hessian_struct: Structure of the Hessian approximation (e.g. ``"kron"``,
                ``"diag"``, ``"full"``). Defaults to ``"kron"``.
            pred_type: Type of posterior predictive. See the Laplace library for
                details. Defaults to ``"glm"``.
            link_approx: How to approximate the classification link function for the
                ``"glm"`` predictive. See the Laplace library for details. Defaults to
                ``"probit"``.
            optimize_prior_precision: Whether to optimize the prior precision by
                marginal likelihood. Defaults to ``True``.

        References:
            [1] `Daxberger, E., Kristiadi, A., Immer, A., Eschenhagen, R., Bauer, M., &
            Hennig, P. (2021). Laplace Redux — Effortless Bayesian Deep Learning. NeurIPS 2021
            <https://arxiv.org/abs/2106.14806>`_.
        """
        super().__init__()
        if not laplace_installed:
            raise ImportError(
                "The laplace-torch library is not installed. Please install "
                "torch_uncertainty with the all option: pip install -U torch_uncertainty[all]"
            )

        self.pred_type = pred_type
        self.link_approx = link_approx
        self.task = task
        self.weight_subset = weight_subset
        self.hessian_struct = hessian_struct
        self.optimize_prior_precision = optimize_prior_precision

        if model is not None:
            self.set_model(model)

    def set_model(self, model: nn.Module) -> None:
        super().set_model(model)
        self.la = Laplace(
            model=model,
            likelihood=self.task,
            subset_of_weights=self.weight_subset,
            hessian_structure=self.hessian_struct,
        )

    def fit(self, dataloader: DataLoader) -> None:
        self.la.fit(train_loader=dataloader)
        if self.optimize_prior_precision:
            self.la.optimize_prior_precision(method="marglik", pred_type=self.pred_type)

    def forward(
        self,
        inputs: Tensor,
    ) -> Tensor:
        out = self.la(inputs, pred_type=self.pred_type, link_approx=self.link_approx, n_samples=100)
        if isinstance(out, tuple):  # coverage: ignore
            return out[0]
        return out
