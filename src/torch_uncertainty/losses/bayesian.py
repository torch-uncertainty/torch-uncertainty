import torch
from torch import Tensor, nn
from torch.distributions import Independent

from torch_uncertainty.layers.bayesian import bayesian_modules
from torch_uncertainty.utils.distributions import get_dist_class


class KLDiv(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        r"""KL divergence loss for Bayesian Neural Networks.

        Aggregates the per-layer Kullback-Leibler divergences between the variational
        posterior :math:`q_\phi(\mathbf{w})` and the prior :math:`p(\mathbf{w})`:

        .. math::
            \mathrm{KL}\!\left[ q_\phi(\mathbf{w}) \;\|\; p(\mathbf{w}) \right]
            = \frac{1}{L} \sum_{\ell=1}^{L} \mathrm{KL}\!\left[ q_\phi^{(\ell)} \;\|\; p^{(\ell)} \right].

        Each Bayesian layer caches a single-sample Monte Carlo estimate of its KL term
        during the forward pass via two scalars:

        - ``log_variational_posterior`` — :math:`\log q_\phi(\mathbf{w}^{(s)})`, the **log variational
          posterior** evaluated at the sampled weight,
        - ``log_prior`` — :math:`\log p(\mathbf{w}^{(s)})`, the **log prior** at the same
          sample,

        so that ``log_variational_posterior - log_prior`` is a one-sample estimate of the layer's KL.
        ``KLDiv`` simply averages these contributions over the :math:`L` Bayesian
        layers of :attr:`model`. The result is intended to be added to the data-fit
        term of an ELBO objective — see :class:`ELBOLoss`.

        Args:
            model: The Bayesian Neural Network whose layers expose ``log_variational_posterior`` and
                ``log_prior`` log-probabilities.
        """
        super().__init__()
        self.model = model

    def forward(self) -> Tensor:
        return self._kl_div()

    def _kl_div(self) -> Tensor:
        """Gathers pre-computed KL-Divergences from :attr:`model`."""
        kl_divergence = torch.zeros(1)
        count = 0
        for module in self.model.modules():
            if isinstance(module, bayesian_modules):
                kl_divergence = kl_divergence.to(device=module.log_variational_posterior.device)
                kl_divergence += module.log_variational_posterior - module.log_prior
                count += 1
        return kl_divergence / count


class ELBOLoss(nn.Module):
    model: nn.Module

    def __init__(
        self,
        model: nn.Module | None,
        inner_loss: nn.Module,
        kl_weight: float,
        num_samples: int,
        dist_family: str | None = None,
    ) -> None:
        r"""The (negative) Evidence Lower Bound (ELBO) loss for Bayesian Neural Networks.

        Combines an inner data-fit loss (e.g., cross-entropy or a distribution NLL)
        with the Kullback-Leibler regulariser :math:`\mathrm{KL}[q_\phi(\mathbf{w}) \|
        p(\mathbf{w})]`, estimated by Monte Carlo over :attr:`num_samples` weight
        samples:

        .. math::
            \mathcal{L}_{\text{ELBO}} = \frac{1}{S} \sum_{s=1}^{S}
            \mathcal{L}_\text{inner}\!\left(f_{\mathbf{w}^{(s)}}(\mathbf{x}), y\right)
            + \beta_\text{KL} \cdot \mathrm{KL}\!\left[ q_\phi(\mathbf{w}) \;\|\; p(\mathbf{w}) \right],

        with :math:`\mathbf{w}^{(s)} \sim q_\phi`. The KL weight :math:`\beta_\text{KL}`
        is typically set to the inverse of the number of training points (or a
        manually annealed schedule).

        Args:
            model: The Bayesian Neural Network to compute the loss for.
            inner_loss: The data-fit loss to use during training.
            kl_weight: The weight :math:`\beta_\text{KL}` of the KL-divergence term.
            num_samples: The number of weight samples :math:`S` used to estimate the
                expectation.
            dist_family: The distribution family to use for the output of the model.
                ``None`` means point-wise prediction. Defaults to ``None``.

        Note:
            Set :attr:`model` to ``None`` when using ``ELBOLoss`` inside a
            ``ClassificationRoutine`` — it will be filled in automatically.
        """
        super().__init__()
        _elbo_loss_checks(inner_loss, kl_weight, num_samples)
        if model is not None:
            self.set_model(model)

        self.inner_loss = inner_loss
        self.kl_weight = kl_weight
        self.num_samples = num_samples
        self.dist_family = dist_family

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Gather the KL divergence from the Bayesian modules and aggregate
        the ELBO loss for a given network.

        Args:
            inputs: The inputs of the Bayesian Neural Network
            targets: The target values

        Returns:
            Tensor: The aggregated ELBO loss
        """
        aggregated_elbo = torch.zeros(1, device=inputs.device)
        dist_class = get_dist_class(self.dist_family) if self.dist_family is not None else None
        for _ in range(self.num_samples):
            out = self.model(inputs)
            if dist_class is not None:
                # Wrap the distribution in an Independent distribution for log_prob computation.
                out = Independent(dist_class(**out), 1)
            aggregated_elbo += self.inner_loss(out, targets)
            # TODO: This shouldn't be necessary
            aggregated_elbo += self.kl_weight * self._kl_div().to(inputs.device)
        return aggregated_elbo / self.num_samples

    def set_model(self, model: nn.Module) -> None:
        self.model = model
        if model is not None:
            self._kl_div = KLDiv(model)


def _elbo_loss_checks(inner_loss: nn.Module, kl_weight: float, num_samples: int) -> None:
    if isinstance(inner_loss, type):
        raise TypeError(f"The inner_loss should be an instance of a class.Got {inner_loss}.")

    if kl_weight < 0:
        raise ValueError(f"The KL weight should be non-negative. Got {kl_weight}.")

    if num_samples < 1:
        raise ValueError(f"The number of samples should not be lower than 1. Got {num_samples}.")
    if not isinstance(num_samples, int):
        raise TypeError(f"The number of samples should be an integer. Got {type(num_samples)}.")
