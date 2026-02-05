import torch
from torch import Tensor, nn
from torch.distributions import Independent

from torch_uncertainty.layers.bayesian import bayesian_modules
from torch_uncertainty.utils.distributions import get_dist_class


class KLDiv(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        """KL divergence loss for Bayesian Neural Networks. Gathers the KL from the
        modules computed in the forward passes.

        Args:
            model (nn.Module): Bayesian Neural Network
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
                kl_divergence = kl_divergence.to(device=module.lvposterior.device)
                kl_divergence += module.lvposterior - module.lprior
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
        """The Evidence Lower Bound (ELBO) loss for Bayesian Neural Networks.

        ELBO loss for Bayesian Neural Networks. Use this loss function with the
        objective that you seek to minimize as :attr:`inner_loss`.

        Args:
            model (nn.Module): The Bayesian Neural Network to compute the loss for
            inner_loss (nn.Module): The loss function to use during training
            kl_weight (float): The weight of the KL divergence term
            num_samples (int): The number of samples to use for the ELBO loss
            dist_family (str, optional): The distribution family to use for the output of the
                model. ``None`` means point-wise prediction. Defaults to ``None``.

        Note:
            Set the model to ``None`` if you use the ELBOLoss within the ClassificationRoutine.
            It will get filled automatically.
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
            inputs (Tensor): The inputs of the Bayesian Neural Network
            targets (Tensor): The target values

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
