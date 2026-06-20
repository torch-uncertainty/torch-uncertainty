import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import init

from .sampler import CenteredGaussianMixture, TrainableDistribution


class BayesLinear(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor
    log_prior: Tensor
    log_variational_posterior: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_sigma_1: float = 0.1,
        prior_sigma_2: float = 0.4,
        prior_pi: float = 1,
        mu_init: float = 0.0,
        sigma_init: float = -7.0,
        frozen: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        """Bayesian Linear Layer with Mixture of Normals prior and Normal posterior.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            prior_sigma_1: Standard deviation of the first prior distribution. Defaults to ``0.1``.
            prior_sigma_2: Standard deviation of the second prior distribution. Defaults to ``0.4``.
            prior_pi: Mixture control variable. Defaults to ``1``.
            mu_init: Initial mean of the posterior distribution. Defaults to ``0.0``.
            sigma_init: Initial standard deviation of the posterior distribution. Defaults to ``-7.0``.
            frozen: Whether to freeze the posterior distribution. Defaults to ``False``.
            bias: Whether to use a bias term. Defaults to ``True.``
            device: Device to use. Defaults to ``None``.
            dtype: Data type to use. Defaults to ``None``.

        References:
            [1] `Blundell, Charles, et al. "Weight uncertainty in neural networks", in ICML 2015
            <https://arxiv.org/abs/1505.05424>`_.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        self.mu_init = mu_init
        self.sigma_init = sigma_init
        self.frozen = frozen

        self.weight_mu = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.weight_sigma = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        if bias:
            self.bias_mu = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            self.bias_sigma = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_log_sigma", None)

        self.reset_parameters()
        self.weight_sampler = TrainableDistribution(self.weight_mu, self.weight_sigma)
        if bias:
            self.bias_sampler = TrainableDistribution(self.bias_mu, self.bias_sigma)

        self.weight_prior_dist = CenteredGaussianMixture(prior_sigma_1, prior_sigma_2, prior_pi)
        if bias:
            self.bias_prior_dist = CenteredGaussianMixture(prior_sigma_1, prior_sigma_2, prior_pi)

    def reset_parameters(self) -> None:
        # TODO: change init
        init.normal_(self.weight_mu, mean=self.mu_init, std=0.1)
        init.normal_(self.weight_sigma, mean=self.sigma_init, std=0.1)

        if self.bias_mu is not None:
            init.normal_(self.bias_mu, mean=self.mu_init, std=0.1)
            init.normal_(self.bias_sigma, mean=self.sigma_init, std=0.1)

    def forward(self, inputs: Tensor) -> Tensor:
        if self.frozen:
            return self._frozen_forward(inputs)
        return self._forward(inputs)

    def _frozen_forward(self, inputs) -> Tensor:
        return F.linear(inputs, self.weight_mu, self.bias_mu)

    def _forward(self, inputs: Tensor) -> Tensor:
        weight = self.weight_sampler.sample()

        if self.bias_mu is not None:
            bias = self.bias_sampler.sample()
            bias_lposterior = self.bias_sampler.log_posterior()
            bias_log_prior = self.bias_prior_dist.log_prob(bias)
        else:
            bias, bias_lposterior, bias_log_prior = None, 0, 0

        self.log_variational_posterior = self.weight_sampler.log_posterior() + bias_lposterior
        self.log_prior = self.weight_prior_dist.log_prob(weight) + bias_log_prior

        return F.linear(inputs, weight, bias)

    def freeze(self) -> None:
        """Freeze the layer by setting the frozen attribute to True."""
        self.frozen = True

    def unfreeze(self) -> None:
        """Unfreeze the layer by setting the frozen attribute to False."""
        self.frozen = False

    def sample(self) -> tuple[Tensor, Tensor | None]:
        """Sample the Bayesian layer's posterior."""
        weight = self.weight_sampler.sample()
        bias = self.bias_sampler.sample() if self.bias_mu is not None else None
        return weight, bias

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias_mu is not None}"
