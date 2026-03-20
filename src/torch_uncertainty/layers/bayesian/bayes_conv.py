from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, init
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import (
    _pair,
    _reverse_repeat_tuple,
    _single,
    _triple,
)
from torch.nn.parameter import Parameter

from .sampler import CenteredGaussianMixture, TrainableDistribution

__all__ = ["BayesConv1d", "BayesConv2d", "BayesConv3d"]


class _BayesConvNd(Module):
    __constants__ = [
        "stride",
        "padding",
        "dilation",
        "groups",
        "padding_mode",
        "output_padding",
        "in_channels",
        "out_channels",
        "kernel_size",
    ]
    __annotations__ = {"bias": torch.Tensor | None}

    def _conv_forward(
        self, inputs: Tensor, weight: Tensor, bias: Tensor | None
    ) -> Tensor:  # coverage: ignore
        ...

    in_channels: int
    _reversed_padding_repeated_twice: list[int]
    out_channels: int
    kernel_size: tuple[int, ...]
    stride: tuple[int, ...]
    padding: str | tuple[int, ...]
    dilation: tuple[int, ...]
    prior_mu: float
    prior_sigma: float
    mu_init: float
    sigma_init: float
    frozen: bool
    transposed: bool
    output_padding: tuple[int, ...]
    groups: int
    padding_mode: str
    weight: Tensor
    bias: Tensor | None
    lprior: Tensor
    lvposterior: Tensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...],
        padding: tuple[int, ...],
        dilation: tuple[int, ...],
        prior_sigma_1: float,
        prior_sigma_2: float,
        prior_pi: float,
        mu_init: float,
        sigma_init: float,
        frozen: bool,
        transposed: bool,
        output_padding: tuple[int, ...],
        groups: int,
        bias: bool,
        padding_mode: str,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        valid_padding_modes = {"zeros", "reflect", "replicate", "circular"}
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                f"padding_mode must be one of {valid_padding_modes}, but got '{padding_mode}'"
            )

        if transposed:
            raise NotImplementedError(
                "Bayesian transposed convolution not implemented yet. Raise an issue if needed."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        self.mu_init = mu_init
        self.sigma_init = sigma_init
        self.frozen = frozen
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode

        self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)

        self.weight_mu = Parameter(
            torch.empty(
                (out_channels, in_channels // groups, *kernel_size),
                **factory_kwargs,
            )
        )
        self.weight_sigma = Parameter(
            torch.empty(
                (out_channels, in_channels // groups, *kernel_size),
                **factory_kwargs,
            )
        )

        if bias:
            self.bias_mu = Parameter(torch.empty(out_channels, **factory_kwargs))
            self.bias_sigma = Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_sigma", None)

        self.weight_prior_dist = CenteredGaussianMixture(prior_sigma_1, prior_sigma_2, prior_pi)
        if bias:
            self.bias_prior_dist = CenteredGaussianMixture(prior_sigma_1, prior_sigma_2, prior_pi)

        self.reset_parameters()

        self.weight_sampler = TrainableDistribution(self.weight_mu, self.weight_sigma)
        if bias:
            self.bias_sampler = TrainableDistribution(self.bias_mu, self.bias_sigma)

    def reset_parameters(self) -> None:
        # TODO: change init
        init.normal_(self.weight_mu, mean=self.mu_init, std=0.1)
        init.normal_(self.weight_sigma, mean=self.sigma_init, std=0.1)

        if self.bias_mu is not None:
            init.normal_(self.bias_mu, mean=self.mu_init, std=0.1)
            init.normal_(self.bias_sigma, mean=self.sigma_init, std=0.1)

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

    def extra_repr(self) -> str:  # coverage: ignore
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}"
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias_mu is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        return s.format(**self.__dict__)

    def __setstate__(self, state) -> None:
        super().__setstate__(state)
        if not hasattr(self, "padding_mode"):  # coverage: ignore
            self.padding_mode = "zeros"


class BayesConv1d(_BayesConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: str | _size_1_t = 0,
        dilation: _size_1_t = 1,
        prior_sigma_1: float = 0.1,
        prior_sigma_2: float = 0.002,
        prior_pi: float = 1,
        mu_init: float = 0.0,
        sigma_init: float = -6.0,
        frozen: bool = False,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["circular", "reflect", "replicate", "zeros"] = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        """Bayesian Conv1d Layer with Mixture of Normals prior and Normal
        posterior.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride)
        padding_ = padding if isinstance(padding, str) else _single(padding)
        dilation_ = _single(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            prior_sigma_1,
            prior_sigma_2,
            prior_pi,
            mu_init,
            sigma_init,
            frozen,
            False,
            _single(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _conv_forward(self, inputs: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
        if self.padding_mode != "zeros":
            return F.conv1d(
                F.pad(
                    inputs,
                    self._reversed_padding_repeated_twice,
                    mode=self.padding_mode,
                ),
                weight,
                bias,
                self.stride,
                _single(0),
                self.dilation,
                self.groups,
            )
        return F.conv1d(
            inputs,
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if self.frozen:
            weight = self.weight_mu
            bias = self.bias_mu
        else:
            weight = self.weight_sampler.sample()

            if self.bias_mu is not None:
                bias = self.bias_sampler.sample()
                bias_lposterior = self.bias_sampler.log_posterior()
                bias_lprior = self.bias_prior_dist.log_prob(bias)
            else:
                bias, bias_lposterior, bias_lprior = None, 0, 0

            self.lvposterior = self.weight_sampler.log_posterior() + bias_lposterior
            self.lprior = self.weight_prior_dist.log_prob(weight) + bias_lprior

        return self._conv_forward(inputs, weight, bias)


class BayesConv2d(_BayesConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: str | _size_2_t = 0,
        dilation: _size_2_t = 1,
        prior_sigma_1: float = 0.1,
        prior_sigma_2: float = 0.002,
        prior_pi: float = 1,
        mu_init: float = 0.0,
        sigma_init: float = -6.0,
        frozen: bool = False,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["circular", "reflect", "replicate", "zeros"] = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        """Bayesian Conv2d Layer with Gaussian Mixture prior and Normal posterior."""
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            prior_sigma_1,
            prior_sigma_2,
            prior_pi,
            mu_init,
            sigma_init,
            frozen,
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _conv_forward(self, inputs: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(
                    inputs,
                    self._reversed_padding_repeated_twice,
                    mode=self.padding_mode,
                ),
                weight,
                bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(
            inputs,
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if self.frozen:
            weight = self.weight_mu
            bias = self.bias_mu
        else:
            weight = self.weight_sampler.sample()

            if self.bias_mu is not None:
                bias = self.bias_sampler.sample()
                bias_lposterior = self.bias_sampler.log_posterior()
                bias_lprior = self.bias_prior_dist.log_prob(bias)
            else:
                bias, bias_lposterior, bias_lprior = None, 0, 0

            self.lvposterior = self.weight_sampler.log_posterior() + bias_lposterior
            self.lprior = self.weight_prior_dist.log_prob(weight) + bias_lprior

        return self._conv_forward(inputs, weight, bias)


class BayesConv3d(_BayesConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: str | _size_3_t = 0,
        dilation: _size_3_t = 1,
        prior_sigma_1: float = 0.1,
        prior_sigma_2: float = 0.002,
        prior_pi: float = 1,
        mu_init: float = 0.0,
        sigma_init: float = 10.0,
        frozen: bool = False,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["circular", "reflect", "replicate", "zeros"] = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        """Bayesian Conv3d Layer with Gaussian mixture prior and Normal posterior."""
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size_ = _triple(kernel_size)
        stride_ = _triple(stride)
        padding_ = padding if isinstance(padding, str) else _triple(padding)
        dilation_ = _triple(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            prior_sigma_1,
            prior_sigma_2,
            prior_pi,
            mu_init,
            sigma_init,
            frozen,
            False,
            _triple(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _conv_forward(self, inputs: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
        if self.padding_mode != "zeros":
            return F.conv3d(
                F.pad(
                    inputs,
                    self._reversed_padding_repeated_twice,
                    mode=self.padding_mode,
                ),
                weight,
                bias,
                self.stride,
                _triple(0),
                self.dilation,
                self.groups,
            )
        return F.conv3d(
            inputs,
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if self.frozen:
            weight = self.weight_mu
            bias = self.bias_mu
        else:
            weight = self.weight_sampler.sample()

            if self.bias_mu is not None:
                bias = self.bias_sampler.sample()
                bias_lposterior = self.bias_sampler.log_posterior()
                bias_lprior = self.bias_prior_dist.log_prob(bias)
            else:
                bias, bias_lposterior, bias_lprior = None, 0, 0

            self.lvposterior = self.weight_sampler.log_posterior() + bias_lposterior
            self.lprior = self.weight_prior_dist.log_prob(weight) + bias_lprior

        return self._conv_forward(inputs, weight, bias)
