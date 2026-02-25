from collections.abc import Callable
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from torch_uncertainty.layers.bayesian import BayesConv2d, BayesLinear
from torch_uncertainty.layers.mc_batch_norm import MCBatchNorm2d
from torch_uncertainty.layers.packed import PackedConv2d, PackedLinear
from torch_uncertainty.models.wrappers.batch_ensemble import BatchEnsemble
from torch_uncertainty.models.wrappers.stochastic import StochasticModel

__all__ = ["batchensemble_lenet", "bayesian_lenet", "lenet", "packed_lenet"]


class _LeNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        linear_layer: type[nn.Module],
        conv2d_layer: type[nn.Module],
        layer_args: dict,
        activation: Callable,
        norm: type[nn.Module],
        groups: int,
        dropout_rate: float,
        first_layer_args: dict | None = None,
        last_layer_args: dict | None = None,
    ) -> None:
        super().__init__()
        self.activation = activation

        if first_layer_args is None:
            first_layer_args = layer_args
        if last_layer_args is None:
            last_layer_args = layer_args

        batchnorm = False
        if norm == nn.Identity:
            self.norm1 = norm()
            self.norm2 = norm()
        elif norm == nn.BatchNorm2d or (isinstance(norm, partial) and norm.func == MCBatchNorm2d):
            batchnorm = True
        else:
            raise ValueError(f"norm must be nn.Identity or nn.BatchNorm2d. Got {norm}.")

        self.dropout_rate = dropout_rate

        self.conv1 = conv2d_layer(in_channels, 6, (5, 5), groups=groups, **first_layer_args)
        if batchnorm:
            self.norm1 = norm(6)
        self.conv_dropout = nn.Dropout2d(p=dropout_rate)
        self.conv2 = conv2d_layer(6, 16, (5, 5), groups=groups, **layer_args)
        if batchnorm:
            self.norm2 = norm(16)
        self.pooling = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = linear_layer(256, 120, **layer_args)
        self.fc_dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = linear_layer(120, 84, **layer_args)
        self.last_fc_dropout = nn.Dropout(p=dropout_rate)
        self.fc3 = linear_layer(84, num_classes, **last_layer_args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_dropout(self.activation(self.norm1(self.conv1(x))))
        out = F.max_pool2d(out, 2)
        out = self.conv_dropout(self.activation(self.norm2(self.conv2(out))))
        out = F.max_pool2d(out, 2)
        out = self.pooling(out)
        out = torch.flatten(out, 1)
        out = self.fc_dropout(
            self.activation(self.fc1(out)),
        )
        out = self.last_fc_dropout(self.activation(self.fc2(out)))
        return self.fc3(out)


def _lenet(
    stochastic: bool,
    in_channels: int,
    num_classes: int,
    layer_args: dict,
    num_samples: int = 16,
    linear_layer: type[nn.Module] = nn.Linear,
    conv2d_layer: type[nn.Module] = nn.Conv2d,
    activation: Callable = nn.ReLU,
    norm: type[nn.Module] = nn.Identity,
    groups: int = 1,
    dropout_rate: float = 0.0,
    first_layer_args: dict | None = None,
    last_layer_args: dict | None = None,
) -> _LeNet | StochasticModel:
    model = _LeNet(
        in_channels=in_channels,
        num_classes=num_classes,
        linear_layer=linear_layer,
        conv2d_layer=conv2d_layer,
        activation=activation,
        norm=norm,
        groups=groups,
        layer_args=layer_args,
        dropout_rate=dropout_rate,
        first_layer_args=first_layer_args,
        last_layer_args=last_layer_args,
    )
    if stochastic:
        return StochasticModel(model, num_samples)
    return model


def lenet(
    in_channels: int,
    num_classes: int,
    activation: Callable = F.relu,
    norm: type[nn.Module] = nn.Identity,
    groups: int = 1,
    dropout_rate: float = 0.0,
) -> _LeNet:
    return _lenet(
        stochastic=False,
        in_channels=in_channels,
        num_classes=num_classes,
        linear_layer=nn.Linear,
        conv2d_layer=nn.Conv2d,
        layer_args={},
        activation=activation,
        norm=norm,
        groups=groups,
        dropout_rate=dropout_rate,
    )


def batchensemble_lenet(
    in_channels: int,
    num_classes: int,
    num_estimators: int = 4,
    activation: Callable = F.relu,
    norm: type[nn.Module] = nn.BatchNorm2d,
    groups: int = 1,
    dropout_rate: float = 0.0,
    repeat_training_inputs: bool = False,
) -> _LeNet:
    model = lenet(
        in_channels=in_channels,
        num_classes=num_classes,
        activation=activation,
        norm=norm,
        groups=groups,
        dropout_rate=dropout_rate,
    )
    return BatchEnsemble(
        core_model=model,
        num_estimators=num_estimators,
        repeat_training_inputs=repeat_training_inputs,
        convert_layers=True,
    )


def packed_lenet(
    in_channels: int,
    num_classes: int,
    num_estimators: int = 4,
    alpha: float = 2,
    gamma: float = 1,
    activation: Callable = F.relu,
    norm: type[nn.Module] = nn.Identity,
    groups: int = 1,
    dropout_rate: float = 0.0,
) -> _LeNet:
    layer_args = {
        "num_estimators": num_estimators,
        "alpha": alpha,
        "gamma": gamma,
    }
    return _lenet(
        stochastic=False,
        in_channels=in_channels,
        num_classes=num_classes,
        linear_layer=PackedLinear,
        conv2d_layer=PackedConv2d,
        norm=norm,
        layer_args=layer_args,
        first_layer_args={**layer_args, "first": True},
        last_layer_args={**layer_args, "last": True},
        activation=activation,
        groups=groups,
        dropout_rate=dropout_rate,
    )


def bayesian_lenet(
    in_channels: int,
    num_classes: int,
    num_samples: int = 16,
    prior_sigma_1: float | None = None,
    prior_sigma_2: float | None = None,
    prior_pi: float | None = None,
    mu_init: float | None = None,
    sigma_init: float | None = None,
    activation: Callable = F.relu,
    norm: type[nn.Module] = nn.Identity,
    groups: int = 1,
    dropout_rate: float = 0.0,
) -> StochasticModel:
    layers_args = {}
    if prior_sigma_1 is not None:
        layers_args["prior_sigma_1"] = prior_sigma_1
    if prior_sigma_2 is not None:
        layers_args["prior_sigma_2"] = prior_sigma_2
    if prior_pi is not None:
        layers_args["prior_pi"] = prior_pi
    if mu_init is not None:
        layers_args["mu_init"] = mu_init
    if sigma_init is not None:
        layers_args["sigma_init"] = sigma_init

    return _lenet(
        stochastic=True,
        num_samples=num_samples,
        in_channels=in_channels,
        num_classes=num_classes,
        linear_layer=BayesLinear,
        conv2d_layer=BayesConv2d,
        norm=norm,
        layer_args=layers_args,
        activation=activation,
        groups=groups,
        dropout_rate=dropout_rate,
    )
