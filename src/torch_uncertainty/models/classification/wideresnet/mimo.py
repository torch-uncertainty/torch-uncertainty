from collections.abc import Callable
from typing import Literal

import torch
from einops import rearrange
from torch import nn
from torch.nn.functional import relu

from .std import _WideResNet

__all__ = [
    "mimo_wideresnet28x10",
]


class _MIMOWideResNet(_WideResNet):
    def __init__(
        self,
        depth: int,
        widen_factor: int,
        in_channels: int,
        num_classes: int,
        num_estimators: int,
        conv_bias: bool,
        dropout_rate: float,
        groups: int = 1,
        style: Literal["imagenet", "cifar"] = "imagenet",
        activation_fn: Callable = relu,
        normalization_layer: type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__(
            depth,
            widen_factor=widen_factor,
            in_channels=in_channels * num_estimators,
            num_classes=num_classes * num_estimators,
            conv_bias=conv_bias,
            dropout_rate=dropout_rate,
            groups=groups,
            style=style,
            activation_fn=activation_fn,
            normalization_layer=normalization_layer,
        )
        self.num_estimators = num_estimators

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            x = x.repeat(self.num_estimators, 1, 1, 1)
        out = rearrange(x, "(m b) c h w -> b (m c) h w", m=self.num_estimators)
        return rearrange(super().forward(out), "b (m d) -> (m b) d", m=self.num_estimators)


def mimo_wideresnet28x10(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    conv_bias: bool = True,
    dropout_rate: float = 0.3,
    groups: int = 1,
    style: Literal["imagenet", "cifar"] = "imagenet",
    activation_fn: Callable = relu,
    normalization_layer: type[nn.Module] = nn.BatchNorm2d,
) -> _MIMOWideResNet:
    """MIMO of Wide-ResNet-28x10.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of classes to predict.
        num_estimators (int): Number of estimators in the ensemble.
        groups (int): Number of subgroups in the convolutions.
        conv_bias (bool): Whether to use bias in convolutions. Defaults to
            ``True``.
        dropout_rate (float, optional): Dropout rate. Defaults to ``0.3``.
        style (str, optional): Whether to use the ImageNet or CIFAR
            structure. Defaults to ``imagenet``.
        activation_fn (Callable, optional): Activation function. Defaults to
            ``torch.nn.functional.relu``.
        normalization_layer (nn.Module, optional): Normalization layer.
            Defaults to ``torch.nn.BatchNorm2d``.

    Returns:
        _MIMOWideResNet: A MIMO Wide-ResNet-28x10.
    """
    return _MIMOWideResNet(
        depth=28,
        widen_factor=10,
        in_channels=in_channels,
        num_classes=num_classes,
        num_estimators=num_estimators,
        conv_bias=conv_bias,
        dropout_rate=dropout_rate,
        groups=groups,
        style=style,
        activation_fn=activation_fn,
        normalization_layer=normalization_layer,
    )
