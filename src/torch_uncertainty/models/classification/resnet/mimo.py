from typing import Literal

import torch
from einops import rearrange
from torch import nn

from .std import _BasicBlock, _Bottleneck, _ResNet
from .utils import ResNetStyle, get_resnet_num_blocks

__all__ = [
    "mimo_resnet",
]


class _MIMOResNet(_ResNet):
    def __init__(
        self,
        block: type[_BasicBlock | _Bottleneck],
        num_blocks: list[int],
        in_channels: int,
        num_classes: int,
        num_estimators: int,
        conv_bias: bool,
        dropout_rate: float,
        groups: int = 1,
        style: ResNetStyle = ResNetStyle.IMAGENET,
        in_planes: int = 64,
        normalization_layer: type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__(
            block=block,
            num_blocks=num_blocks,
            in_channels=in_channels * num_estimators,
            num_classes=num_classes * num_estimators,
            conv_bias=conv_bias,
            dropout_rate=dropout_rate,
            groups=groups,
            style=style,
            in_planes=in_planes,
            normalization_layer=normalization_layer,
        )

        self.num_estimators = num_estimators

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            x = x.repeat(self.num_estimators, 1, 1, 1)
        out = rearrange(x, "(m b) c h w -> b (m c) h w", m=self.num_estimators)
        out = super().forward(out)
        return rearrange(out, "b (m d) -> (m b) d", m=self.num_estimators)


def mimo_resnet(
    in_channels: int,
    num_classes: int,
    arch: int,
    num_estimators: int,
    conv_bias: bool = True,
    dropout_rate: float = 0.0,
    width_multiplier: float = 1.0,
    groups: int = 1,
    style: ResNetStyle | Literal["imagenet", "cifar"] = ResNetStyle.IMAGENET,
    normalization_layer: type[nn.Module] = nn.BatchNorm2d,
) -> _MIMOResNet:
    """MIMO ResNet.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of classes to predict.
        arch (int): The architecture of the ResNet.
        num_estimators (int): Number of estimators in the ensemble.
        conv_bias (bool, optional): Whether to use bias in convolutional layers. Defaults to ``True``.
        dropout_rate (float, optional): Dropout rate. Defaults to ``0.0``.
        width_multiplier (float, optional): Width multiplier. Defaults to ``1.0``.
        groups (int, optional): Number of groups for grouped convolution. Defaults to ``1``.
        style (ResNetStyle | Literal["imagenet", "cifar"]): Whether to use the ImageNet or CIFAR
            structure. Defaults to ``ResNetStyle.IMAGENET``.
        normalization_layer (nn.Module, optional): Normalization layer.
            Defaults to ``torch.nn.BatchNorm2d``.

    Returns:
        _MIMOResNet: A MIMO-style ResNet.
    """
    if isinstance(style, str):
        style = ResNetStyle(style)

    block = _BasicBlock if arch in [18, 20, 34, 44, 56, 110, 1202] else _Bottleneck
    in_planes = 16 if arch in [20, 44, 56, 110, 1202] else 64
    return _MIMOResNet(
        block=block,
        num_blocks=get_resnet_num_blocks(arch),
        in_channels=in_channels,
        num_classes=num_classes,
        num_estimators=num_estimators,
        conv_bias=conv_bias,
        dropout_rate=dropout_rate,
        groups=groups,
        style=style,
        in_planes=int(in_planes * width_multiplier),
        normalization_layer=normalization_layer,
    )
