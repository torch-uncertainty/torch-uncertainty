from typing import Literal

import torch.nn.functional as F
from einops import repeat
from torch import Tensor, nn

from torch_uncertainty.layers import MaskedConv2d, MaskedLinear

from .utils import ResNetStyle, get_resnet_num_blocks

__all__ = [
    "masked_resnet",
]


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int,
        num_estimators: int,
        scale: float,
        conv_bias: bool,
        dropout_rate: float,
        groups: int,
        normalization_layer: type[nn.Module],
    ) -> None:
        super().__init__()

        self.conv1 = MaskedConv2d(
            in_planes,
            planes,
            kernel_size=3,
            num_estimators=num_estimators,
            scale=scale,
            groups=groups,
            stride=stride,
            padding=1,
            bias=conv_bias,
        )
        self.bn1 = normalization_layer(planes)
        self.conv2 = MaskedConv2d(
            planes,
            planes,
            kernel_size=3,
            num_estimators=num_estimators,
            scale=scale,
            stride=1,
            padding=1,
            groups=groups,
            bias=conv_bias,
        )
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.bn2 = normalization_layer(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                MaskedConv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    num_estimators=num_estimators,
                    scale=scale,
                    stride=stride,
                    groups=groups,
                    bias=conv_bias,
                ),
                normalization_layer(self.expansion * planes),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.dropout(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int,
        num_estimators: int,
        scale: float,
        conv_bias: bool,
        dropout_rate: float,
        groups: int,
        normalization_layer: type[nn.Module],
    ) -> None:
        super().__init__()

        self.conv1 = MaskedConv2d(
            in_planes,
            planes,
            kernel_size=1,
            num_estimators=num_estimators,
            scale=scale,
            groups=groups,
            bias=conv_bias,
        )
        self.bn1 = normalization_layer(planes)
        self.conv2 = MaskedConv2d(
            planes,
            planes,
            kernel_size=3,
            num_estimators=num_estimators,
            scale=scale,
            stride=stride,
            padding=1,
            groups=groups,
            bias=conv_bias,
        )
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.bn2 = normalization_layer(planes)
        self.conv3 = MaskedConv2d(
            planes,
            self.expansion * planes,
            kernel_size=1,
            num_estimators=num_estimators,
            scale=scale,
            groups=groups,
            bias=conv_bias,
        )
        self.bn3 = normalization_layer(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                MaskedConv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    num_estimators=num_estimators,
                    scale=scale,
                    stride=stride,
                    groups=groups,
                    bias=conv_bias,
                ),
                normalization_layer(self.expansion * planes),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.dropout(self.bn2(self.conv2(out))))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)


class _MaskedResNet(nn.Module):
    def __init__(
        self,
        block: type[_BasicBlock | _Bottleneck],
        num_blocks: list[int],
        in_channels: int,
        num_classes: int,
        num_estimators: int,
        dropout_rate: float,
        scale: float = 2.0,
        conv_bias: bool = True,
        groups: int = 1,
        style: ResNetStyle = ResNetStyle.IMAGENET,
        in_planes: int = 64,
        normalization_layer: type[nn.Module] = nn.BatchNorm2d,
        repeat_strategy: Literal["legacy", "paper"] = "legacy",
    ) -> None:
        if repeat_strategy not in ("legacy", "paper"):
            raise ValueError(f"Unknown repeat_strategy. Got {repeat_strategy}.")

        super().__init__()
        self.in_channels = in_channels
        self.in_planes = in_planes
        block_planes = self.in_planes

        self.num_estimators = num_estimators
        self.repeat_strategy = repeat_strategy

        if style == ResNetStyle.IMAGENET:
            self.conv1 = nn.Conv2d(
                self.in_channels,
                block_planes,
                kernel_size=7,
                stride=2,
                padding=3,
                groups=groups,
                bias=False,
            )
        else:  # style == "cifar":
            self.conv1 = nn.Conv2d(
                self.in_channels,
                block_planes,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=groups,
                bias=False,
            )

        self.bn1 = normalization_layer(block_planes)

        if style == ResNetStyle.IMAGENET:
            self.optional_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.optional_pool = nn.Identity()

        self.layer1 = self._make_layer(
            block,
            block_planes,
            num_blocks[0],
            stride=1,
            num_estimators=num_estimators,
            conv_bias=conv_bias,
            dropout_rate=dropout_rate,
            scale=scale,
            groups=groups,
            normalization_layer=normalization_layer,
        )
        self.layer2 = self._make_layer(
            block,
            block_planes * 2,
            num_blocks[1],
            stride=2,
            num_estimators=num_estimators,
            conv_bias=conv_bias,
            dropout_rate=dropout_rate,
            scale=scale,
            groups=groups,
            normalization_layer=normalization_layer,
        )
        self.layer3 = self._make_layer(
            block,
            block_planes * 4,
            num_blocks[2],
            stride=2,
            num_estimators=num_estimators,
            conv_bias=conv_bias,
            dropout_rate=dropout_rate,
            scale=scale,
            groups=groups,
            normalization_layer=normalization_layer,
        )
        if len(num_blocks) == 4:
            self.layer4 = self._make_layer(
                block,
                block_planes * 8,
                num_blocks[3],
                stride=2,
                num_estimators=num_estimators,
                conv_bias=conv_bias,
                dropout_rate=dropout_rate,
                scale=scale,
                groups=groups,
                normalization_layer=normalization_layer,
            )
            linear_multiplier = 8
        else:
            self.layer4 = nn.Identity()
            linear_multiplier = 4

        self.final_dropout = nn.Dropout(p=dropout_rate)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten(1)

        self.linear = MaskedLinear(
            block_planes * linear_multiplier * block.expansion,
            num_classes,
            num_estimators,
            scale=scale,
        )

    def _make_layer(
        self,
        block: type[_BasicBlock | _Bottleneck],
        planes: int,
        num_blocks: int,
        stride: int,
        num_estimators: int,
        conv_bias: bool,
        dropout_rate: float,
        scale: float,
        groups: int,
        normalization_layer: type[nn.Module],
    ) -> nn.Module:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    in_planes=self.in_planes,
                    planes=planes,
                    stride=stride,
                    num_estimators=num_estimators,
                    conv_bias=conv_bias,
                    dropout_rate=dropout_rate,
                    scale=scale,
                    groups=groups,
                    normalization_layer=normalization_layer,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.repeat_strategy == "legacy":
            x = repeat(x, "b ... -> (m b) ...", m=self.num_estimators)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.optional_pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.pool(out)
        out = self.final_dropout(self.flatten(out))
        return self.linear(out)


def masked_resnet(
    in_channels: int,
    num_classes: int,
    arch: int,
    num_estimators: int,
    scale: float,
    width_multiplier: float = 1.0,
    groups: int = 1,
    conv_bias: bool = True,
    dropout_rate: float = 0,
    style: ResNetStyle | Literal["imagenet", "cifar"] = ResNetStyle.IMAGENET,
    normalization_layer: type[nn.Module] = nn.BatchNorm2d,
    repeat_strategy: Literal["legacy", "paper"] = "paper",
) -> _MaskedResNet:
    """Masksembles of ResNet.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of classes to predict.
        arch (int): The architecture of the ResNet.
        num_estimators (int): Number of estimators in the ensemble.
        scale (float): The scale of the mask.
        width_multiplier (float): Width multiplier. Defaults to 1.
        groups (int): Number of groups within each estimator. Defaults to 1.
        conv_bias (bool): Whether to use bias in convolutions. Defaults to ``True``.
        dropout_rate (float): Dropout rate. Defaults to ``0``.
        style (ResNetStyle | Literal["imagenet", "cifar"]): Whether to use the ImageNet or CIFAR
            structure. Defaults to ``ResNetStyle.IMAGENET``.
        normalization_layer (nn.Module, optional): Normalization layer.
        repeat_strategy ("legacy"|"paper", optional): The repeat
            strategy to use during training:

            - "legacy": Repeat inputs for each estimator during both training
              and evaluation.
            - "paper"(default): Repeat inputs for each estimator only during
              evaluation.

    Returns:
        _MaskedResNet: A Masksembles-style ResNet.
    """
    if isinstance(style, str):
        style = ResNetStyle(style)

    block = _BasicBlock if arch in [18, 20, 34, 44, 56, 110, 1202] else _Bottleneck
    in_planes = 16 if arch in [20, 44, 56, 110, 1202] else 64
    return _MaskedResNet(
        block=block,
        num_blocks=get_resnet_num_blocks(arch),
        num_classes=num_classes,
        in_channels=in_channels,
        num_estimators=num_estimators,
        scale=scale,
        groups=groups,
        conv_bias=conv_bias,
        dropout_rate=dropout_rate,
        style=style,
        in_planes=int(in_planes * width_multiplier),
        normalization_layer=normalization_layer,
        repeat_strategy=repeat_strategy,
    )
