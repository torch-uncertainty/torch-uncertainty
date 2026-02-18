from typing import Literal

import torch.nn.functional as F
from einops import repeat
from torch import Tensor, nn

from torch_uncertainty.layers import BatchConv2d, BatchLinear

from .utils import ResNetStyle, get_resnet_num_blocks

__all__ = [
    "batched_resnet",
]


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int,
        num_estimators: int,
        conv_bias: bool,
        dropout_rate: float,
        groups: int,
        normalization_layer: type[nn.Module],
    ) -> None:
        super().__init__()
        self.conv1 = BatchConv2d(
            in_planes,
            planes,
            kernel_size=3,
            num_estimators=num_estimators,
            groups=groups,
            stride=stride,
            padding=1,
            bias=conv_bias,
        )
        self.bn1 = normalization_layer(planes)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv2 = BatchConv2d(
            planes,
            planes,
            kernel_size=3,
            num_estimators=num_estimators,
            groups=groups,
            stride=1,
            padding=1,
            bias=conv_bias,
        )
        self.bn2 = normalization_layer(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    groups=groups,
                    kernel_size=1,
                    stride=stride,
                    bias=conv_bias,
                ),
                normalization_layer(self.expansion * planes),
            )

    def forward(self, inputs: Tensor) -> Tensor:
        out = F.relu(self.dropout(self.bn1(self.conv1(inputs))))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(inputs)
        return F.relu(out)


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int,
        num_estimators: int,
        conv_bias: bool,
        dropout_rate: float,
        groups: int,
        normalization_layer: type[nn.Module],
    ) -> None:
        super().__init__()
        self.conv1 = BatchConv2d(
            in_planes,
            planes,
            kernel_size=1,
            num_estimators=num_estimators,
            groups=groups,
            bias=conv_bias,
        )
        self.bn1 = normalization_layer(planes)
        self.conv2 = BatchConv2d(
            planes,
            planes,
            kernel_size=3,
            num_estimators=num_estimators,
            groups=groups,
            stride=stride,
            padding=1,
            bias=conv_bias,
        )
        self.bn2 = normalization_layer(planes)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv3 = BatchConv2d(
            planes,
            self.expansion * planes,
            num_estimators=num_estimators,
            groups=groups,
            kernel_size=1,
            bias=conv_bias,
        )
        self.bn3 = normalization_layer(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                BatchConv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    num_estimators=num_estimators,
                    groups=groups,
                    stride=stride,
                    bias=conv_bias,
                ),
                normalization_layer(self.expansion * planes),
            )

    def forward(self, inputs: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(inputs)))
        out = F.relu(self.dropout(self.bn2(self.conv2(out))))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(inputs)
        return F.relu(out)


class _BatchedResNet(nn.Module):
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
        width_multiplier: int = 1,
        style: ResNetStyle = ResNetStyle.IMAGENET,
        in_planes: int = 64,
        normalization_layer: type[nn.Module] = nn.BatchNorm2d,
        repeat_strategy: Literal["legacy", "paper"] = "legacy",
    ) -> None:
        if repeat_strategy not in ("legacy", "paper"):
            raise ValueError(f"Unknown repeat_strategy. Got {repeat_strategy}.")

        super().__init__()
        self.in_planes = in_planes * width_multiplier
        block_planes = in_planes * width_multiplier

        self.num_estimators = num_estimators
        self.dropout_rate = dropout_rate
        self.width_multiplier = width_multiplier
        self.repeat_strategy = repeat_strategy

        if style == ResNetStyle.IMAGENET:
            self.conv1 = BatchConv2d(
                3,
                block_planes,
                kernel_size=7,
                stride=2,
                padding=3,
                num_estimators=num_estimators,
                groups=groups,
                bias=False,
            )
        else:  # style == "cifar":
            self.conv1 = BatchConv2d(
                in_channels,
                block_planes,
                kernel_size=3,
                stride=1,
                padding=1,
                num_estimators=num_estimators,
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

        self.linear = BatchLinear(
            block_planes * linear_multiplier * block.expansion,
            num_classes,
            num_estimators=num_estimators,
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
                    conv_bias=conv_bias,
                    dropout_rate=dropout_rate,
                    num_estimators=num_estimators,
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


def batched_resnet(
    in_channels: int,
    num_classes: int,
    arch: int,
    num_estimators: int,
    conv_bias: bool = True,
    dropout_rate: float = 0,
    width_multiplier: float = 1.0,
    groups: int = 1,
    style: ResNetStyle | Literal["imagenet", "cifar"] = ResNetStyle.IMAGENET,
    normalization_layer: type[nn.Module] = nn.BatchNorm2d,
    repeat_strategy: Literal["legacy", "paper"] = "paper",
) -> _BatchedResNet:
    """BatchEnsemble of ResNet.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of classes to predict.
        arch (int): The architecture of the ResNet.
        num_estimators (int): Number of estimators in the ensemble.
        conv_bias (bool): Whether to use bias in convolutions. Defaults to ``True``.
        dropout_rate (float): Dropout rate. Defaults to ``0``.
        width_multiplier (float): Width multiplier. Defaults to ``1.0``.
        groups (int): Number of groups within each estimator.
        style (ResNetStyle | Literal["imagenet", "cifar"]): Whether to use the ImageNet or CIFAR
            structure. Defaults to ``ResNetStyle.IMAGENET``.
        normalization_layer (nn.Module, optional): Normalization layer.
        repeat_strategy (Literal["legacy", "paper"], optional): The repeatrepeat_strategy ("legacy"|"paper", optional): The repeat
            strategy to use during training:

            - "legacy": Repeat inputs for each estimator during both training
              and evaluation.
            - "paper"(default): Repeat inputs for each estimator only during
              evaluation.

    Returns:
        _BatchedResNet: A BatchEnsemble-style ResNet.
    """
    if isinstance(style, str):
        style = ResNetStyle(style)

    block = _BasicBlock if arch in [18, 20, 34, 44, 56, 110, 1202] else _Bottleneck
    in_planes = 16 if arch in [20, 44, 56, 110, 1202] else 64
    return _BatchedResNet(
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
        repeat_strategy=repeat_strategy,
    )
