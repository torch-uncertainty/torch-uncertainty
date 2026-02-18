from collections.abc import Callable
from typing import Literal

from torch import Tensor, nn
from torch.nn.functional import relu

from torch_uncertainty.layers import PackedConv2d, PackedLinear

__all__ = [
    "packed_wideresnet28x10",
]


class _WideBasicBlock(nn.Module):
    def __init__(
        self,
        in_planes: int,
        planes: int,
        conv_bias: bool,
        dropout_rate: float,
        stride: int,
        alpha: int,
        num_estimators: int,
        gamma: int,
        groups: int,
        activation_fn: Callable,
        normalization_layer: type[nn.Module],
    ) -> None:
        super().__init__()
        self.activation_fn = activation_fn
        self.conv1 = PackedConv2d(
            in_planes,
            planes,
            kernel_size=3,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            groups=groups,
            padding=1,
            bias=conv_bias,
        )
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.bn1 = normalization_layer(alpha * planes)
        self.conv2 = PackedConv2d(
            planes,
            planes,
            kernel_size=3,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            groups=groups,
            stride=stride,
            padding=1,
            bias=conv_bias,
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                PackedConv2d(
                    in_planes,
                    planes,
                    kernel_size=1,
                    alpha=alpha,
                    num_estimators=num_estimators,
                    gamma=gamma,
                    groups=groups,
                    stride=stride,
                    bias=conv_bias,
                ),
            )
        self.bn2 = normalization_layer(alpha * planes)

    def forward(self, x: Tensor) -> Tensor:
        out = self.activation_fn(self.bn1(self.dropout(self.conv1(x))))
        out = self.conv2(out)
        out += self.shortcut(x)
        return self.activation_fn(self.bn2(out))


class _PackedWideResNet(nn.Module):
    def __init__(
        self,
        depth: int,
        widen_factor: int,
        in_channels: int,
        num_classes: int,
        conv_bias: bool,
        dropout_rate: float,
        num_estimators: int,
        alpha: int = 2,
        gamma: int = 1,
        groups: int = 1,
        style: Literal["imagenet", "cifar"] = "imagenet",
        activation_fn: Callable = relu,
        normalization_layer: type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        self.num_estimators = num_estimators
        self.activation_fn = activation_fn
        self.in_planes = 16

        if (depth - 4) % 6 != 0:
            raise ValueError(f"Wide-resnet depth should be 6n+4. Got {depth}.")
        num_blocks = int((depth - 4) / 6)
        num_stages = [
            16,
            16 * widen_factor,
            32 * widen_factor,
            64 * widen_factor,
        ]

        if style == "imagenet":
            self.conv1 = PackedConv2d(
                in_channels,
                num_stages[0],
                kernel_size=7,
                alpha=alpha,
                num_estimators=self.num_estimators,
                stride=2,
                padding=3,
                gamma=1,  # No groups for the first layer
                groups=groups,
                bias=conv_bias,
                first=True,
            )
        elif style == "cifar":
            self.conv1 = PackedConv2d(
                in_channels,
                num_stages[0],
                kernel_size=3,
                alpha=alpha,
                num_estimators=self.num_estimators,
                stride=1,
                padding=1,
                gamma=gamma,
                groups=groups,
                bias=conv_bias,
                first=True,
            )
        else:
            raise ValueError(f"Unknown WideResNet style: {style}. ")

        self.bn1 = normalization_layer(num_stages[0] * alpha)

        if style == "imagenet":
            self.optional_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.optional_pool = nn.Identity()

        self.layer1 = self._wide_layer(
            _WideBasicBlock,
            num_stages[1],
            num_blocks,
            conv_bias=conv_bias,
            dropout_rate=dropout_rate,
            stride=1,
            alpha=alpha,
            num_estimators=self.num_estimators,
            gamma=gamma,
            groups=groups,
            activation_fn=activation_fn,
            normalization_layer=normalization_layer,
        )
        self.layer2 = self._wide_layer(
            _WideBasicBlock,
            num_stages[2],
            num_blocks,
            conv_bias=conv_bias,
            dropout_rate=dropout_rate,
            stride=2,
            alpha=alpha,
            num_estimators=self.num_estimators,
            gamma=gamma,
            groups=groups,
            activation_fn=activation_fn,
            normalization_layer=normalization_layer,
        )
        self.layer3 = self._wide_layer(
            _WideBasicBlock,
            num_stages[3],
            num_blocks,
            conv_bias=conv_bias,
            dropout_rate=dropout_rate,
            stride=2,
            alpha=alpha,
            num_estimators=self.num_estimators,
            gamma=gamma,
            groups=groups,
            activation_fn=activation_fn,
            normalization_layer=normalization_layer,
        )

        self.final_dropout = nn.Dropout(p=dropout_rate)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten(1)

        self.linear = PackedLinear(
            num_stages[3],
            num_classes,
            alpha=alpha,
            num_estimators=num_estimators,
            last=True,
        )

    def _wide_layer(
        self,
        block: type[_WideBasicBlock],
        planes: int,
        num_blocks: int,
        conv_bias: bool,
        dropout_rate: float,
        stride: int,
        alpha: int,
        num_estimators: int,
        gamma: int,
        groups: int,
        activation_fn: Callable,
        normalization_layer: type[nn.Module],
    ) -> nn.Module:
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(
                block(
                    in_planes=self.in_planes,
                    planes=planes,
                    conv_bias=conv_bias,
                    dropout_rate=dropout_rate,
                    stride=stride,
                    alpha=alpha,
                    num_estimators=num_estimators,
                    gamma=gamma,
                    groups=groups,
                    activation_fn=activation_fn,
                    normalization_layer=normalization_layer,
                )
            )
            self.in_planes = planes
        return nn.Sequential(*layers)

    def feats_forward(self, x: Tensor) -> Tensor:
        out = self.activation_fn(self.bn1(self.conv1(x)))
        out = self.optional_pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)
        return self.final_dropout(self.flatten(out))

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(self.feats_forward(x))


def packed_wideresnet28x10(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    alpha: int,
    gamma: int,
    conv_bias: bool = True,
    dropout_rate: float = 0.3,
    groups: int = 1,
    style: Literal["imagenet", "cifar"] = "imagenet",
    activation_fn: Callable = relu,
    normalization_layer: type[nn.Module] = nn.BatchNorm2d,
) -> _PackedWideResNet:
    """Packed-Ensembles of Wide-ResNet-28x10.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of classes to predict.
        num_estimators (int): Number of estimators in the ensemble.
        alpha (int): Expansion factor affecting the width of the estimators.
        gamma (int): Number of groups within each estimator.
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
        _PackedWideResNet: A Packed-Ensembles Wide-ResNet-28x10.
    """
    return _PackedWideResNet(
        in_channels=in_channels,
        num_classes=num_classes,
        depth=28,
        widen_factor=10,
        conv_bias=conv_bias,
        dropout_rate=dropout_rate,
        num_estimators=num_estimators,
        alpha=alpha,
        gamma=gamma,
        groups=groups,
        style=style,
        activation_fn=activation_fn,
        normalization_layer=normalization_layer,
    )
