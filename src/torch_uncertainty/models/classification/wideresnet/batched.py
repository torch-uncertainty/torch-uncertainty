from collections.abc import Callable
from typing import Literal

from einops import repeat
from torch import Tensor, nn
from torch.nn.functional import relu

from torch_uncertainty.layers import BatchConv2d, BatchLinear

__all__ = [
    "batched_wideresnet28x10",
]


class _WideBasicBlock(nn.Module):
    def __init__(
        self,
        in_planes: int,
        planes: int,
        dropout_rate: float,
        stride: int,
        num_estimators: int,
        groups: int,
        conv_bias: bool,
        activation_fn: Callable,
        normalization_layer: type[nn.Module],
    ) -> None:
        super().__init__()
        self.activation_fn = activation_fn
        self.conv1 = BatchConv2d(
            in_planes,
            planes,
            kernel_size=3,
            num_estimators=num_estimators,
            padding=1,
            groups=groups,
            bias=conv_bias,
        )
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.bn1 = normalization_layer(planes)
        self.conv2 = BatchConv2d(
            planes,
            planes,
            kernel_size=3,
            num_estimators=num_estimators,
            stride=stride,
            padding=1,
            groups=groups,
            bias=conv_bias,
        )
        self.bn2 = normalization_layer(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                BatchConv2d(
                    in_planes,
                    planes,
                    kernel_size=1,
                    num_estimators=num_estimators,
                    stride=stride,
                    groups=groups,
                    bias=conv_bias,
                ),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = self.activation_fn(self.bn1(self.dropout(self.conv1(x))))
        out = self.conv2(out)
        out += self.shortcut(x)
        return self.activation_fn(self.bn2(out))


class _BatchWideResNet(nn.Module):
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
        repeat_strategy: Literal["legacy", "paper"] = "legacy",
    ) -> None:
        if repeat_strategy not in ("legacy", "paper"):
            raise ValueError(f"Unknown repeat_strategy. Got {repeat_strategy}.")

        super().__init__()
        self.num_estimators = num_estimators
        self.activation_fn = activation_fn
        self.in_planes = 16
        self.repeat_strategy = repeat_strategy

        if (depth - 4) % 6 != 0:
            raise ValueError(f"Wide-resnet depth should be 6n+4. Got {depth}.")
        num_blocks = (depth - 4) // 6
        num_stages = [
            16,
            16 * widen_factor,
            32 * widen_factor,
            64 * widen_factor,
        ]

        if style == "imagenet":
            self.conv1 = BatchConv2d(
                in_channels,
                num_stages[0],
                num_estimators=self.num_estimators,
                groups=groups,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=conv_bias,
            )
        elif style == "cifar":
            self.conv1 = BatchConv2d(
                in_channels,
                num_stages[0],
                num_estimators=self.num_estimators,
                groups=groups,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=conv_bias,
            )
        else:
            raise ValueError(f"Unknown WideResNet style: {style}. ")

        self.bn1 = normalization_layer(num_stages[0])

        if style == "imagenet":
            self.optional_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.optional_pool = nn.Identity()

        self.layer1 = self._wide_layer(
            _WideBasicBlock,
            num_stages[1],
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
            stride=1,
            num_estimators=self.num_estimators,
            groups=groups,
            conv_bias=conv_bias,
            activation_fn=activation_fn,
            normalization_layer=normalization_layer,
        )
        self.layer2 = self._wide_layer(
            _WideBasicBlock,
            num_stages[2],
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
            stride=2,
            num_estimators=self.num_estimators,
            groups=groups,
            conv_bias=conv_bias,
            activation_fn=activation_fn,
            normalization_layer=normalization_layer,
        )
        self.layer3 = self._wide_layer(
            _WideBasicBlock,
            num_stages[3],
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
            stride=2,
            num_estimators=self.num_estimators,
            groups=groups,
            conv_bias=conv_bias,
            activation_fn=activation_fn,
            normalization_layer=normalization_layer,
        )

        self.final_dropout = nn.Dropout(p=dropout_rate)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten(1)
        self.linear = BatchLinear(
            num_stages[3],
            num_classes,
            num_estimators,
        )

    def _wide_layer(
        self,
        block: type[nn.Module],
        planes: int,
        num_blocks: int,
        dropout_rate: float,
        stride: int,
        num_estimators: int,
        groups: int,
        conv_bias: bool,
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
                    num_estimators=num_estimators,
                    groups=groups,
                    activation_fn=activation_fn,
                    normalization_layer=normalization_layer,
                )
            )
            self.in_planes = planes
        return nn.Sequential(*layers)

    def feats_forward(self, x: Tensor) -> Tensor:
        if not self.training or self.repeat_strategy == "legacy":
            x = repeat(x, "b ... -> (m b) ...", m=self.num_estimators)
        out = self.activation_fn(self.bn1(self.conv1(x)))
        out = self.optional_pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)
        return self.final_dropout(self.flatten(out))

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(self.feats_forward(x))


def batched_wideresnet28x10(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    conv_bias: bool = True,
    dropout_rate: float = 0.3,
    groups: int = 1,
    style: Literal["imagenet", "cifar"] = "imagenet",
    activation_fn: Callable = relu,
    normalization_layer: type[nn.Module] = nn.BatchNorm2d,
    repeat_strategy: Literal["legacy", "paper"] = "paper",
) -> _BatchWideResNet:
    """BatchEnsemble of Wide-ResNet-28x10.

    Args:
        in_channels (int): Number of input channels.
        num_estimators (int): Number of estimators in the ensemble.
        conv_bias (bool): Whether to use bias in convolutions. Defaults to
            ``True``.
        dropout_rate (float, optional): Dropout rate. Defaults to ``0.3``.
        num_classes (int): Number of classes to predict.
        groups (int): Number of groups in the convolutions. Defaults to ``1``.
        style (bool, optional): Whether to use the ImageNet
            structure. Defaults to ``True``.
        activation_fn (Callable, optional): Activation function. Defaults to
            ``torch.nn.functional.relu``.
        normalization_layer (nn.Module, optional): Normalization layer.
            Defaults to ``torch.nn.BatchNorm2d``.
        repeat_strategy ("legacy"|"paper", optional): The repeat
            strategy to use during training:

            - "legacy": Repeat inputs for each estimator during both training
              and evaluation.
            - "paper"(default): Repeat inputs for each estimator only during
              evaluation.

    Returns:
        _BatchWideResNet: A BatchEnsemble-style Wide-ResNet-28x10.
    """
    return _BatchWideResNet(
        in_channels=in_channels,
        depth=28,
        widen_factor=10,
        conv_bias=conv_bias,
        dropout_rate=dropout_rate,
        num_classes=num_classes,
        num_estimators=num_estimators,
        groups=groups,
        style=style,
        activation_fn=activation_fn,
        normalization_layer=normalization_layer,
        repeat_strategy=repeat_strategy,
    )
