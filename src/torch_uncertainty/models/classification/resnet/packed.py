from typing import Any, Literal

import torch.nn.functional as F
from torch import Tensor, nn

from torch_uncertainty.layers import PackedConv2d, PackedLinear
from torch_uncertainty.utils import load_hf

from .utils import ResNetStyle, get_resnet_num_blocks

__all__ = [
    "packed_resnet",
]

weight_ids = {
    "10": {
        "18": "pe_resnet18_c10",
        "32": None,
        "50": "pe_resnet50_c10",
        "101": None,
        "152": None,
    },
    "100": {
        "18": "pe_resnet18_c100",
        "32": None,
        "50": "pe_resnet50_c100",
        "101": None,
        "152": None,
    },
    "1000": {
        "18": None,
        "32": None,
        "50": "pe_resnet50_in1k",
        "101": None,
        "152": None,
    },
}


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int,
        alpha: int,
        num_estimators: int,
        gamma: int,
        conv_bias: bool,
        dropout_rate: float,
        groups: int,
        normalization_layer: type[nn.Module],
    ) -> None:
        super().__init__()

        # No subgroups for the first layer
        self.conv1 = PackedConv2d(
            in_planes,
            planes,
            kernel_size=3,
            alpha=alpha,
            num_estimators=num_estimators,
            groups=groups,
            stride=stride,
            padding=1,
            bias=conv_bias,
        )
        self.bn1 = normalization_layer(planes * alpha)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv2 = PackedConv2d(
            planes,
            planes,
            kernel_size=3,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            groups=groups,
            stride=1,
            padding=1,
            bias=conv_bias,
        )
        self.bn2 = normalization_layer(planes * alpha)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                PackedConv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    alpha=alpha,
                    num_estimators=num_estimators,
                    gamma=gamma,
                    groups=groups,
                    stride=stride,
                    bias=conv_bias,
                ),
                normalization_layer(self.expansion * planes * alpha),
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
        alpha: int,
        num_estimators: int,
        gamma: int,
        conv_bias: bool,
        dropout_rate: float,
        groups: int,
        normalization_layer: type[nn.Module],
    ) -> None:
        super().__init__()

        # No subgroups for the first layer
        self.conv1 = PackedConv2d(
            in_planes,
            planes,
            kernel_size=1,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=1,  # No groups from gamma in the first layer
            groups=groups,
            bias=conv_bias,
        )
        self.bn1 = normalization_layer(planes * alpha)
        self.conv2 = PackedConv2d(
            planes,
            planes,
            kernel_size=3,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            stride=stride,
            padding=1,
            groups=groups,
            bias=conv_bias,
        )
        self.bn2 = normalization_layer(planes * alpha)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv3 = PackedConv2d(
            planes,
            self.expansion * planes,
            kernel_size=1,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            groups=groups,
            bias=conv_bias,
        )
        self.bn3 = normalization_layer(self.expansion * planes * alpha)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                PackedConv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    alpha=alpha,
                    num_estimators=num_estimators,
                    gamma=gamma,
                    groups=groups,
                    stride=stride,
                    bias=conv_bias,
                ),
                normalization_layer(self.expansion * planes * alpha),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.dropout(self.bn2(self.conv2(out))))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)


class _PackedResNet(nn.Module):
    def __init__(
        self,
        block: type[_BasicBlock | _Bottleneck],
        num_blocks: list[int],
        in_channels: int,
        num_classes: int,
        conv_bias: bool,
        num_estimators: int,
        dropout_rate: float,
        alpha: int = 2,
        gamma: int = 1,
        groups: int = 1,
        style: ResNetStyle = ResNetStyle.IMAGENET,
        in_planes: int = 64,
        normalization_layer: type[nn.Module] = nn.BatchNorm2d,
        linear_implementation: str = "conv1d",
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.alpha = alpha
        self.gamma = gamma
        self.groups = groups
        self.num_estimators = num_estimators

        self.in_planes = in_planes
        block_planes = in_planes

        if style == ResNetStyle.IMAGENET:
            self.conv1 = PackedConv2d(
                self.in_channels,
                block_planes,
                kernel_size=7,
                stride=2,
                padding=3,
                alpha=alpha,
                num_estimators=num_estimators,
                gamma=1,  # No groups for the first layer
                groups=groups,
                bias=conv_bias,
                first=True,
            )
        else:  # style == "cifar":
            self.conv1 = PackedConv2d(
                self.in_channels,
                block_planes,
                kernel_size=3,
                stride=1,
                padding=1,
                alpha=alpha,
                num_estimators=num_estimators,
                gamma=1,  # No groups for the first layer
                groups=groups,
                bias=conv_bias,
                first=True,
            )

        self.bn1 = normalization_layer(block_planes * alpha)

        if style == ResNetStyle.IMAGENET:
            self.optional_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.optional_pool = nn.Identity()

        self.layer1 = self._make_layer(
            block,
            block_planes,
            num_blocks[0],
            stride=1,
            alpha=alpha,
            num_estimators=num_estimators,
            conv_bias=conv_bias,
            dropout_rate=dropout_rate,
            gamma=gamma,
            groups=groups,
            normalization_layer=normalization_layer,
        )
        self.layer2 = self._make_layer(
            block,
            block_planes * 2,
            num_blocks[1],
            stride=2,
            alpha=alpha,
            num_estimators=num_estimators,
            conv_bias=conv_bias,
            dropout_rate=dropout_rate,
            gamma=gamma,
            groups=groups,
            normalization_layer=normalization_layer,
        )
        self.layer3 = self._make_layer(
            block,
            block_planes * 4,
            num_blocks[2],
            stride=2,
            alpha=alpha,
            num_estimators=num_estimators,
            conv_bias=conv_bias,
            dropout_rate=dropout_rate,
            gamma=gamma,
            groups=groups,
            normalization_layer=normalization_layer,
        )
        if len(num_blocks) == 4:
            self.layer4 = self._make_layer(
                block,
                block_planes * 8,
                num_blocks[3],
                stride=2,
                alpha=alpha,
                num_estimators=num_estimators,
                conv_bias=conv_bias,
                dropout_rate=dropout_rate,
                gamma=gamma,
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

        self.linear = PackedLinear(
            block_planes * linear_multiplier * block.expansion,
            num_classes,
            alpha=alpha,
            num_estimators=num_estimators,
            last=True,
            implementation=linear_implementation,
        )

    def _make_layer(
        self,
        block: type[_BasicBlock | _Bottleneck],
        planes: int,
        num_blocks: int,
        stride: int,
        alpha: int,
        num_estimators: int,
        conv_bias: bool,
        dropout_rate: float,
        gamma: int,
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
                    alpha=alpha,
                    num_estimators=num_estimators,
                    conv_bias=conv_bias,
                    dropout_rate=dropout_rate,
                    gamma=gamma,
                    groups=groups,
                    normalization_layer=normalization_layer,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.optional_pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = self.final_dropout(self.flatten(out))
        return self.linear(out)

    def check_config(self, config: dict[str, Any]) -> bool:
        """Check if the pretrained configuration matches the current model."""
        return (
            (config["alpha"] == self.alpha)
            * (config["gamma"] == self.gamma)
            * (config["groups"] == self.groups)
            * (config["num_estimators"] == self.num_estimators)
        )


def packed_resnet(
    in_channels: int,
    num_classes: int,
    arch: int,
    num_estimators: int,
    alpha: int,
    gamma: int,
    conv_bias: bool = False,
    width_multiplier: float = 1.0,
    groups: int = 1,
    dropout_rate: float = 0,
    style: Literal["imagenet", "cifar"] = "imagenet",
    normalization_layer: type[nn.Module] = nn.BatchNorm2d,
    pretrained: bool = False,
    linear_implementation: str = "conv1d",
) -> _PackedResNet:
    """Packed-Ensembles of ResNet.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of classes to predict.
        arch (int): The architecture of the ResNet.
        conv_bias (bool): Whether to use bias in convolutions. Defaults to ``True``.
        dropout_rate (float): Dropout rate. Defaults to ``0``.
        num_estimators (int): Number of estimators in the ensemble.
        alpha (int): Expansion factor affecting the width of the estimators.
        gamma (int): Number of groups within each estimator.
        width_multiplier (float): Width multiplier. Defaults to ``1``.
        groups (int): Number of groups within each estimator group.
        style (ResNetStyle | Literal["imagenet", "cifar"]): Whether to use the ImageNet or CIFAR
            structure. Defaults to ``ResNetStyle.IMAGENET``.
        normalization_layer (nn.Module): Normalization layer. Defaults to ``nn.BatchNorm2d``.
        pretrained (bool, optional): Whether to load pretrained weights.
            Defaults to ``False``.
        linear_implementation (str, optional): Implementation of the
            packed linear layer. Defaults to ``"conv1d"``.

    Returns:
        _PackedResNet: A Packed-Ensembles ResNet.
    """
    if isinstance(style, str):
        style = ResNetStyle(style)

    block = _BasicBlock if arch in [18, 20, 34, 44, 56, 110, 1202] else _Bottleneck
    in_planes = 16 if arch in [20, 44, 56, 110, 1202] else 64
    net = _PackedResNet(
        block=block,
        num_blocks=get_resnet_num_blocks(arch),
        in_channels=in_channels,
        num_estimators=num_estimators,
        alpha=alpha,
        gamma=gamma,
        conv_bias=conv_bias,
        dropout_rate=dropout_rate,
        groups=groups,
        num_classes=num_classes,
        style=style,
        in_planes=int(in_planes * width_multiplier),
        normalization_layer=normalization_layer,
        linear_implementation=linear_implementation,
    )
    if pretrained:  # coverage: ignore
        weights = weight_ids[str(num_classes)][str(arch)]
        if weights is None:
            raise ValueError("No pretrained weights for this configuration")
        state_dict, config = load_hf(weights)
        if not net.check_config(config):
            raise ValueError("Pretrained weights do not match current configuration.")
        net.load_state_dict(state_dict)
    return net
