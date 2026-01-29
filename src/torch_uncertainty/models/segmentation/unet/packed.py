import torch
from einops import rearrange
from torch import Tensor, nn
from torchvision.transforms import functional as F

from torch_uncertainty.layers.packed import (
    PackedConv2d,
    PackedConvTranspose2d,
)

from .standard import check_unet_parameters


class _PackedDoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        alpha: float,
        num_estimators: int,
        gamma: int,
        mid_channels: int | None = None,
        first: bool = False,
        last: bool = False,
    ) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.conv = nn.Sequential(
            PackedConv2d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=3,
                alpha=alpha,
                num_estimators=num_estimators,
                gamma=gamma,
                padding=1,
                first=first,
            ),
            nn.BatchNorm2d(num_features=int(mid_channels * (num_estimators if last else alpha))),
            nn.ReLU(inplace=True),
            PackedConv2d(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=3,
                alpha=alpha,
                num_estimators=num_estimators,
                gamma=gamma,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=int(out_channels * (num_estimators if last else alpha))),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class _PackedInconv(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, alpha: float, num_estimators: int, gamma: int
    ) -> None:
        super().__init__()
        self.conv = _PackedDoubleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            first=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class _PackedDown(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, alpha: float, num_estimators: int, gamma: int
    ) -> None:
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            _PackedDoubleConv(
                in_channels=in_channels,
                out_channels=out_channels,
                alpha=alpha,
                num_estimators=num_estimators,
                gamma=gamma,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mpconv(x)


class _PackedUp(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        alpha: float,
        num_estimators: int,
        gamma: int,
        bilinear: bool = True,
    ) -> None:
        super().__init__()
        self.num_estimators = num_estimators

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = _PackedDoubleConv(
                in_channels=in_channels,
                out_channels=out_channels,
                alpha=alpha,
                num_estimators=num_estimators,
                gamma=gamma,
                mid_channels=in_channels // 2,
            )
        else:
            self.up = PackedConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=2,
                stride=2,
                alpha=alpha,
                num_estimators=num_estimators,
                gamma=gamma,
            )
            self.conv = _PackedDoubleConv(
                in_channels=in_channels,
                out_channels=out_channels,
                alpha=alpha,
                num_estimators=num_estimators,
                gamma=gamma,
            )

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_y // 2, diff_x - diff_x // 2, diff_y - diff_y // 2])
        x1 = rearrange(x1, "b (m c) h w -> b m c h w", m=self.num_estimators)
        x2 = rearrange(x2, "b (m c) h w -> b m c h w", m=self.num_estimators)
        x = torch.cat([x2, x1], dim=2)
        x = rearrange(x, "b m c h w -> b (m c) h w")
        return self.conv(x)


class _PackedOutconv(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, alpha: float, num_estimators: int, gamma: int
    ) -> None:
        super().__init__()
        self.conv = PackedConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            last=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class _PackedUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_blocks: list[int],
        alpha: float = 1,
        num_estimators: int = 1,
        gamma: int = 1,
        bilinear: bool = False,
        dropout_rate: float = 0.0,
    ) -> None:
        check_unet_parameters(in_channels, num_classes, num_blocks, bilinear)
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.alpha = alpha
        self.num_estimators = num_estimators
        self.gamma = gamma
        self.bilinear = bilinear

        # Downsampling
        self.inc = _PackedInconv(in_channels, num_blocks[0], alpha, num_estimators, gamma)
        self.down1 = _PackedDown(num_blocks[0], num_blocks[1], alpha, num_estimators, gamma)
        self.down2 = _PackedDown(num_blocks[1], num_blocks[2], alpha, num_estimators, gamma)
        self.down3 = _PackedDown(num_blocks[2], num_blocks[3], alpha, num_estimators, gamma)
        factor = 2 if bilinear else 1
        self.down4 = _PackedDown(
            num_blocks[3], num_blocks[4] // factor, alpha, num_estimators, gamma
        )
        # Upsampling
        self.up1 = _PackedUp(
            num_blocks[4], num_blocks[3] // factor, alpha, num_estimators, gamma, bilinear
        )
        self.up2 = _PackedUp(
            num_blocks[3], num_blocks[2] // factor, alpha, num_estimators, gamma, bilinear
        )
        self.up3 = _PackedUp(
            num_blocks[2], num_blocks[1] // factor, alpha, num_estimators, gamma, bilinear
        )
        self.up4 = _PackedUp(num_blocks[1], num_blocks[0], alpha, num_estimators, gamma, bilinear)

        # Dropout
        self.dropout = nn.Dropout2d(dropout_rate)

        # Final output
        self.outc = _PackedOutconv(num_blocks[0], num_classes, alpha, num_estimators, gamma)

    def forward(self, x: Tensor) -> Tensor:
        # Downsampling
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.dropout(x2)
        x3 = self.down2(x2)
        x3 = self.dropout(x3)
        x4 = self.down3(x3)
        x4 = self.dropout(x4)
        x5 = self.down4(x4)
        # Upsampling
        x5 = self.dropout(x5)
        x = self.up1(x5, x4)
        x = self.dropout(x)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # Final output
        return self.outc(x)


def _packed_unet(
    in_channels: int,
    num_classes: int,
    num_blocks: list[int],
    bilinear: bool = False,
    alpha: float = 1,
    num_estimators: int = 1,
    gamma: int = 1,
    dropout_rate: float = 0.0,
) -> _PackedUNet:
    """_summary_.

    Args:
        in_channels (int): _description_
        num_classes (int): _description_
        num_blocks (list[int]): _description_
        bilinear (bool): _description_. Defaults to False.
        alpha (float): _description_. Defaults to 1.
        num_estimators (int): _description_. Defaults to 1.
        gamma (int): _description_. Defaults to 1.
        dropout_rate (float): Dropout rate for the model. Defaults to 0.0.

    Returns:
        PackedUNet: _description_
    """
    return _PackedUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        num_blocks=num_blocks,
        alpha=alpha,
        num_estimators=num_estimators,
        gamma=gamma,
        bilinear=bilinear,
        dropout_rate=dropout_rate,
    )


def packed_small_unet(
    in_channels: int,
    num_classes: int,
    bilinear: bool = False,
    alpha: float = 1,
    num_estimators: int = 1,
    gamma: int = 1,
    dropout_rate: float = 0.0,
) -> _PackedUNet:
    """Create a Packed-Ensembles of small U-Net models.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of classes to predict.
        bilinear (bool): If ``True``, use bilinear interpolation instead of
            transposed convolutions for upsampling. This can help to reduce the number
            of parameters and improve the performance of the model. Defaults to ``False``.
        alpha (float): Expansion factor affecting the width of the estimators. Defaults to ``1``.
        num_estimators (int): Number of estimators in the ensemble. Defaults to ``1``.
        gamma (int):  Number of groups within each estimator.. Defaults to ``1``.
        dropout_rate (float): Dropout rate for the model. Defaults to ``0.0``.

    Returns:
        PackedUNet: U-Net model.
    """
    return _packed_unet(
        in_channels=in_channels,
        num_classes=num_classes,
        num_blocks=[32, 64, 128, 256, 512],
        bilinear=bilinear,
        alpha=alpha,
        num_estimators=num_estimators,
        gamma=gamma,
        dropout_rate=dropout_rate,
    )


def packed_unet(
    in_channels: int,
    num_classes: int,
    bilinear: bool = False,
    alpha: float = 1,
    num_estimators: int = 1,
    gamma: int = 1,
    dropout_rate: float = 0.0,
) -> _PackedUNet:
    """Create a Packed-Ensembles of U-Net models.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        bilinear (bool): If ``True``, use bilinear interpolation instead
            of transposed convolutions for upsampling. This can help to reduce the number
            of parameters and improve the performance of the model. Defaults to ``False``.
        alpha (float): Expansion factor affecting the width of the estimators. Defaults to ``1``.
        num_estimators (int): Number of estimators in the ensemble. Defaults to ``1``.
        gamma (int): Number of groups within each estimator. Defaults to ``1``.
        dropout_rate (float): Dropout rate for the model. Defaults to ``0.0``.

    Returns:
        PackedUNet: U-Net model.
    """
    return _packed_unet(
        in_channels=in_channels,
        num_classes=num_classes,
        num_blocks=[64, 128, 256, 512, 1024],
        bilinear=bilinear,
        alpha=alpha,
        num_estimators=num_estimators,
        gamma=gamma,
        dropout_rate=dropout_rate,
    )
