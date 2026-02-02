# Code inspired by https://github.com/timeseriesAI/tsai/blob/main/tsai/models/InceptionTime.py

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from torch_uncertainty.layers import PackedConv1d, PackedLinear


class _PackedInceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_estimators: int,
        alpha: float,
        gamma: int,
        bottleneck: bool = True,
        first: bool = False,
    ):
        super().__init__()
        kernel_sizes = [kernel_size // (2**i) for i in range(3)]
        kernel_sizes = [k if k % 2 != 0 else k - 1 for k in kernel_sizes]  # ensure odd kernel sizes

        bottleneck = bottleneck if in_channels > out_channels else False

        self.bottleneck = (
            PackedConv1d(
                in_channels,
                out_channels,
                1,
                alpha=alpha,
                num_estimators=num_estimators,
                gamma=gamma,
                padding="same",
                bias=False,
                first=first,
            )
            if bottleneck
            else None
        )
        self.convs = nn.ModuleList(
            [
                PackedConv1d(
                    in_channels=out_channels if bottleneck else in_channels,
                    out_channels=out_channels,
                    kernel_size=k,
                    alpha=alpha,
                    num_estimators=num_estimators,
                    gamma=gamma,
                    padding="same",
                    bias=False,
                    first=first and not bottleneck,
                )
                for k in kernel_sizes
            ]
        )
        self.maxconvpool = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            PackedConv1d(
                in_channels,
                out_channels,
                1,
                alpha=alpha,
                num_estimators=num_estimators,
                gamma=gamma,
                padding="same",
                bias=False,
                first=first,
            ),
        )
        self.batch_norm = nn.BatchNorm1d(int((out_channels * 4) * alpha))

    def forward(self, x: Tensor) -> Tensor:
        out = self.bottleneck(x) if self.bottleneck is not None else x
        out = torch.cat(
            [conv(out) for conv in self.convs] + [self.maxconvpool(x)],
            dim=1,
        )
        return F.relu(self.batch_norm(out))


class _PackedInceptionTime(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_estimators: int,
        alpha: float,
        gamma: int = 1,
        kernel_size: int = 40,
        embed_dim: int = 32,
        num_blocks: int = 6,
        dropout: float = 0.0,
        residual: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.residual = residual

        self.layers = nn.ModuleList()
        self.shortcut = nn.ModuleList() if residual else None

        for i in range(num_blocks):
            self.layers.append(
                nn.Sequential(
                    _PackedInceptionBlock(
                        in_channels=in_channels if i == 0 else embed_dim * 4,
                        out_channels=embed_dim,
                        kernel_size=kernel_size,
                        num_estimators=num_estimators,
                        alpha=alpha,
                        gamma=gamma,
                        first=(i == 0),
                    ),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                )
            )
            if self.shortcut is not None and i % 3 == 2:
                n_in = in_channels if i == 2 else embed_dim * 4
                n_out = embed_dim * 4
                self.shortcut.append(
                    nn.BatchNorm1d(int(n_out * alpha))
                    if n_in == n_out
                    else nn.Sequential(
                        PackedConv1d(
                            n_in,
                            n_out,
                            1,
                            alpha=alpha,
                            num_estimators=num_estimators,
                            gamma=gamma,
                            bias=False,
                            first=i == 2,
                        ),
                        nn.BatchNorm1d(int(n_out * alpha)),
                    )
                )

        self.adaptive_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.last_layer = PackedLinear(
            embed_dim * 4,
            num_classes,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            last=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_channels, seq_len).

        Returns:
            Tensor: Output tensor of shape (batch_size, num_classes).
        """
        res = x
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.shortcut is not None and i % 3 == 2:
                shortcut = self.shortcut[i // 3](res)
                x = F.relu(x + shortcut)
                res = x

        x = self.adaptive_avg_pool(x)
        x = x.flatten(1)
        return self.last_layer(x)


def packed_inception_time(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    alpha: float,
    gamma: int = 1,
    kernel_size: int = 40,
    embed_dim: int = 32,
    num_blocks: int = 6,
    dropout: float = 0.0,
    residual: bool = True,
) -> _PackedInceptionTime:
    """Packed-Ensembles of InceptionTime.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        num_estimators (int): Number of estimators for Packed-Ensembles.
        alpha (float): Alpha parameter for Packed-Ensembles.
        gamma (int): Gamma parameter for Packed-Ensembles. Default is ``1``.
        kernel_size (int): Size of the convolutional kernels. Default is ``40``.
        embed_dim (int): Dimension of the embedding. Default is ``32``.
        num_blocks (int): Number of inception blocks. Default is ``6``.
        dropout (float): Dropout rate. Default is ``0.0``.
        residual (bool): Whether to use residual connections. Default is ``True``.

    Returns:
        _InceptionTime: An instance of the InceptionTime model.
    """
    return _PackedInceptionTime(
        in_channels,
        num_classes,
        num_estimators,
        alpha,
        gamma,
        kernel_size,
        embed_dim,
        num_blocks,
        dropout,
        residual,
    )
