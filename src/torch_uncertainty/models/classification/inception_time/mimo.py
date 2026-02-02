from einops import rearrange
from torch import Tensor

from .std import _InceptionTime


class _MIMOInceptionTime(_InceptionTime):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_estimators: int,
        kernel_size: int = 40,
        embed_dim: int = 32,
        num_blocks: int = 6,
        dropout: float = 0.0,
        residual: bool = True,
    ):
        super().__init__(
            in_channels=in_channels * num_estimators,
            num_classes=num_classes * num_estimators,
            kernel_size=kernel_size,
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            dropout=dropout,
            residual=residual,
        )
        self.num_estimators = num_estimators

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            x = x.repeat(self.num_estimators, 1, 1)
        out = rearrange(x, "(m b) c t -> b (m c) t", m=self.num_estimators)
        out = super().forward(out)
        return rearrange(out, "b (m d) -> (m b) d", m=self.num_estimators)


def mimo_inception_time(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    kernel_size: int = 40,
    embed_dim: int = 32,
    num_blocks: int = 6,
    dropout: float = 0.0,
    residual: bool = True,
) -> _MIMOInceptionTime:
    """MIMO of InceptionTime.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        num_estimators (int): Number of estimators for MIMO.
        kernel_size (int): Size of the convolutional kernel. Default is ``40``.
        embed_dim (int): Dimension of the embedding. Default is ``32``.
        num_blocks (int): Number of inception blocks. Default is ``6``.
        dropout (float): Dropout rate. Default is ``0.0``.
        residual (bool): Whether to use residual connections. Default is ``True``.

    Returns:
        _MIMOInceptionTime: The MIMO InceptionTime model.
    """
    return _MIMOInceptionTime(
        in_channels=in_channels,
        num_classes=num_classes,
        num_estimators=num_estimators,
        kernel_size=kernel_size,
        embed_dim=embed_dim,
        num_blocks=num_blocks,
        dropout=dropout,
        residual=residual,
    )
