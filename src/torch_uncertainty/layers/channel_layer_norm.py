import torch
from torch import Tensor
from torch.nn import LayerNorm

from .utils import ChannelBack, ChannelFront


class ChannelLayerNorm(LayerNorm):
    def __init__(
        self,
        normalized_shape: int | list[int],
        eps: float = 0.00001,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ) -> None:
        """Channel-wise layer-norm."""
        super().__init__(normalized_shape, eps, elementwise_affine, bias, device, dtype)
        self.cback = ChannelBack()
        self.cfront = ChannelFront()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.cfront(super().forward(self.cback(inputs)))
