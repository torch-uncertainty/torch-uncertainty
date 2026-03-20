"""Modified from https://github.com/nikitadurasov/masksembles/."""

from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.common_types import _size_2_t


def _generate_masks(m: int, n: int, s: float) -> np.ndarray:
    """Generates set of binary masks with properties defined by n, m, s params.
    Results of this function are stochastic, that is, calls with the same sets
    of arguments might generate outputs of different shapes. Check
    generate_masks and generation_wrapper function for more deterministic
    behaviour.

    Args:
        m (int): Number of ones in each mask.
        n (int): Number of masks in the set.
        s (float): Scale param controls overlap of generated masks.

    Returns:
        np.ndarray: Matrix of binary vectors.
    """
    rng = np.random.default_rng()
    total_positions = int(m * s)
    masks = []

    for _ in range(n):
        new_vector = np.zeros([total_positions])
        idx = rng.choice(range(total_positions), m, replace=False)
        new_vector[idx] = 1
        masks.append(new_vector)

    masks = np.array(masks)
    # drop useless positions
    return masks[:, ~np.all(masks == 0, axis=0)]


def generate_masks(m: int, n: int, s: float) -> np.ndarray:
    """Generates set of binary masks with properties defined by n, m, s params
    Resulting masks are required to have fixed features size.
    Since process of masks generation is stochastic therefore function
    evaluates _generate_masks multiple times till expected size is acquired.

    Args:
        m (int): number of ones in each mask
        n (int): number of masks in the set
        s (float): scale param controls overlap of generated masks

    Returns:
        np.ndarray: matrix of binary vectors
    """
    masks = _generate_masks(m, n, s)
    # hardcoded formula for expected size, check reference
    expected_size = int(m * s * (1 - (1 - 1 / s) ** n))
    while masks.shape[1] != expected_size:
        masks = _generate_masks(m, n, s)
    return masks


def generation_wrapper(c: int, n: int, scale: float) -> np.ndarray:
    """Generates set of binary masks with properties defined by c, n, scale
    params. Allows to generate masks sets with predefined features number c.
    Particularly convenient to use in torch-like layers where one need to
    define shapes inputs tensors beforehand.

    Args:
        c (int): number of channels in generated masks.
        n (int): number of masks in the set.
        scale (float): scale param controls overlap of generated masks.

    Raises:
        ValueError: If :attr:`c` < 10.
        ValueError: If :attr:`s` > 0.6.

    Returns:
        np.ndarray: matrix of binary vectors
    """
    if c < 10:
        raise ValueError(
            "Masksembles approach couldn't be used in such setups where "
            "number of channels is less then 10. Current value is "
            f"(channels={c})."
            "Please increase number of features in your layer or remove this "
            "particular instance of Masksembles from your architecture."
        )

    if scale > 6.0:
        raise ValueError(
            "Masksembles approach couldn't be used in such setups where "
            "scale parameter is larger then 6. Current value is  "
            f"(scale={scale})."
        )

    # inverse formula for number of active features in masks
    active_features = int(int(c) / (scale * (1 - (1 - 1 / scale) ** n)))

    # Use binary search to find the correct value of the scale
    masks = generate_masks(active_features, n, scale)
    up = 4 * scale
    down = max(0.2 * scale, 1.0)
    s = (down + up) / 2
    im_s = -1
    while im_s != c:
        masks = generate_masks(active_features, n, s)
        im_s = masks.shape[-1]
        if im_s < c:
            down = s
            s = (down + up) / 2
        elif im_s > c:
            up = s
            s = (down + up) / 2

    return masks


class Mask1d(nn.Module):
    def __init__(self, channels: int, num_masks: int, scale: float, **factory_kwargs) -> None:
        super().__init__()
        self.num_masks = num_masks

        masks = generation_wrapper(channels, num_masks, scale)
        masks = torch.from_numpy(masks)
        self.masks = torch.nn.Parameter(masks, requires_grad=False).to(
            device=factory_kwargs["device"]
        )

    def forward(self, inputs: Tensor) -> Tensor:
        batch = inputs.shape[0]
        x = torch.split(inputs.unsqueeze(1), batch // self.num_masks, dim=0)
        x = torch.cat(x, dim=1).permute([1, 0, 2])
        x = x * self.masks.unsqueeze(1)
        x = torch.cat(torch.split(x, 1, dim=0), dim=1)
        return torch.as_tensor(x, dtype=inputs.dtype).squeeze(0)


class Mask2d(nn.Module):
    def __init__(self, channels: int, num_masks: int, scale: float, **factory_kwargs) -> None:
        super().__init__()
        self.num_masks = num_masks

        masks = generation_wrapper(channels, num_masks, scale)
        masks = torch.from_numpy(masks)
        self.masks = torch.nn.Parameter(masks, requires_grad=False).to(
            device=factory_kwargs["device"]
        )

    def forward(self, inputs: Tensor) -> Tensor:
        batch = inputs.shape[0]
        x = torch.split(inputs.unsqueeze(1), batch // self.num_masks, dim=0)
        x = torch.cat(x, dim=1).permute([1, 0, 2, 3, 4])
        x = x * self.masks.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        x = torch.cat(torch.split(x, 1, dim=0), dim=1)
        return torch.as_tensor(x, dtype=inputs.dtype).squeeze(0)


class MaskedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_estimators: int,
        scale: float,
        bias: bool = True,
        device: Any | None = None,
        dtype: Any | None = None,
    ) -> None:
        r"""Masksembles-style Linear layer.

        This layer computes fully-connected operation for a given number of
        estimators (:attr:`num_estimators`) with a given :attr:`scale`.

        Args:
            in_features (int): Number of input features of the linear layer.
            out_features (int): Number of channels produced by the linear layer.
            num_estimators (int): The number of estimators grouped in the layer.
            scale (float): The scale parameter for the masks.
            bias (bool, optional): It ``True``, adds a learnable bias to the output. Defaults to ``True``.
            groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to ``1``.
            device (Any, optional): The desired device of returned tensor. Defaults to ``None``.
            dtype (Any, optional): The desired data type of returned tensor. Defaults to ``None``.

        Warning:
            Be sure to apply a repeat on the batch at the start of the training
            if you use `MaskedLinear`.

        References:
            [1] `Masksembles for Uncertainty Estimation, Nikita Durasov, Timur Bagautdinov, Pierre Baque, Pascal Fua
            <https://arxiv.org/abs/2012.08334>`_.

        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if scale is None:
            raise ValueError("You must specify the value of the arg. `scale`")
        if scale < 1:
            raise ValueError(f"Attribute `scale` should be >= 1, not {scale}.")

        self.mask = Mask1d(in_features, num_masks=num_estimators, scale=scale, **factory_kwargs)
        self.linear = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            **factory_kwargs,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.linear(self.mask(inputs))


class MaskedConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        num_estimators: int,
        scale: float,
        stride: _size_2_t = 1,
        padding: str | _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        device: Any | None = None,
        dtype: Any | None = None,
    ) -> None:
        r"""Masksembles-style Conv2d layer.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int or tuple): Size of the convolving kernel.
            num_estimators (int): Number of estimators in the ensemble.
            scale (float): The scale parameter for the masks.
            stride (int or tuple, optional): Stride of the convolution. Defaults to ``1``.
            padding (int, tuple or str, optional): Padding added to all four sides of the input. Defaults to ``0``.
            dilation (int or tuple, optional): Spacing between kernel elements. Defaults to ``1``.
            groups (int, optional): Number of blocked connexions from input channels to output channels for each estimator. Defaults to ``1``.
            bias (bool, optional): If ``True``, adds a learnable bias to the output. Defaults to ``True``.
            device (Any, optional): The desired device of returned tensor. Defaults to ``None``.
            dtype (Any, optional): The desired data type of returned tensor. Defaults to ``None``.

        Warning:
            Be sure to apply a repeat on the batch at the start of the training
            if you use `MaskedConv2d`.

        References:
            [1] `Masksembles for Uncertainty Estimation, Nikita Durasov, Timur Bagautdinov, Pierre Baque, Pascal Fua
            <https://arxiv.org/abs/2012.08334>`_.


        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if scale is None:
            raise ValueError("You must specify the value of the arg. `scale`")
        if scale < 1:
            raise ValueError(f"Attribute `scale` should be >= 1, not {scale}.")

        self.mask = Mask2d(in_channels, num_masks=num_estimators, scale=scale, **factory_kwargs)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode="zeros",
            **factory_kwargs,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(self.mask(inputs))


class MaskedConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        num_estimators: int,
        scale: float,
        stride: _size_2_t = 1,
        padding: str | _size_2_t = 0,
        output_padding: str | _size_2_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_2_t = 1,
        padding_mode: Literal["circular", "reflect", "replicate", "zeros"] = "zeros",
        device: Any | None = None,
        dtype: Any | None = None,
    ) -> None:
        r"""Masksembles-style ConvTranspose2d layer.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int or tuple): Size of the convolving kernel.
            num_estimators (int): Number of estimators in the ensemble.
            scale (float): The scale parameter for the masks.
            stride (int or tuple, optional): Stride of the convolution. Defaults to ``1``.
            padding (int, tuple or str, optional): Padding added to all four sides of the input. Defaults to ``0``.
            output_padding (int, tuple or str, optional): Additional size added to one side of each dimension in the output shape. Defaults to ``0``.
            groups (int, optional): Number of blocked connexions from input channels to output channels for each estimator. Defaults to ``1``.
            bias (bool, optional): If ``True``, adds a learnable bias to the output. Defaults to ``True``.
            dilation (int or tuple, optional): Spacing between kernel elements. Defaults to ``1``.
            padding_mode (str, optional): _description_. Defaults to ``'zeros'``.
            device (Any, optional): The desired device of returned tensor. Defaults to ``None``.
            dtype (Any, optional): The desired data type of returned tensor. Defaults to ``None``.

        Warning:
            Be sure to apply a repeat on the batch at the start of the training
            if you use `MaskedConvTranspose2d`.

        References:
            [1] `Masksembles for Uncertainty Estimation, Nikita Durasov, Timur Bagautdinov, Pierre Baque, Pascal Fua
            <https://arxiv.org/abs/2012.08334>`_
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if scale is None:
            raise ValueError("You must specify the value of the arg. `scale`")
        if scale < 1:
            raise ValueError(f"Attribute `scale` should be >= 1, not {scale}.")
        self.mask = Mask2d(in_channels, num_masks=num_estimators, scale=scale, **factory_kwargs)
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
            **factory_kwargs,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(self.mask(inputs))
