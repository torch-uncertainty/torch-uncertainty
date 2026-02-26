import math
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t

from .functional.packed import packed_linear, packed_multi_head_attention_forward


def check_packed_parameters_consistency(alpha: float, gamma: int, num_estimators: int) -> None:
    """Check the consistency of the parameters of the Packed-Ensembles layers.

    Args:
        alpha (float): The width multiplier of the layer.
        gamma (int): The number of groups in the ensemble.
        num_estimators (int): The number of estimators in the ensemble.
    """
    if alpha is None:
        raise ValueError("You must specify the value of the arg. `alpha`")

    if alpha <= 0:
        raise ValueError(f"Attribute `alpha` should be > 0, not {alpha}")

    if not isinstance(gamma, int):
        raise TypeError(f"Attribute `gamma` should be an int, not {type(gamma)}")
    if gamma <= 0:
        raise ValueError(f"Attribute `gamma` should be >= 1, not {gamma}")

    if num_estimators is None:
        raise ValueError("You must specify the value of the arg. `num_estimators`")
    if not isinstance(num_estimators, int):
        raise TypeError(f"Attribute `num_estimators` should be an int, not {type(num_estimators)}")
    if num_estimators <= 0:
        raise ValueError(f"Attribute `num_estimators` should be >= 1, not {num_estimators}")


class PackedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        alpha: float,
        num_estimators: int,
        gamma: int = 1,
        bias: bool = True,
        first: bool = False,
        last: bool = False,
        implementation: str = "einsum",
        device=None,
        dtype=None,
    ) -> None:
        r"""Packed-Ensembles-style Linear layer.

        This layer computes fully-connected operation for a given number of
        estimators (:attr:`num_estimators`).

        Args:
            in_features (int): Number of input features of the linear layer.
            out_features (int): Number of channels produced by the linear layer.
            alpha (float): The width multiplier of the linear layer.
            num_estimators (int): The number of estimators grouped in the layer.
            gamma (int, optional): Defaults to ``1``.
            bias (bool, optional): If ``True``, adds a learnable bias to the
                output. Defaults to ``True``.
            first (bool, optional): Whether this is the first layer of the
                network. Defaults to ``False``.
            last (bool, optional): Whether this is the last layer of the network.
                Defaults to ``False``.
            implementation (str, optional): The implementation to use. Available implementations:

                - ``"conv1d"``: The conv1d implementation of the linear layer.
                - ``"sparse"``: The sparse implementation of the linear layer.
                - ``"full"``: The full implementation of the linear layer.
                - ``"einsum"`` (default): The einsum implementation of the linear layer.
            device (torch.device, optional): The device to use for the layer's
                parameters. Defaults to ``None``.
            dtype (torch.dtype, optional): The dtype to use for the layer's
                parameters. Defaults to ``None``.

        Shape:
            - Input:
                - If :attr:`first` is ``True``: :math:`(B, \ast, H_{\text{in}})` where
                  :math:`B` is the batch size, :math:`\ast` means any number of
                  additional dimensions and :math:`H_{\text{in}}=\text{in\_features}`.
                - Otherwise: :math:`(B, \ast, H_{\text{in}} \times \alpha)`
            - Output:
                - If :attr:`last` is ``True``: :math:`(B, \ast, H_{\text{out}}\times M)` where
                  :math:`H_{\text{out}}=\text{out\_features}` and :math:`M=\text{num\_estimators}`.
                - Otherwise: :math:`(B, \ast, H_{\text{out}} \times \alpha)`

        Explanation Note:
            Increasing :attr:`alpha` will increase the number of channels of the
            ensemble, increasing its representation capacity. Increasing
            :attr:`gamma` will increase the number of groups in the network and
            therefore reduce the number of parameters.

        Note:
            Each ensemble member will only see
            :math:`\frac{\text{in_features}}{\text{num_estimators}}` features,
            so when using :attr:`gamma` you should make sure that
            :attr:`in_features` and :attr:`out_features` are both divisible by
            :attr:`num_estimators` :math:`\times`:attr:`gamma`. However, the
            number of input and output features will be changed to comply with
            this constraint.
        """
        check_packed_parameters_consistency(alpha, gamma, num_estimators)

        if implementation not in ["sparse", "full", "einsum", "conv1d"]:
            raise ValueError(
                f"Unknown implementation: {implementation} for PackedLinear. "
                "Available implementations are: 'sparse', 'full', 'einsum', 'conv1d'"
            )

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.first = first
        self.last = last
        self.num_estimators = num_estimators
        self.rearrange = rearrange
        self.implementation = implementation

        # Define the number of features of the underlying convolution
        extended_in_features = int(in_features * (1 if first else alpha))
        extended_out_features = int(out_features * (num_estimators if last else alpha))

        # Define the number of groups of the underlying convolution
        actual_groups = num_estimators * gamma if not first else 1

        if extended_in_features % actual_groups:
            raise ValueError(
                f"PackedLinear: `in_features` ({in_features}) is not divisible by "
                f"`actual_groups` ({actual_groups}). Please make `in_features` divisible "
                f"by `num_estimators * gamma` ({num_estimators * gamma})."
            )
        if extended_out_features % (num_estimators * gamma):
            raise ValueError(
                f"PackedLinear: `out_features` ({out_features}) is not divisible by "
                f"`num_estimators * gamma` ({num_estimators * gamma}). Please make "
                f"`out_features` divisible by `num_estimators * gamma` ({num_estimators * gamma})."
            )

        self.weight = nn.Parameter(
            torch.empty(
                (
                    actual_groups,
                    extended_out_features // actual_groups,
                    extended_in_features // actual_groups,
                ),
                **factory_kwargs,
            )
        )

        self.in_features = extended_in_features // actual_groups
        self.out_features = extended_out_features // actual_groups
        self.groups = actual_groups

        if bias:
            self.bias = nn.Parameter(torch.empty(extended_out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for n in range(self.groups):
            nn.init.kaiming_uniform_(self.weight[n], a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        if self.implementation == "sparse":
            self.weight = nn.Parameter(torch.block_diag(*self.weight).to_sparse())

    def forward(self, inputs: Tensor) -> Tensor:
        out = packed_linear(
            inputs=inputs,
            weight=self.weight,
            num_groups=self.groups,
            implementation=self.implementation,
            bias=self.bias,
        )
        return (
            out
            if not self.last
            else rearrange(out, "b ... (m h) -> (m b) ... h", m=self.num_estimators)
        )


class PackedConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        alpha: int,
        num_estimators: int,
        gamma: int = 1,
        stride: _size_1_t = 1,
        padding: str | _size_1_t = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        minimum_channels_per_group: int = 64,
        bias: bool = True,
        padding_mode: str = "zeros",
        first: bool = False,
        last: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        r"""Packed-Ensembles-style Conv1d layer.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int or tuple): Size of the convolving kernel.
            alpha (int): The channel multiplier of the convolutional layer.
            num_estimators (int): Number of estimators in the ensemble.
            gamma (int, optional): Defaults to ``1``.
            stride (int or tuple, optional): Stride of the convolution. Defaults to ``1``.
            padding (int, tuple or str, optional): Padding added to both sides of the input. Defaults to ``0``.
            dilation (int or tuple, optional): Spacing between kernel elements. Defaults to ``1``.
            groups (int, optional): Number of blocked connexions from input
            channels to output channels for each estimator. Defaults to ``1``.
            minimum_channels_per_group (int, optional): Smallest possible number of channels per group.
            bias (bool, optional): If ``True``, adds a learnable bias to the output. Defaults to ``True``.
            padding_mode (str, optional): ``'zeros'``, ``'reflect'``,``'replicate'`` or ``'circular'``. Defaults to ``'zeros'``.
            first (bool, optional): Whether this is the first layer of the network. Defaults to ``False``.
            last (bool, optional): Whether this is the last layer of the network. Defaults to ``False``.
            device (torch.device, optional): The device to use for the layer's
            parameters. Defaults to ``None``.
            dtype (torch.dtype, optional): The dtype to use for the layer's
            parameters. Defaults to ``None``.

        Shape:
            - Input:
                - If :attr:`first` is ``True``: :math:`(B, C_{\text{in}}, L_{\text{in}})` where
                  :math:`B` is the batch size, :math:`C_{\text{in}}=\text{in\_channels}`, and
                  :math:`L_{\text{in}}` is the length of the signal sequence.
                - Otherwise: :math:`(B, C_{\text{in}} \times \alpha, L_{\text{in}})`
            - Output:
                - If :attr:`last` is ``True``: :math:`(B, C_{\text{out}}\times M, L_{\text{out}})`
                  where :math:`C_{\text{out}}=\text{out\_channels}` and :math:`M=\text{num\_estimators}`.
                - Otherwise: :math:`(B, C_{\text{out}} \times \alpha, L_{\text{out}})`

        Explanation Note:
            Increasing :attr:`alpha` will increase the number of channels of the
            ensemble, increasing its representation capacity. Increasing
            :attr:`gamma` will increase the number of groups in the network and
            therefore reduce the number of parameters.

        Note:
            Each ensemble member will only see
            :math:`\frac{\text{in_channels}}{\text{num_estimators}}` channels,
            so when using :attr:`groups` you should make sure that
            :attr:`in_channels` and :attr:`out_channels` are both divisible by
            :attr:`num_estimators` :math:`\times`:attr:`gamma` :math:`\times`
            :attr:`groups`. However, the number of input and output channels will
            be changed to comply with this constraint.
        """
        check_packed_parameters_consistency(alpha, gamma, num_estimators)
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.first = first
        self.last = last
        self.num_estimators = num_estimators

        # Define the number of channels of the underlying convolution
        extended_in_channels = int(in_channels * (1 if first else alpha))
        extended_out_channels = int(out_channels * (num_estimators if last else alpha))

        # Define the number of groups of the underlying convolution
        actual_groups = 1 if first else gamma * groups * num_estimators

        while (
            extended_in_channels % actual_groups != 0
            or extended_in_channels // actual_groups < minimum_channels_per_group
        ) and actual_groups // (groups * num_estimators) > 1:
            gamma -= 1
            actual_groups = gamma * groups * num_estimators

        if extended_in_channels % actual_groups:
            raise ValueError(
                f"PackedConv1d: `in_channels` ({in_channels}) is not divisible by "
                f"`actual_groups` ({actual_groups}). Please make `in_channels` divisible "
                f"by `gamma * groups * num_estimators` ({gamma * groups * num_estimators})."
            )
        if extended_out_channels % (gamma * groups * num_estimators):
            raise ValueError(
                f"PackedConv1d: `out_channels` ({out_channels}) is not divisible by "
                f"`gamma * groups * num_estimators` ({gamma * groups * num_estimators}). "
                f"Please make `out_channels` divisible by `gamma * groups * num_estimators` "
                f"({gamma * groups * num_estimators})."
            )

        self.conv = nn.Conv1d(
            in_channels=extended_in_channels,
            out_channels=extended_out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=actual_groups,
            bias=bias,
            padding_mode=padding_mode,
            **factory_kwargs,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        out = self.conv(inputs)
        return (
            out
            if not self.last
            else rearrange(out, "b (m c) ... -> (m b) c ...", m=self.num_estimators)
        )

    @property
    def weight(self) -> Tensor:
        r"""The weight of the underlying convolutional layer."""
        return self.conv.weight

    @property
    def bias(self) -> Tensor | None:
        r"""The bias of the underlying convolutional layer."""
        return self.conv.bias


class PackedConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        alpha: int,
        num_estimators: int,
        gamma: int = 1,
        stride: _size_2_t = 1,
        padding: str | _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        minimum_channels_per_group: int = 64,
        bias: bool = True,
        padding_mode: str = "zeros",
        first: bool = False,
        last: bool = False,
        device: Any | None = None,
        dtype: Any | None = None,
    ) -> None:
        r"""Packed-Ensembles-style Conv2d layer.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int or tuple): Size of the convolving kernel.
            alpha (int): The channel multiplier of the convolutional layer.
            num_estimators (int): Number of estimators in the ensemble.
            gamma (int, optional): Defaults to ``1``.
            stride (int or tuple, optional): Stride of the convolution. Defaults to ``1``.
            padding (int, tuple or str, optional): Padding added to all four sides of the input. Defaults to ``0``.
            dilation (int or tuple, optional): Spacing between kernel elements. Defaults to ``1``.
            groups (int, optional): Number of blocked connexions from input channels to output channels for each
                estimator. Defaults to ``1``.
            minimum_channels_per_group (int, optional): Smallest possible number of channels per group.
            bias (bool, optional): If ``True``, adds a learnable bias to the output. Defaults to ``True``.
            padding_mode (str, optional): ``'zeros'``, ``'reflect'``,``'replicate'`` or ``'circular'``. Defaults
                to ``'zeros'``.
            first (bool, optional): Whether this is the first layer of the network. Defaults to ``False``.
            last (bool, optional): Whether this is the last layer of the network. Defaults to ``False``.
            device (torch.device, optional): The device to use for the layer's parameters. Defaults to ``None``.
            dtype (torch.dtype, optional): The dtype to use for the layer's parameters. Defaults to ``None``.

        Shape:
            - Input:
                - If :attr:`first` is ``True``: :math:`(B, C_{\text{in}}, H_{\text{in}}, W_{\text{in}})` where
                  :math:`B` is the batch size, :math:`C_{\text{in}}=\text{in\_channels}`,
                  :math:`H_{\text{in}}` and :math:`W_{\text{in}}` are the height and width of the input image.
                - Otherwise: :math:`(B, C_{\text{in}} \times \alpha, H_{\text{in}}, W_{\text{in}})`
            - Output:
                - If :attr:`last` is ``True``: :math:`(B, C_{\text{out}}\times M, H_{\text{out}}, W_{\text{out}})`
                  where :math:`C_{\text{out}}=\text{out\_channels}` and :math:`M=\text{num\_estimators}`.
                - Otherwise: :math:`(B, C_{\text{out}} \times \alpha, H_{\text{out}}, W_{\text{out}})`

        Explanation Note:
            Increasing :attr:`alpha` will increase the number of channels of the
            ensemble, increasing its representation capacity. Increasing
            :attr:`gamma` will increase the number of groups in the network and
            therefore reduce the number of parameters.

        Note:
            Each ensemble member will only see
            :math:`\frac{\text{in_channels}}{\text{num_estimators}}` channels,
            so when using :attr:`groups` you should make sure that
            :attr:`in_channels` and :attr:`out_channels` are both divisible by
            :attr:`num_estimators` :math:`\times`:attr:`gamma` :math:`\times`
            :attr:`groups`. However, the number of input and output channels will
            be changed to comply with this constraint.
        """
        check_packed_parameters_consistency(alpha, gamma, num_estimators)
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.first = first
        self.last = last
        self.num_estimators = num_estimators

        # Define the number of channels of the underlying convolution
        extended_in_channels = int(in_channels * (1 if first else alpha))
        extended_out_channels = int(out_channels * (num_estimators if last else alpha))

        # Define the number of groups of the underlying convolution
        actual_groups = 1 if first else gamma * groups * num_estimators

        while (
            extended_in_channels % actual_groups != 0
            or extended_in_channels // actual_groups < minimum_channels_per_group
        ) and actual_groups // (groups * num_estimators) > 1:
            gamma -= 1
            actual_groups = gamma * groups * num_estimators

        if extended_in_channels % actual_groups:
            raise ValueError(
                f"PackedConv2d: `in_channels` ({in_channels}) is not divisible by "
                f"`actual_groups` ({actual_groups}). Please make `in_channels` divisible "
                f"by `gamma * groups * num_estimators` ({gamma * groups * num_estimators})."
            )
        if extended_out_channels % (gamma * groups * num_estimators):
            raise ValueError(
                f"PackedConv2d: `out_channels` ({out_channels}) is not divisible by "
                f"`gamma * groups * num_estimators` ({gamma * groups * num_estimators}). "
                f"Please make `out_channels` divisible by `gamma * groups * num_estimators` "
                f"({gamma * groups * num_estimators})."
            )

        self.conv = nn.Conv2d(
            in_channels=extended_in_channels,
            out_channels=extended_out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=actual_groups,
            bias=bias,
            padding_mode=padding_mode,
            **factory_kwargs,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        out = self.conv(inputs)
        return (
            out
            if not self.last
            else rearrange(out, "b (m c) ... -> (m b) c ...", m=self.num_estimators)
        )

    @property
    def weight(self) -> Tensor:
        r"""The weight of the underlying convolutional layer."""
        return self.conv.weight

    @property
    def bias(self) -> Tensor | None:
        r"""The bias of the underlying convolutional layer."""
        return self.conv.bias


class PackedConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        alpha: int,
        num_estimators: int,
        gamma: int = 1,
        stride: _size_3_t = 1,
        padding: str | _size_3_t = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        minimum_channels_per_group: int = 64,
        bias: bool = True,
        padding_mode: str = "zeros",
        first: bool = False,
        last: bool = False,
        device: Any | None = None,
        dtype: Any | None = None,
    ) -> None:
        r"""Packed-Ensembles-style Conv3d layer.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int or tuple): Size of the convolving kernel.
            alpha (int): The channel multiplier of the convolutional layer.
            num_estimators (int): Number of estimators in the ensemble.
            gamma (int, optional): Defaults to ``1``.
            stride (int or tuple, optional): Stride of the convolution. Defaults to ``1``.
            padding (int, tuple or str, optional): Padding added to all six sides of the input. Defaults to ``0``.
            dilation (int or tuple, optional): Spacing between kernel elements. Defaults to ``1``.
            groups (int, optional): Number of blocked connexions from input
            channels to output channels for each estimator. Defaults to ``1``.
            minimum_channels_per_group (int, optional): Smallest possible number of channels per group.
            bias (bool, optional): If ``True``, adds a learnable bias to the output. Defaults to ``True``.
            padding_mode (str, optional): ``'zeros'``, ``'reflect'``,``'replicate'`` or ``'circular'``. Defaults to ``'zeros'``.
            first (bool, optional): Whether this is the first layer of the network. Defaults to ``False``.
            last (bool, optional): Whether this is the last layer of the network. Defaults to ``False``.
            device (torch.device, optional): The device to use for the layer's parameters. Defaults to ``None``.
            dtype (torch.dtype, optional): The dtype to use for the layer's parameters. Defaults to ``None``.

        Shape:
            - Input:
                - If :attr:`first` is ``True``: :math:`(B, C_{\text{in}}, D_{\text{in}}, H__{\text{in}}, W__{\text{in}})`
                  where :math:`B` is the batch size, :math:`C_{\text{in}}=\text{in\_channels}`,
                  :math:`D_{\text{in}}` is the depth of the input, :math:`H_{\text{in}}`
                  and :math:`W_{\text{in}}` are height and width of the input planes.
                - Otherwise: :math:`(B, C_{\text{in}} \times \alpha, D__{\text{in}}, H_{\text{in}}, W_{\text{in}})`
            - Output:
                - If :attr:`last` is ``True``: :math:`(B, C_{\text{out}}\times M, D_{\text{out}}, H__{\text{out}}, W__{\text{out}})` where
                  :math:`C_{\text{out}}=\text{out\_channels}` and :math:`M=\text{num\_estimators}`.
                - Otherwise: :math:`(B, C_{\text{out}} \times \alpha, D_{\text{out}}, H__{\text{out}}, W__{\text{out}})`

        Explanation Note:
            Increasing :attr:`alpha` will increase the number of channels of the
            ensemble, increasing its representation capacity. Increasing
            :attr:`gamma` will increase the number of groups in the network and
            therefore reduce the number of parameters.

        Note:
            Each ensemble member will only see
            :math:`\frac{\text{in_channels}}{\text{num_estimators}}` channels,
            so when using :attr:`groups` you should make sure that
            :attr:`in_channels` and :attr:`out_channels` are both divisible by
            :attr:`num_estimators` :math:`\times`:attr:`gamma` :math:`\times`
            :attr:`groups`. However, the number of input and output channels will
            be changed to comply with this constraint.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        check_packed_parameters_consistency(alpha, gamma, num_estimators)

        self.first = first
        self.last = last
        self.num_estimators = num_estimators

        # Define the number of channels of the underlying convolution
        extended_in_channels = int(in_channels * (1 if first else alpha))
        extended_out_channels = int(out_channels * (num_estimators if last else alpha))

        # Define the number of groups of the underlying convolution
        actual_groups = 1 if first else gamma * groups * num_estimators

        while (
            extended_in_channels % actual_groups != 0
            or extended_in_channels // actual_groups < minimum_channels_per_group
        ) and actual_groups // (groups * num_estimators) > 1:
            gamma -= 1
            actual_groups = gamma * groups * num_estimators

        if extended_in_channels % actual_groups:
            raise ValueError(
                f"PackedConv3d: `in_channels` ({in_channels}) is not divisible by "
                f"`actual_groups` ({actual_groups}). Please make `in_channels` divisible "
                f"by `gamma * groups * num_estimators` ({gamma * groups * num_estimators})."
            )
        if extended_out_channels % (gamma * groups * num_estimators):
            raise ValueError(
                f"PackedConv3d: `out_channels` ({out_channels}) is not divisible by "
                f"`gamma * groups * num_estimators` ({gamma * groups * num_estimators}). "
                f"Please make `out_channels` divisible by `gamma * groups * num_estimators` "
                f"({gamma * groups * num_estimators})."
            )

        self.conv = nn.Conv3d(
            in_channels=extended_in_channels,
            out_channels=extended_out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=actual_groups,
            bias=bias,
            padding_mode=padding_mode,
            **factory_kwargs,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        out = self.conv(inputs)
        return (
            out
            if not self.last
            else rearrange(out, "b (m c) ... -> (m b) c ...", m=self.num_estimators)
        )

    @property
    def weight(self) -> Tensor:
        r"""The weight of the underlying convolutional layer."""
        return self.conv.weight

    @property
    def bias(self) -> Tensor | None:
        r"""The bias of the underlying convolutional layer."""
        return self.conv.bias


class PackedConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        alpha: int,
        num_estimators: int,
        gamma: int = 1,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        output_padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        minimum_channels_per_group: int = 64,
        bias: bool = True,
        first: bool = False,
        last: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        r"""Packed-Ensembles-style ConvTranspose2d layer with debug flags.

        Args:
            in_channels (int): Number of channels in the input.
            out_channels (int): Number of channels produced by the transposed convolution.
            kernel_size (int or tuple): Size of the convolving kernel.
            alpha (int): The channel multiplier for the layer.
            num_estimators (int): Number of estimators in the ensemble.
            gamma (int, optional): Defaults to ``1``.
            stride (int or tuple, optional): Stride of the convolution. Defaults to ``1``.
            padding (int or tuple, optional): Zero-padding added to both sides of the input. Defaults to ``0``.
            output_padding (int or tuple, optional): Additional size added to one side of the output shape. Defaults to ``0``.
            dilation (int or tuple, optional): Spacing between kernel elements. Defaults to ``1``.
            groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to ``1``.
            minimum_channels_per_group (int, optional): Smallest possible number of channels per group.
            bias (bool, optional): If ``True``, adds a learnable bias to the output. Defaults to ``True``.
            first (bool, optional): Whether this is the first layer of the network. Defaults to ``False``.
            last (bool, optional): Whether this is the last layer of the network. Defaults to ``False``.
            device (torch.device, optional): The device to use for the layer's parameters. Defaults to ``None``.
            dtype (torch.dtype, optional): The dtype to use for the layer's parameters. Defaults to ``None``.
        """
        check_packed_parameters_consistency(alpha, gamma, num_estimators)
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.num_estimators = num_estimators
        self.first = first
        self.last = last

        # Define the number of channels for the underlying convolution
        self.extended_in_channels = int(in_channels * (1 if first else alpha))
        self.extended_out_channels = int(out_channels * (num_estimators if last else alpha))

        # Define the number of groups of the underlying convolution
        self.actual_groups = 1 if first else gamma * groups * num_estimators

        while (
            self.extended_in_channels % self.actual_groups != 0
            or self.extended_in_channels // self.actual_groups < minimum_channels_per_group
        ) and self.actual_groups // (groups * num_estimators) > 1:
            gamma -= 1
            self.actual_groups = gamma * groups * num_estimators

        if self.extended_in_channels % self.actual_groups:
            raise ValueError(
                f"PackedConvTranspose2d: `in_channels` ({in_channels}) is not divisible by "
                f"`actual_groups` ({self.actual_groups}). Please make `in_channels` divisible "
                f"by `gamma * groups * num_estimators` ({gamma * groups * num_estimators})."
            )
        if self.extended_out_channels % (gamma * groups * num_estimators):
            raise ValueError(
                f"PackedConvTranspose2d: `out_channels` ({out_channels}) is not divisible by "
                f"`gamma * groups * num_estimators` ({gamma * groups * num_estimators}). "
                f"Please make `out_channels` divisible by `gamma * groups * num_estimators` "
                f"({gamma * groups * num_estimators})."
            )

        # Initialize the transposed convolutional layer
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=self.extended_in_channels,
            out_channels=self.extended_out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=self.actual_groups,
            bias=bias,
            **factory_kwargs,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        out = self.conv_transpose(inputs)
        return (
            out
            if not self.last
            else rearrange(out, "b (m c) ... -> (m b) c ...", m=self.num_estimators)
        )

    @property
    def weight(self) -> Tensor:
        r"""The weight of the underlying transposed convolutional layer."""
        return self.conv_transpose.weight

    @property
    def bias(self) -> Tensor | None:
        r"""The bias of the underlying transposed convolutional layer."""
        return self.conv_transpose.bias


class PackedLayerNorm(nn.GroupNorm):
    def __init__(
        self,
        embed_dim: int,
        num_estimators: int,
        alpha: float,
        eps: float = 1e-5,
        affine: bool = True,
        first: bool = False,
        last: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        r"""Packed-Ensembles-style LayerNorm layer.

        Args:
            embed_dim (int): the number of features in the input tensor.
            num_estimators (int): the number of estimators in the ensemble.
            alpha (float): the width multiplier of the layer.
            eps (float, optional): a value added to the denominator for numerical stability. Defaults to 1e-5.
            affine (bool, optional): a boolean value that when set to ``True``, this module has learnable per_channel affine parameters initialized to ones (for weights) and zeros (for biases). Defaults to ``True``.
            first (bool, optional): Whether this layer processes the raw inputs of the model. Defaults to ``False``.
            last (bool, optional): Whether this layer processes the final outputs of the model. Defaults to ``False``.
            device (torch.device, optional): The device to use for the layer's parameters. Defaults to ``None``.
            dtype (torch.dtype, optional): The dtype to use for the layer's parameters. Defaults to ``None``.

        Shape:
            - Input: :math:`(N, *)` where :math:`*` means any number of additional dimensions.
            - Output: :math:`(N, *)` (same shape as input)

        Warnings:
            This layer is only suitable to replace ``nn.LayerNorm`` when only the last dimension is normalized.
        """
        expansion_factor = num_estimators if last else alpha
        num_groups = num_estimators if first else 1

        super().__init__(
            num_groups=num_groups,
            num_channels=int(embed_dim * expansion_factor),
            eps=eps,
            affine=affine,
            device=device,
            dtype=dtype,
        )

    def forward(self, input: Tensor) -> Tensor:  # noqa: A002
        shapes = {f"d{i}": size for i, size in enumerate(input.shape[1:-1])}
        shape_str = " ".join(shapes.keys())

        x = rearrange(input, "b ... h -> (b ...) h")
        x = F.group_norm(
            x,
            self.num_groups,
            self.weight,
            self.bias,
            self.eps,
        )
        return rearrange(x, f"(b {shape_str}) h -> b {shape_str} h", **shapes)


class PackedMultiheadAttention(nn.Module):
    __constants__ = ["batch_first"]
    bias_k: Tensor | None
    bias_v: Tensor | None

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        alpha: float,
        num_estimators: int,
        gamma: int = 1,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim: int | None = None,
        vdim: int | None = None,
        batch_first=False,
        first=False,
        last=False,
        device=None,
        dtype=None,
    ) -> None:
        r"""Packed-Ensembles-style MultiheadAttention layer.

        Args:
            embed_dim (int): Size of the embedding dimension.
            num_heads (int): Number of parallel attention heads.
            alpha (float): The width multiplier of the embedding dimension.
            num_estimators (int): The number of estimators packed in the layer.
            gamma (int, optional): Defaults to ``1``.
            dropout (float, optional): Dropout probability on ``attn_output_weights``. Defaults to ``0.0``
                (no dropout).
            bias (bool, optional): If specified, adds bias to input / output projection layers.
                Defaults to ``True``.
            add_bias_kv (bool, optional): If specified, adds bias to the key and value sequences at
                ``dim=0``. Defaults to ``False``.
            add_zero_attn (bool, optional): If specified, adds a new batch of zeros to the key and
                value sequences at ``dim=1``. Defaults to ``False``.
            kdim (int | None, optional): Total number of features for keys. Defaults to ``None``
                (uses ``kdim=embed_dim``).
            vdim (int | None, optional): Total number of features for values. Defaults to ``None``
                (uses ``vdim=embed_dim``).
            batch_first (bool, optional): If ``True``, then the input and output tensors are provided
                as (batch, seq, feature). Defaults to ``False`` (seq, batch, feature).
            first (bool, optional): Whether this is the first layer of the network. Defaults to
                ``False``.
            last (bool, optional): Whether this is the last layer of the network. Defaults to
                ``False``.
            device (torch.device, optional): The device to use for the layer's parameters. Defaults
                to ``None``.
            dtype (torch.dtype, optional): The dtype to use for the layer's parameters. Defaults to
                ``None``.

        Reference:
            - `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_: Original Multihead Attention formulation.
            - `Hierarchical Light Tranformer Ensembles for Multimodal Trajectory Forecasting <https://arxiv.org/abs/2403.17678>`_
              : Packed-Ensembles-style Multihead Attention formulation.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.embed_dim = int(embed_dim * alpha)

        augmentation = 1 if first else alpha
        in_embed_dim = int(embed_dim * augmentation)
        self.kdim = int(self.kdim * augmentation)
        self.vdim = int(self.vdim * augmentation)

        self.num_groups = 1 if first else num_estimators * gamma

        self.num_heads = num_heads * self.num_groups
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )

        self.num_estimators = num_estimators
        self.alpha = alpha
        self.gamma = gamma
        self.first = first
        self.last = last

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = nn.Parameter(
                torch.empty(
                    (
                        self.num_groups,
                        self.embed_dim // self.num_groups,
                        in_embed_dim // self.num_groups,
                    ),
                    **factory_kwargs,
                )
            )
            self.k_proj_weight = nn.Parameter(
                torch.empty(
                    (
                        self.num_groups,
                        self.embed_dim // self.num_groups,
                        self.kdim // self.num_groups,
                    ),
                    **factory_kwargs,
                )
            )
            self.v_proj_weight = nn.Parameter(
                torch.empty(
                    (
                        self.num_groups,
                        self.embed_dim // self.num_groups,
                        self.vdim // self.num_groups,
                    ),
                    **factory_kwargs,
                )
            )
            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = nn.Parameter(
                torch.empty(
                    (
                        self.num_groups,
                        3 * self.embed_dim // self.num_groups,
                        in_embed_dim // self.num_groups,
                    ),
                    **factory_kwargs,
                )
            )
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * self.embed_dim, **factory_kwargs))
        else:
            self.register_parameter("in_proj_bias", None)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty((1, 1, self.embed_dim), **factory_kwargs))
            self.bias_v = nn.Parameter(torch.empty((1, 1, self.embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        out_embed_dim = int(embed_dim * (num_estimators if last else alpha))

        self.out_proj_weight = nn.Parameter(
            torch.empty(
                (
                    self.num_groups,
                    out_embed_dim // self.num_groups,
                    self.embed_dim // self.num_groups,
                ),
                **factory_kwargs,
            )
        )
        if bias:
            self.out_proj_bias = nn.Parameter(torch.empty(out_embed_dim, **factory_kwargs))
        else:
            self.register_parameter("out_proj_bias", None)

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        if self._qkv_same_embed_dim:
            for i in range(self.in_proj_weight.size(0)):
                nn.init.xavier_uniform_(self.in_proj_weight[i])
        else:
            for i in range(self.q_proj_weight.size(0)):
                nn.init.xavier_uniform_(self.q_proj_weight[i])
                nn.init.xavier_uniform_(self.k_proj_weight[i])
                nn.init.xavier_uniform_(self.v_proj_weight[i])

        for i in range(self.out_proj_weight.size(0)):
            nn.init.xavier_uniform_(self.out_proj_weight[i])

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj_bias, 0.0)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = False,
        attn_mask: Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[Tensor, None]:
        r"""Computes attention outputs given query, key, and value tensors.

        Args:
            query (Tensor): Query embeddings of shape :math:`(L, E_q)` for unbatched input,
                :math:`(L, B, E_q)` when ``batch_first=False`` or :math:`(B, L, E_q)` when
                ``batch_first=True``, where :math:`L` is the target sequence length, :math:`B` is
                the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
            key (Tensor): Key embeddings of shape :math:`(S, E_k)` for unbatched input,
                :math:`(S, B, E_k)` when ``batch_first=False`` or :math:`(B, S, E_k)` when
                ``batch_first=True``, where :math:`S` is the source sequence length, :math:`B` is
                the batch size and :math:`E_k` is the key embedding dimension ``kdim``.
            value (Tensor): Value embeddings of shape :math:`(S, E_v)` for unbatched input,
                :math:`(S, B, E_v)` when ``batch_first=False`` or :math:`(B, S, E_v)` when
                ``batch_first=True``, where :math:`S` is the source sequence length, :math:`B` is
                the batch size and :math:`E_v` is the value embedding dimension ``vdim``.
            key_padding_mask (Tensor | None, optional): If specified, a mask of shape
                :math:`(B, S)` indicating which elements within ``key`` to ignore for the purpose
                of attention (i.e. treat as "padding"). For unbatched `query`, shape should be
                :math:`(S)`. Binary and float masks are supported. For a binary mask, a ``True``
                value indicates that the corresponding ``key`` value will be ignored for the
                purpose of attention. For a float mask, it will be directly added to the
                corresponding ``key`` value. Defaults to ``None``.
            need_weights (bool, optional): If specified, returns ``attn_output_weights`` in
                addition to ``attn_outputs``. Set ``need_weights=False`` to use the optimized
                ``scale_dot_product_attention`` and achieve the best performance for MHA.
                Defaults to ``False``.
            attn_mask (Tensor | None, optional): If specified, a 2D or 3D mask preventing attention
                to certain positions. Must be of shape :math:`(L,S)` or
                :math:`(B \times \text{num_heads}, L, S)`, where :math:`B` is the batch size, :math:`L`
                is the target sequence length, and :math:`S` is the source sequence length. A 2D mask
                will be broadcasted across the batch while a 3D mask allows for a different mask for
                each entry in the batch. Binary and float masks are supported. For a binary mask, a
                ``True`` value indicates that the corresponding position is not allowed to attend to.
                For a float mask, the mask values will be added to the attention weight. If both
                ``attn_mask`` and ``key_padding_mask`` are provided, their types should match.
                Defaults to ``None``.
            average_attn_weights (bool, optional): If ``True``, indicates that the returned
                ``attn_weights`` should be averaged across heads. Otherwise, ``attn_weights`` are
                provided separately per head. Note that this flag only has an effect when
                ``need_weights=True``. Defaults to ``True``.
            is_causal (bool, optional): If specified, applies a causal mask as ``attn_mask``.
                Defaults to ``False``.

        Warning:
            ``need_weights=True`` and therefore ``average_attn_weights`` are not supported yet thus
            have no effect.

        Returns:
            tuple[Tensor, None]:
                - *attn_output* (Tensor): The output tensor of shape :math:`(L, E_q)`, :math:`(L, B, E_q)`
                  or :math:`(B, L, E_q)` where :math:`L` is the target sequence length, :math:`B` is
                  the batch size, and :math:`E_q` is the embedding dimension ``embed_dim``.
                - *attn_output_weights* (None): Always ``None`` as we do not support
                  ``need_weights=True`` yet.
        """
        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        if not self._qkv_same_embed_dim:
            (
                attn_output,
                _,
            ) = packed_multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.num_groups,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj_weight,
                self.out_proj_bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        else:
            (
                attn_output,
                _,
            ) = packed_multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.num_groups,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj_weight,
                self.out_proj_bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )

        if self.last:
            attn_output = rearrange(attn_output, "l b (m e) -> l (m b) e", m=self.num_estimators)

        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), None
        return attn_output, None


class PackedTransformerEncoderLayer(nn.Module):
    __constants__ = ["batch_first", "norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        alpha: float,
        num_estimators: int,
        gamma: int = 1,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable[[Tensor], Tensor] = F.relu,
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
        batch_first: bool = False,
        norm_first: bool = False,
        first: bool = False,
        last: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        r"""Packed-Ensembles-style TransformerEncoderLayer (made up of self-attention followed by a
        feedforward network).

        Args:
            d_model (int): the number of expected features in the input.
            nhead (int): the number of heads in the multiheadattention models.
            alpha (float): the width multiplier of the layer.
            num_estimators (int): the number of estimators packed in the layer.
            gamma (int, optional): Defaults to ``1``.
            dim_feedforward (int, optional): the dimension of the feedforward network model. Defaults
                to ``2048``.
            dropout (float, optional): the dropout value. Defaults to ``0.1``.
            activation (Callable[[Tensor], Tensor], optional): the activation function of the
                intermediate layer, that is a unary callable. Defaults to ``F.relu``.
            layer_norm_eps (float, optional): the eps value in layer normalization components. Defaults
                to ``1e-5``.
            bias (bool, optional): If ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an
                additive bias. Defaults to ``True``.
            batch_first (bool, optional): If ``True``, then the input and output tensors are provided
                as :math:`(\text{batch}, \text{seq}, \text{d_model})`. Defaults to ``False``
                :math:`(\text{seq}, \text{batch}, \text{d_model})`.
            norm_first (bool, optional): If ``True``, the layer norm is done prior to attention and
                feedforward operations, respectively. Otherwise, it is done after. Defaults to
                ``False``.
            first (bool, optional): Whether this is the first layer of the network. Defaults to
                ``False``.
            last (bool, optional): Whether this is the last layer of the network. Defaults to
                ``False``.
            device (torch.device, optional): The device to use for the layer's parameters. Defaults
                to ``None``.
            dtype (torch.dtype, optional): The dtype to use for the layer's parameters. Defaults to
                ``None``.

        Reference:
            - `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_: Original Multihead Attention formulation.
            - `Hierarchical Light Tranformer Ensembles for Multimodal Trajectory Forecasting <https://arxiv.org/abs/2403.17678>`_
              : Packed-Ensembles-style Multihead Attention formulation.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.self_attn = PackedMultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            alpha=alpha,
            num_estimators=num_estimators,
            bias=bias,
            gamma=gamma,
            dropout=dropout,
            batch_first=batch_first,
            first=first,
            **factory_kwargs,
        )

        self.linear1 = PackedLinear(
            in_features=d_model,
            out_features=dim_feedforward,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            implementation="einsum",
            bias=bias,
            **factory_kwargs,
        )
        self.dropout = nn.Dropout(dropout)
        self.linear2 = PackedLinear(
            in_features=dim_feedforward,
            out_features=d_model,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            implementation="einsum",
            last=last,
            bias=bias,
            **factory_kwargs,
        )

        self.norm_first = norm_first

        self.norm1 = PackedLayerNorm(
            embed_dim=d_model,
            num_estimators=num_estimators,
            alpha=alpha,
            eps=layer_norm_eps,
            first=first and self.norm_first,
            **factory_kwargs,
        )

        self.norm2 = PackedLayerNorm(
            embed_dim=d_model,
            num_estimators=num_estimators,
            alpha=alpha,
            eps=layer_norm_eps,
            last=last and not self.norm_first,
            **factory_kwargs,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src (Tensor): The sequence to the encoder layer. Shape: :math:`(B, L, E)` or
                :math:`(L, B, E)`.
            src_mask (Tensor | None, optional): The mask for the ``src`` sequence. Defaults to ``None``.
            src_key_padding_mask (Tensor | None, optional): The mask for the ``src`` keys per
                batch. Defaults to ``None``.
            is_causal (bool, optional): If specified, applies a causal mask as ``src_mask``.
                Defaults to ``False``. Warning: ``is_causal`` provides a hint the ``src_mask`` is
                a causal mask. Providing incorrect hints can result in incorrect execution,
                including forward and backward compatibility.

        Returns:
            Tensor: The output of the encoder layer. Shape: :math:`(B, L, E)` or :math:`(L, B, E)`.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype,
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x),
                src_mask,
                src_key_padding_mask,
                is_causal=is_causal,
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)
            )
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        is_causal: bool = False,
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
        )[0]
        return self.dropout1(x)

    # feed-forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class PackedTransformerDecoderLayer(nn.Module):
    __constants__ = ["batch_first", "norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        alpha: float,
        num_estimators: int,
        gamma: int = 1,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable[[Tensor], Tensor] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        first: bool = False,
        last: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        r"""Packed-Ensembles-style TransformerDecoderLayer (made up of self-attention, multi-head
        attention, and feedforward network).

        Args:
            d_model (int): the number of expected features in the input.
            nhead (int): the number of heads in the multiheadattention models.
            alpha (float): the width multiplier of the layer.
            num_estimators (int): the number of estimators packed in the layer.
            gamma (int, optional): Defaults to ``1``.
            dim_feedforward (int, optional): the dimension of the feedforward network model. Defaults
                to ``2048``.
            dropout (float, optional): the dropout value. Defaults to ``0.1``.
            activation (Callable[[Tensor], Tensor], optional): the activation function of the
                intermediate layer, that is a unary callable. Defaults to ``F.relu``.
            layer_norm_eps (float, optional): the eps value in layer normalization components. Defaults
                to ``1e-5``.
            bias (bool, optional): If ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an
                additive bias. Defaults to ``True``.
            batch_first (bool, optional): If ``True``, then the input and output tensors are provided
                as :math:`(\text{batch}, \text{seq}, \text{d_model})`. Defaults to ``False``
                :math:`(\text{seq}, \text{batch}, \text{d_model})`.
            norm_first (bool, optional): If ``True``, the layer norm is done prior to attention and
                feedforward operations, respectively. Otherwise, it is done after. Defaults to
                ``False``.
            first (bool, optional): Whether this is the first layer of the network. Defaults to
                ``False``.
            last (bool, optional): Whether this is the last layer of the network. Defaults to
                ``False``.
            device (torch.device, optional): The device to use for the layer's parameters. Defaults
                to ``None``.
            dtype (torch.dtype, optional): The dtype to use for the layer's parameters. Defaults to
                ``None``.

        Reference:
            - `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_: Original Multihead Attention formulation.
            - `Hierarchical Light Tranformer Ensembles for Multimodal Trajectory Forecasting <https://arxiv.org/abs/2403.17678>`_
              : Packed-Ensembles-style Multihead Attention formulation.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.self_attn = PackedMultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            first=first,
            **factory_kwargs,
        )

        self.multihead_attn = PackedMultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            **factory_kwargs,
        )

        self.linear1 = PackedLinear(
            in_features=d_model,
            out_features=dim_feedforward,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            implementation="einsum",
            bias=bias,
            **factory_kwargs,
        )
        self.dropout = nn.Dropout(dropout)
        self.linear2 = PackedLinear(
            in_features=dim_feedforward,
            out_features=d_model,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            implementation="einsum",
            bias=bias,
            last=last,
            **factory_kwargs,
        )

        self.norm_first = norm_first

        self.norm1 = PackedLayerNorm(
            embed_dim=d_model,
            num_estimators=num_estimators,
            alpha=alpha,
            eps=layer_norm_eps,
            first=first and self.norm_first,
            **factory_kwargs,
        )

        self.norm2 = PackedLayerNorm(
            embed_dim=d_model,
            num_estimators=num_estimators,
            alpha=alpha,
            eps=layer_norm_eps,
            **factory_kwargs,
        )

        self.norm3 = PackedLayerNorm(
            embed_dim=d_model,
            num_estimators=num_estimators,
            alpha=alpha,
            eps=layer_norm_eps,
            last=last and not self.norm_first,
            **factory_kwargs,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the input (and mask) through the decoder layer.

        Args:
            tgt (Tensor): The sequence to the decoder layer. Shape: :math:`(B, L, E)` or
                :math:`(L, B, E)`.
            memory (Tensor): The sequence from the last layer of the encoder. Shape:
                :math:`(B, S, E)` or :math:`(S, B, E)`.
            tgt_mask (Tensor | None, optional): The mask for the ``tgt`` sequence. Defaults to
                ``None``.
            memory_mask (Tensor | None, optional): The mask for the ``memory`` sequence. Defaults
                to ``None``.
            tgt_key_padding_mask (Tensor | None, optional): The mask for the ``tgt`` keys per
                batch. Defaults to ``None``.
            memory_key_padding_mask (Tensor | None, optional): The mask for the ``memory`` keys per
                batch. Defaults to ``None``.
            tgt_is_causal (bool, optional): If specified, applies a causal mask as ``tgt_mask``.
                Defaults to ``False``. Warning: ``tgt_is_causal`` provides a hint the ``tgt_mask``
                is a causal mask. Providing incorrect hints can result in incorrect execution,
                including forward and backward compatibility.
            memory_is_causal (bool, optional): If specified, applies a causal mask as ``memory_mask``.
                Defaults to ``False``. Warning: ``memory_is_causal`` provides a hint the ``memory_mask``
                is a causal mask. Providing incorrect hints can result in incorrect execution,
                including forward and backward compatibility.

        Returns:
            Tensor: The output of the decoder layer. Shape: :math:`(B, L, E)` or :math:`(L, B, E)`.
        """
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x = x + self._mha_block(
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
                memory_is_causal,
            )
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = self.norm2(
                x
                + self._mha_block(
                    x,
                    memory,
                    memory_mask,
                    memory_key_padding_mask,
                    memory_is_causal,
                )
            )
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        is_causal: bool = False,
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
        )[0]
        return self.dropout1(x)

    # multi-head attention block
    def _mha_block(
        self,
        x: Tensor,
        memory: Tensor,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        is_causal: bool = False,
    ) -> Tensor:
        x = self.multihead_attn(
            x,
            memory,
            memory,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
        )[0]
        return self.dropout2(x)

    # feed-forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)
