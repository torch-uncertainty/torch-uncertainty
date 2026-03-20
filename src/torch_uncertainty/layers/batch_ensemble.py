import math
from typing import Literal

import torch
from einops import repeat
from torch import Tensor, nn
from torch.nn.common_types import _size_1_t, _size_2_t
from torch.nn.modules.utils import _pair


class BatchLinear(nn.Module):
    __constants__ = ["in_features", "out_features", "num_estimators"]
    in_features: int
    out_features: int
    num_estimators: int
    r_group: Tensor
    s_group: Tensor
    bias: Tensor | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_estimators: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        r"""BatchEnsemble-style Linear layer.

        Apply a linear transformation using BatchEnsemble method to the incoming
        data.

        .. math::
            y=(x\circ \widehat{r_{group}})W^{T}\circ \widehat{s_{group}} + \widehat{b}

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            num_estimators (int): Number of estimators in the ensemble, referred as
                :math:`M`.
            bias (bool, optional): If ``True``, adds a learnable bias to the
                output. Defaults to ``True``.
            device (Any, optional): Device to use for the parameters and
                buffers of this module. Defaults to ``None``.
            dtype (Any, optional): Data type to use for the parameters and
                buffers of this module. Defaults to ``None``.

        Reference:
            Introduced by the paper `BatchEnsemble: An Alternative Approach to
            Efficient Ensemble and Lifelong Learning
            <https://arxiv.org/abs/2002.06715>`_, we present here an implementation
            of a Linear BatchEnsemble layer in `PyTorch <https://pytorch.org>`_
            heavily inspired by its `official implementation
            <https://github.com/google/edward2>`_ in `TensorFlow
            <https://www.tensorflow.org>`_.

        Attributes:
            weight: The learnable weights (:math:`W`) of shape
                :math:`(H_{out}, H_{in})` shared between the estimators. The values
                are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`,
                where :math:`k = \frac{1}{H_{in}}`.
            r_group: The learnable matrice of shape :math:`(M, H_{in})` where each row
                consist of the vector :math:`r_{i}` corresponding to the
                :math:`i^{th}` ensemble member. The values are initialized from
                :math:`\mathcal{N}(1.0, 0.5)`.
            s_group: The learnable matrice of shape :math:`(M, H_{out})` where each row
                consist of the vector :math:`s_{i}` corresponding to the
                :math:`i^{th}` ensemble member. The values are initialized from
                :math:`\mathcal{N}(1.0, 0.5)`.
            bias: The learnable bias (:math:`b`) of shape :math:`(M, H_{out})`
                where each row corresponds to the bias of the :math:`i^{th}`
                ensemble member. If :attr:`bias` is ``True``, the values are
                initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{H_{in}}`.

        Shape:
            - Input: :math:`(N, H_{in})` where :math:`N` is the batch size and
              :math:`H_{in} = \text{in_features}`.
            - Output: :math:`(N, H_{out})` where
              :math:`H_{out} = \text{out_features}`.

        Warning:
            It is advised to ensure that `batch_size` is divisible by :attr:`num_estimators` when
            calling :func:`forward()`, so each estimator receives the same number of examples.
            In a BatchEnsemble architecture, the input is typically **repeated** `num_estimators`
            times along the batch dimension. Incorrect batch size may lead to unexpected results.

            To simplify batch handling, wrap your model with `torch_uncertainty.wrappers.BatchEnsemble`,
            which automatically repeats the batch before passing it through the network.


        Examples:
            >>> # With three estimators
            >>> m = BatchLinear(20, 30, 3)
            >>> input = torch.randn(8, 20)
            >>> output = m(input)
            >>> print(output.size())
            torch.Size([8, 30])
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_estimators = num_estimators

        self.linear = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
            **factory_kwargs,
        )

        self.r_group = nn.Parameter(torch.empty((num_estimators, in_features), **factory_kwargs))
        self.s_group = nn.Parameter(torch.empty((num_estimators, out_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((num_estimators, out_features), **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    @classmethod
    def from_linear(cls, linear: nn.Linear, num_estimators: int) -> "BatchLinear":
        r"""Create a BatchEnsemble-style Linear layer from an existing Linear layer.

        Args:
            linear (nn.Linear): The Linear layer to convert.
            num_estimators (int): Number of ensemble members.

        Returns:
            BatchLinear: The converted BatchEnsemble-style Linear layer.

        Example:
            >>> linear = nn.Linear(20, 30)
            >>> be_linear = BatchLinear.from_linear(linear, num_estimators=3)
        """
        return cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            num_estimators=num_estimators,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )

    def reset_parameters(self) -> None:
        nn.init.normal_(self.r_group, mean=1.0, std=0.5)
        nn.init.normal_(self.s_group, mean=1.0, std=0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.linear.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size = inputs.size(0)
        examples_per_estimator = torch.tensor(
            batch_size // self.num_estimators, device=inputs.device
        )
        extra = batch_size % self.num_estimators

        r_group = repeat(self.r_group, "m h -> (m b) h", b=examples_per_estimator)
        r_group = torch.cat([r_group, r_group[:extra]], dim=0)
        s_group = repeat(self.s_group, "m h -> (m b) h", b=examples_per_estimator)
        s_group = torch.cat([s_group, s_group[:extra]], dim=0)
        if self.bias is not None:
            bias = repeat(self.bias, "m h -> (m b) h", b=examples_per_estimator)
            bias = torch.cat([bias, bias[:extra]], dim=0)
        else:
            bias = None

        return self.linear(inputs * r_group) * s_group + (bias if bias is not None else 0)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features},"
            f" out_features={self.out_features},"
            f" num_estimators={self.num_estimators},"
            f" bias={self.bias is not None}"
        )


class BatchConv1d(nn.Module):
    __constants__ = [
        "stride",
        "padding",
        "dilation",
        "groups",
        "in_channels",
        "out_channels",
        "kernel_size",
        "num_estimators",
    ]
    in_channels: int
    out_channels: int
    kernel_size: tuple[int, ...]
    num_estimators: int
    stride: tuple[int, ...]
    padding: str | tuple[int, ...]
    dilation: tuple[int, ...]
    groups: int
    weight: Tensor
    r_group: Tensor
    s_group: Tensor
    bias: Tensor | None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        num_estimators: int,
        stride: _size_1_t = 1,
        padding: str | _size_1_t = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        r"""BatchEnsemble-style Conv1d layer.

        Applies a 1d convolution over an input signal composed of several input
        planes using BatchEnsemble method to the incoming data.

        In the simplest case, the output value of the layer with input size
        :math:`(N, C_{in}, L_{in})` and output
        :math:`(N, C_{out}, L_{out})` can be precisely described as:

        .. math::
            \text{out}(N_i, C_{\text{out}_j})=\
                &\widehat{b}(N_i,C_{\text{out}_j})
            +\widehat{s_{group}}(N_{i},C_{\text{out}_j}) \\
                &\times \sum_{k = 0}^{C_{\text{in}} - 1}
            \text{weight}(C_{\text{out}_j}, k)\star (\text{input}(N_i, k)
            \times \widehat{r_{group}}(N_i, k))

        Reference:
            Introduced by the paper `BatchEnsemble: An Alternative Approach to
            Efficient Ensemble and Lifelong Learning
            <https://arxiv.org/abs/2002.06715>`_, we present here an implementation
            of a Conv2d BatchEnsemble layer in `PyTorch <https://pytorch.org>`_
            heavily inspired by its `official implementation
            <https://github.com/google/edward2>`_ in `TensorFlow
            <https://www.tensorflow.org>`_.

        Args:
            in_channels (int): Number of channels in the input images.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int): Size of the convolving kernel.
            num_estimators (int): Number of estimators in the ensemble referred as
                :math:`M` here.
            stride (int, optional): Stride of the convolution. Defaults to
                ``1``.
            padding (int or str, optional): Padding added to all four sides
                of the input. Defaults to ``0``.
            dilation (int, optional): Spacing between kernel elements.
                Defaults to ``1``.
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Defaults to ``1``.
            bias (bool, optional): If ``True``, adds a learnable bias to the
                output. Defaults to ``True``.
            device (Any, optional): Device to use for the parameters and
                buffers of this module. Defaults to ``None``.
            dtype (Any, optional): Data type to use for the parameters and
                buffers of this module. Defaults to ``None``.

        Attributes:
            weight: The learnable weights of the module of shape
                :math:`(\text{out_channels}, \frac{\text{in_channels}}
                {\text{groups}},`:math:`\text{kernel_size})` shared between the
                estimators. The values of these weights are sampled from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{\text{groups}}{C_\text{in} *
                \prod_{i=0}^{1}\text{kernel_size}[i]}`.
            r_group: The learnable matrice of shape :math:`(M, C_{in})` where each row
                consist of the vector :math:`r_{i}` corresponding to the
                :math:`i^{th}` ensemble member. The values are initialized from
                :math:`\mathcal{N}(1.0, 0.5)`.
            s_group: The learnable matrice of shape :math:`(M, C_{out})` where each row
                consist of the vector :math:`s_{i}` corresponding to the
                :math:`i^{th}` ensemble member. The values are initialized from
                :math:`\mathcal{N}(1.0, 0.5)`.
            bias: The learnable bias (:math:`b`) of shape :math:`(M, C_{out})`
                where each row corresponds to the bias of the :math:`i^{th}`
                ensemble member. If :attr:`bias` is ``True``, the values are
                initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k=\frac{\text{groups}}{C_\text{in}*\prod_{i=0}^{1}
                \text{kernel_size}[i]}`.

        Shape:
            - Input: :math:`(N, C_{in}, L_{in})`.
            - Output: :math:`(N, C_{out}, L_{out})`.

        .. math::
            L_{out} = \left\lfloor\frac{L_{in} + 2 \times \text{padding} -
            \text{dilation} \times (\text{kernel_size} - 1) - 1}
            {\text{stride}} + 1\right\rfloor

        Warning:
            Ensure that `batch_size` is divisible by :attr:`num_estimators` when calling :func:`forward()`.
            In a BatchEnsemble architecture, the input batch is typically **repeated** `num_estimators`
            times along the first axis. Incorrect batch size may lead to unexpected results.

            To simplify batch handling, wrap your model with `BatchEnsembleWrapper`, which automatically
            repeats the batch before passing it through the network. See `BatchEnsembleWrapper` for details.

        Examples:
            >>> # With square kernels, four estimators and equal stride
            >>> m = BatchConv1d(3, 32, 3, 4, stride=1)
            >>> input = torch.randn(8, 3, 16)
            >>> output = m(input)
            >>> print(output.size())
            torch.Size([8, 32, 14])
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_estimators = num_estimators
        self.stride = _pair(stride)
        self.padding = padding if isinstance(padding, str) else _pair(padding)
        self.dilation = _pair(dilation)

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
            **factory_kwargs,
        )
        self.r_group = nn.Parameter(torch.empty((num_estimators, in_channels), **factory_kwargs))
        self.s_group = nn.Parameter(torch.empty((num_estimators, out_channels), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((num_estimators, out_channels), **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    @classmethod
    def from_conv1d(cls, conv1d: nn.Conv1d, num_estimators: int) -> "BatchConv1d":
        r"""Create a BatchEnsemble-style Conv1d layer from an existing Conv1d layer.

        Args:
            conv1d (nn.Conv1d): The Conv1d layer to convert.
            num_estimators (int): Number of ensemble members.

        Returns:
            BatchConv1d: The converted BatchEnsemble-style Conv1d layer.

        Warning:
            All parameters of the original Conv1d layer will be discarded.

        Example:
            >>> conv1d = nn.Conv1d(3, 32, kernel_size=3)
            >>> be_conv1d = BatchConv1d.from_conv1d(conv1d, num_estimators=3)
        """
        return cls(
            in_channels=conv1d.in_channels,
            out_channels=conv1d.out_channels,
            kernel_size=conv1d.kernel_size,
            stride=conv1d.stride,
            padding=conv1d.padding,
            dilation=conv1d.dilation,
            groups=conv1d.groups,
            bias=conv1d.bias is not None,
            num_estimators=num_estimators,
            device=conv1d.weight.device,
            dtype=conv1d.weight.dtype,
        )

    def reset_parameters(self) -> None:
        nn.init.normal_(self.r_group, mean=1.0, std=0.5)
        nn.init.normal_(self.s_group, mean=1.0, std=0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size = inputs.size(0)
        examples_per_estimator = batch_size // self.num_estimators
        extra = batch_size % self.num_estimators

        r_group = repeat(self.r_group, "m h -> (m b) h 1", b=examples_per_estimator)
        r_group = torch.cat([r_group, r_group[:extra]], dim=0)
        s_group = repeat(self.s_group, "m h -> (m b) h 1", b=examples_per_estimator)
        s_group = torch.cat([s_group, s_group[:extra]], dim=0)

        if self.bias is not None:
            bias = repeat(self.bias, "m h -> (m b) h 1", b=examples_per_estimator)
            bias = torch.cat([bias, bias[:extra]], dim=0)
        else:
            bias = None

        return self.conv(inputs * r_group) * s_group + (bias if bias is not None else 0)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels},"
            f" out_channels={self.out_channels},"
            f" kernel_size={self.kernel_size},"
            f" num_estimators={self.num_estimators},"
            f" stride={self.stride}"
        )


class BatchConv2d(nn.Module):
    __constants__ = [
        "stride",
        "padding",
        "dilation",
        "groups",
        "in_channels",
        "out_channels",
        "kernel_size",
        "num_estimators",
    ]
    in_channels: int
    out_channels: int
    kernel_size: tuple[int, ...]
    num_estimators: int
    stride: tuple[int, ...]
    padding: str | tuple[int, ...]
    dilation: tuple[int, ...]
    groups: int
    weight: Tensor
    r_group: Tensor
    s_group: Tensor
    bias: Tensor | None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        num_estimators: int,
        stride: _size_2_t = 1,
        padding: str | _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        r"""BatchEnsemble-style Conv2d layer.

        Applies a 2d convolution over an input signal composed of several input
        planes using BatchEnsemble method to the incoming data.

        In the simplest case, the output value of the layer with input size
        :math:`(N, C_{in}, H_{in}, W_{in})` and output
        :math:`(N, C_{out}, H_{out}, W_{out})` can be precisely described as:

        .. math::
            \text{out}(N_i, C_{\text{out}_j})=\
                &\widehat{b}(N_i,C_{\text{out}_j})
            +\widehat{s_{group}}(N_{i},C_{\text{out}_j}) \\
                &\times \sum_{k = 0}^{C_{\text{in}} - 1}
            \text{weight}(C_{\text{out}_j}, k)\star (\text{input}(N_i, k)
            \times \widehat{r_{group}}(N_i, k))

        Reference:
            Introduced by the paper `BatchEnsemble: An Alternative Approach to
            Efficient Ensemble and Lifelong Learning
            <https://arxiv.org/abs/2002.06715>`_, we present here an implementation
            of a Conv2d BatchEnsemble layer in `PyTorch <https://pytorch.org>`_
            heavily inspired by its `official implementation
            <https://github.com/google/edward2>`_ in `TensorFlow
            <https://www.tensorflow.org>`_.

        Args:
            in_channels (int): Number of channels in the input images.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int or tuple): Size of the convolving kernel.
            num_estimators (int): Number of estimators in the ensemble referred as
                :math:`M` here.
            stride (int or tuple, optional): Stride of the convolution. Defaults to
                ``1``.
            padding (int, tuple or str, optional): Padding added to all four sides
                of the input. Defaults to ``0``.
            dilation (int or tuple, optional): Spacing between kernel elements.
                Defaults to ``1``.
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Defaults to ``1``.
            bias (bool, optional): If ``True``, adds a learnable bias to the
                output. Defaults to ``True``.
            device (Any, optional): Device to use for the parameters and
                buffers of this module. Defaults to ``None``.
            dtype (Any, optional): Data type to use for the parameters and
                buffers of this module. Defaults to ``None``.

        Attributes:
            weight: The learnable weights of the module of shape
                :math:`(\text{out_channels}, \frac{\text{in_channels}}
                {\text{groups}},`:math:`\text{kernel_size[0]},
                \text{kernel_size[1]})` shared between the estimators. The values
                of these weights are sampled from :math:`\mathcal{U}(-\sqrt{k},
                \sqrt{k})` where :math:`k = \frac{\text{groups}}{C_\text{in} *
                \prod_{i=0}^{1}\text{kernel_size}[i]}`.
            r_group: The learnable matrice of shape :math:`(M, C_{in})` where each row
                consist of the vector :math:`r_{i}` corresponding to the
                :math:`i^{th}` ensemble member. The values are initialized from
                :math:`\mathcal{N}(1.0, 0.5)`.
            s_group: The learnable matrice of shape :math:`(M, C_{out})` where each row
                consist of the vector :math:`s_{i}` corresponding to the
                :math:`i^{th}` ensemble member. The values are initialized from
                :math:`\mathcal{N}(1.0, 0.5)`.
            bias: The learnable bias (:math:`b`) of shape :math:`(M, C_{out})`
                where each row corresponds to the bias of the :math:`i^{th}`
                ensemble member. If :attr:`bias` is ``True``, the values are
                initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k=\frac{\text{groups}}{C_\text{in}*\prod_{i=0}^{1}
                \text{kernel_size}[i]}`.

        Shape:
            - Input: :math:`(N, C_{in}, H_{in}, W_{in})`.
            - Output: :math:`(N, C_{out}, H_{out}, W_{out})`.

        .. math::
            H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[0] -
            \text{dilation}[0] \times (\text{kernel_size}[0] - 1) - 1}
            {\text{stride}[0]} + 1\right\rfloor

        .. math::
            W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] -
            \text{dilation}[1] \times (\text{kernel_size}[1] - 1) - 1}
            {\text{stride}[1]} + 1\right\rfloor

        Warning:
            Ensure that `batch_size` is divisible by :attr:`num_estimators` when calling :func:`forward()`.
            In a BatchEnsemble architecture, the input batch is typically **repeated** `num_estimators`
            times along the first axis. Incorrect batch size may lead to unexpected results.

            To simplify batch handling, wrap your model with `BatchEnsembleWrapper`, which automatically
            repeats the batch before passing it through the network. See `BatchEnsembleWrapper` for details.

        Examples:
            >>> # With square kernels, four estimators and equal stride
            >>> m = BatchConv2d(3, 32, 3, 4, stride=1)
            >>> input = torch.randn(8, 3, 16, 16)
            >>> output = m(input)
            >>> print(output.size())
            torch.Size([8, 32, 14, 14])
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.num_estimators = num_estimators
        self.stride = _pair(stride)
        self.padding = padding if isinstance(padding, str) else _pair(padding)
        self.dilation = _pair(dilation)

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
            **factory_kwargs,
        )
        self.r_group = nn.Parameter(torch.empty((num_estimators, in_channels), **factory_kwargs))
        self.s_group = nn.Parameter(torch.empty((num_estimators, out_channels), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((num_estimators, out_channels), **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    @classmethod
    def from_conv2d(cls, conv2d: nn.Conv2d, num_estimators: int) -> "BatchConv2d":
        r"""Create a BatchEnsemble-style Conv2d layer from an existing Conv2d layer.

        Args:
            conv2d (nn.Conv2d): The Conv2d layer to convert.
            num_estimators (int): Number of ensemble members.

        Returns:
            BatchConv2d: The converted BatchEnsemble-style Conv2d layer.

        Warning:
            All parameters of the original Conv2d layer will be discarded.

        Example:
            >>> conv2d = nn.Conv2d(3, 32, kernel_size=3)
            >>> be_conv2d = BatchConv2d.from_conv2d(conv2d, num_estimators=3)
        """
        return cls(
            in_channels=conv2d.in_channels,
            out_channels=conv2d.out_channels,
            kernel_size=conv2d.kernel_size,
            stride=conv2d.stride,
            padding=conv2d.padding,
            dilation=conv2d.dilation,
            groups=conv2d.groups,
            bias=conv2d.bias is not None,
            num_estimators=num_estimators,
            device=conv2d.weight.device,
            dtype=conv2d.weight.dtype,
        )

    def reset_parameters(self) -> None:
        nn.init.normal_(self.r_group, mean=1.0, std=0.5)
        nn.init.normal_(self.s_group, mean=1.0, std=0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size = inputs.size(0)
        examples_per_estimator = batch_size // self.num_estimators
        extra = batch_size % self.num_estimators

        r_group = repeat(self.r_group, "m h -> (m b) h 1 1", b=examples_per_estimator)
        r_group = torch.cat([r_group, r_group[:extra]], dim=0)
        s_group = repeat(self.s_group, "m h -> (m b) h 1 1", b=examples_per_estimator)
        s_group = torch.cat([s_group, s_group[:extra]], dim=0)

        if self.bias is not None:
            bias = repeat(self.bias, "m h -> (m b) h 1 1", b=examples_per_estimator)
            bias = torch.cat([bias, bias[:extra]], dim=0)
        else:
            bias = None

        return self.conv(inputs * r_group) * s_group + (bias if bias is not None else 0)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels},"
            f" out_channels={self.out_channels},"
            f" kernel_size={self.kernel_size},"
            f" num_estimators={self.num_estimators},"
            f" stride={self.stride}"
        )


class BatchConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        num_estimators: int,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        output_padding: _size_2_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_2_t = 1,
        padding_mode: Literal["circular", "reflect", "replicate", "zeros"] = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        r"""BatchEnsemble-style ConvTranspose2d layer.

        Args:
            in_channels (int): Number of channels in the input images.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (_size_2_t): Size of the convolving kernel.
            num_estimators (int): Number of estimators in the ensemble referred as
                :math:`M` here.
            stride (_size_2_t, optional): Stride of the convolution. Defaults to ``1``.
            padding (_size_2_t, optional): ``dilation * (kernel_size - 1) - padding`` zero-padding
                will be added to both sides of each dimension in the input. Defaults to ``0``.
            output_padding (_size_2_t, optional): Additional size added to one side
                of each dimension in the output shape. Defaults to ``0``.
            groups (int, optional): Number of blocked connections from input channels to output
                channels. Defaults to ``1``.
            bias (bool, optional): If ``True``, adds a learnable bias to the output. Defaults to
                ``True``.
            dilation (_size_2_t, optional): Spacing between kernel elements. Defaults to ``1``.
            padding_mode (str, optional): Padding mode for the convolution. Defaults to ``"zeros"``.
            device (Any, optional): Device to use for the parameters and
                buffers of this module. Defaults to ``None``.
            dtype (Any, optional): Data type to use for the parameters and
                buffers of this module. Defaults to ``None``.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.num_estimators = num_estimators
        self.stride = _pair(stride)
        self.padding = padding if isinstance(padding, str) else _pair(padding)
        self.dilation = _pair(dilation)

        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=False,
            dilation=dilation,
            padding_mode=padding_mode,
            **factory_kwargs,
        )

        self.r_group = nn.Parameter(torch.empty((num_estimators, in_channels), **factory_kwargs))
        self.s_group = nn.Parameter(torch.empty((num_estimators, out_channels), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((num_estimators, out_channels), **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    @classmethod
    def from_conv_transpose2d(
        cls, conv_transpose2d: nn.ConvTranspose2d, num_estimators: int
    ) -> "BatchConvTranspose2d":
        r"""Create a BatchEnsemble-style ConvTranspose2d layer from an existing ConvTranspose2d layer.

        Args:
            conv_transpose2d (nn.ConvTranspose2d): The ConvTranspose2d layer to convert.
            num_estimators (int): Number of ensemble members.

        Returns:
            BatchConvTranspose2d: The converted BatchEnsemble-style ConvTranspose2d layer.

        Warning:
            All parameters of the original ConvTranspose2d layer will be discarded.

        Example:
            >>> conv_transpose2d = nn.ConvTranspose2d(3, 32, kernel_size=3)
            >>> be_conv_transpose2d = BatchConvTranspose2d.from_conv_transpose2d(
            ...     conv_transpose2d, num_estimators=3
            ... )
        """
        return cls(
            in_channels=conv_transpose2d.in_channels,
            out_channels=conv_transpose2d.out_channels,
            kernel_size=conv_transpose2d.kernel_size,
            num_estimators=num_estimators,
            stride=conv_transpose2d.stride,
            padding=conv_transpose2d.padding,
            output_padding=conv_transpose2d.output_padding,
            groups=conv_transpose2d.groups,
            bias=conv_transpose2d.bias is not None,
            dilation=conv_transpose2d.dilation,
            padding_mode=conv_transpose2d.padding_mode,
            device=conv_transpose2d.weight.device,
            dtype=conv_transpose2d.weight.dtype,
        )

    def reset_parameters(self) -> None:
        nn.init.normal_(self.r_group, mean=1.0, std=0.5)
        nn.init.normal_(self.s_group, mean=1.0, std=0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv_transpose.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size = inputs.size(0)
        examples_per_estimator = batch_size // self.num_estimators
        extra = batch_size % self.num_estimators

        r_group = repeat(self.r_group, "m h -> (m b) h 1 1", b=examples_per_estimator)
        r_group = torch.cat([r_group, r_group[:extra]], dim=0)
        s_group = repeat(self.s_group, "m h -> (m b) h 1 1", b=examples_per_estimator)
        s_group = torch.cat([s_group, s_group[:extra]], dim=0)

        if self.bias is not None:
            bias = repeat(self.bias, "m h -> (m b) h 1 1", b=examples_per_estimator)
            bias = torch.cat([bias, bias[:extra]], dim=0)
        else:
            bias = None

        return self.conv_transpose(inputs * r_group) * s_group + (bias if bias is not None else 0)
