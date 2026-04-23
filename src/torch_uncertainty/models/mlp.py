from collections.abc import Callable

import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

from torch_uncertainty.layers import BatchLinear, BayesLinear, MaskedLinear, PackedLinear
from torch_uncertainty.layers.distributions import get_dist_linear_layer
from torch_uncertainty.methods import StochasticModel

__all__ = ["batched_mlp", "bayesian_mlp", "mimo_mlp", "mlp", "packed_mlp"]


class _MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_outputs: int,
        hidden_dims: list[int],
        layer: type[nn.Module],
        activation: Callable,
        layer_args: dict,
        dropout_rate: float,
        dist_family: str | None,
        dist_args: dict,
        flatten_start_dim: int,
    ) -> None:
        """Multi-layer perceptron class.

        Args:
            in_features (int): Number of input features.
            num_outputs (int): Number of output features.
            hidden_dims (list[int]): Number of features for each hidden layer.
            layer (nn.Module): Layer class.
            activation (Callable): Activation function.
            layer_args (Dict): Arguments for the layer class.
            dropout_rate (float): Dropout probability.
            dist_family (str, optional): Distribution family name. ``None`` means point-wise
                prediction.
            dist_args (Dict, optional): Arguments for the distribution layer class.
            flatten_start_dim (int, optional): Dimension to start flattening the input.
        """
        super().__init__()
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.flatten_start_dim = flatten_start_dim
        layers = nn.ModuleList()

        if len(hidden_dims) == 0:
            if layer == PackedLinear:
                layer_args |= {"first": True, "last": True}

            self.final_layer = layer(
                in_features=in_features, out_features=num_outputs, **layer_args
            )
        else:
            if layer == PackedLinear:
                layer_args |= {"first": True, "last": False}

            layers.append(layer(in_features=in_features, out_features=hidden_dims[0], **layer_args))

            if layer == PackedLinear:
                layer_args |= {"first": False}

            for i in range(1, len(hidden_dims)):
                layers.append(
                    layer(in_features=hidden_dims[i - 1], out_features=hidden_dims[i], **layer_args)
                )

            if layer == PackedLinear:
                layer_args |= {"last": True}

            if dist_family is not None:
                dist_layer_class = get_dist_linear_layer(dist_family)
                self.final_layer = dist_layer_class(
                    base_layer=layer,
                    event_dim=num_outputs,
                    in_features=hidden_dims[-1],
                    **layer_args,
                    **dist_args,
                )
            else:
                self.final_layer = layer(
                    in_features=hidden_dims[-1], out_features=num_outputs, **layer_args
                )

        self.layer_type = layer
        self.layer_args = layer_args
        self.probabilistic = dist_family is not None
        self.layers = layers
        self.fc_dropout = nn.Dropout(p=dropout_rate)
        self.last_fc_dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: Tensor) -> Tensor | dict[str, Tensor]:
        if self.layer_type == BatchLinear or self.layer_type == MaskedLinear:
            x = repeat(x, "b ... -> (m b) ...", m=self.layer_args["num_estimators"])
        x = x.flatten(self.flatten_start_dim)
        for i, layer in enumerate(self.layers):
            dropout = self.fc_dropout if i < len(self.layers) - 1 else self.last_fc_dropout
            x = dropout(layer(x))
            x = self.activation(x)
        return self.final_layer(x)


class _MIMOMLP(_MLP):
    def __init__(
        self,
        in_features: int,
        num_outputs: int,
        hidden_dims: list[int],
        num_estimators: int,
        layer: type[nn.Module],
        activation: Callable,
        layer_args: dict,
        dropout_rate: float,
        dist_family: str | None,
        dist_args: dict,
        flatten_start_dim: int,
    ) -> None:
        super().__init__(
            in_features=in_features * num_estimators,
            num_outputs=num_outputs * num_estimators,
            hidden_dims=hidden_dims,
            layer=layer,
            activation=activation,
            layer_args=layer_args,
            dropout_rate=dropout_rate,
            dist_family=dist_family,
            dist_args=dist_args,
            flatten_start_dim=flatten_start_dim,
        )
        self.num_estimators = num_estimators

    def forward(self, x: Tensor) -> Tensor | dict[str, Tensor]:
        if not self.training:
            x = repeat(x, "b ... -> (m b) ...", m=self.num_estimators)
        x = rearrange(x, "(m b) ... c -> b ... (m c)", m=self.num_estimators)
        out = super().forward(x)
        if self.probabilistic:
            return {
                k: rearrange(v, "b ... (m c) -> (m b) ... c", m=self.num_estimators)
                for k, v in out.items()
            }

        return rearrange(out, "b ... (m c) -> (m b) ... c", m=self.num_estimators)


def _mlp(
    stochastic: bool,
    in_features: int,
    num_outputs: int,
    hidden_dims: list[int],
    num_samples: int = 16,
    layer: type[nn.Module] = nn.Linear,
    layer_args: dict | None = None,
    activation: Callable = F.relu,
    dropout_rate: float = 0.0,
    dist_family: str | None = None,
    dist_args: dict | None = None,
    flatten_start_dim: int = -1,
) -> _MLP | StochasticModel:
    model = _MLP(
        in_features=in_features,
        num_outputs=num_outputs,
        hidden_dims=hidden_dims,
        layer_args=layer_args or {},
        layer=layer,
        activation=activation,
        dropout_rate=dropout_rate,
        dist_family=dist_family,
        dist_args=dist_args or {},
        flatten_start_dim=flatten_start_dim,
    )
    if stochastic:
        return StochasticModel(
            core_model=model, num_samples=num_samples, probabilistic=dist_family is not None
        )
    return model


def mlp(
    in_features: int,
    num_outputs: int,
    hidden_dims: list[int],
    activation: Callable = F.relu,
    dropout_rate: float = 0.0,
    dist_family: str | None = None,
    dist_args: dict | None = None,
    flatten_start_dim: int = -1,
) -> _MLP:
    """Multi-layer perceptron.

    Args:
        in_features (int): Number of input features.
        num_outputs (int): Number of output features.
        hidden_dims (list[int]): Number of features in each hidden layer.
        activation (Callable, optional): Activation function. Defaults to
            ``F.relu``.
        dropout_rate (float, optional): Dropout probability. Defaults to ``0.0``.
        dist_family (str, optional): Distribution family. Defaults to ``None``.
        dist_args (Dict, optional): Arguments for the distribution layer class. Defaults
            to ``None``.
        flatten_start_dim (int, optional): Dimension to start flattening the input.
            Defaults to ``-1``.

    Returns:
        _MLP: A Multi-Layer-Perceptron model.
    """
    return _mlp(
        stochastic=False,
        in_features=in_features,
        num_outputs=num_outputs,
        hidden_dims=hidden_dims,
        activation=activation,
        dropout_rate=dropout_rate,
        dist_family=dist_family,
        dist_args=dist_args,
        flatten_start_dim=flatten_start_dim,
    )


def packed_mlp(
    in_features: int,
    num_outputs: int,
    hidden_dims: list[int],
    num_estimators: int = 4,
    alpha: float = 2,
    gamma: int = 1,
    activation: Callable = F.relu,
    dropout_rate: float = 0.0,
    dist_family: str | None = None,
    dist_args: dict | None = None,
    flatten_start_dim: int = -1,
) -> _MLP:
    layer_args = {
        "num_estimators": num_estimators,
        "alpha": alpha,
        "gamma": gamma,
    }
    return _mlp(
        stochastic=False,
        in_features=in_features,
        num_outputs=num_outputs,
        hidden_dims=hidden_dims,
        layer=PackedLinear,
        activation=activation,
        layer_args=layer_args,
        dropout_rate=dropout_rate,
        dist_family=dist_family,
        dist_args=dist_args,
        flatten_start_dim=flatten_start_dim,
    )


def batched_mlp(
    in_features: int,
    num_outputs: int,
    hidden_dims: list[int],
    num_estimators: int = 4,
    activation: Callable = F.relu,
    dropout_rate: float = 0.0,
    dist_family: str | None = None,
    dist_args: dict | None = None,
    flatten_start_dim: int = -1,
) -> _MLP:
    layer_args = {
        "num_estimators": num_estimators,
    }
    return _mlp(
        stochastic=False,
        in_features=in_features,
        num_outputs=num_outputs,
        hidden_dims=hidden_dims,
        layer=BatchLinear,
        activation=activation,
        layer_args=layer_args,
        dropout_rate=dropout_rate,
        dist_family=dist_family,
        dist_args=dist_args,
        flatten_start_dim=flatten_start_dim,
    )


def bayesian_mlp(
    in_features: int,
    num_outputs: int,
    hidden_dims: list[int],
    num_samples: int = 16,
    activation: Callable = F.relu,
    dropout_rate: float = 0.0,
    dist_family: str | None = None,
    dist_args: dict | None = None,
    flatten_start_dim: int = -1,
) -> StochasticModel:
    return _mlp(
        stochastic=True,
        num_samples=num_samples,
        in_features=in_features,
        num_outputs=num_outputs,
        hidden_dims=hidden_dims,
        layer=BayesLinear,
        activation=activation,
        dropout_rate=dropout_rate,
        dist_family=dist_family,
        dist_args=dist_args,
        flatten_start_dim=flatten_start_dim,
    )


def mimo_mlp(
    in_features: int,
    num_outputs: int,
    hidden_dims: list[int],
    num_estimators: int,
    activation: Callable = F.relu,
    dropout_rate: float = 0.0,
    dist_family: str | None = None,
    dist_args: dict | None = None,
    flatten_start_dim: int = -1,
) -> _MIMOMLP:
    return _MIMOMLP(
        in_features=in_features,
        num_outputs=num_outputs,
        hidden_dims=hidden_dims,
        num_estimators=num_estimators,
        layer=nn.Linear,
        activation=activation,
        layer_args={},
        dropout_rate=dropout_rate,
        dist_family=dist_family,
        dist_args=dist_args or {},
        flatten_start_dim=flatten_start_dim,
    )
