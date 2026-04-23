import torch
from torch import nn

from torch_uncertainty.layers import BayesConv2d, BayesLinear, NormalLinear
from torch_uncertainty.methods import StochasticModel


class DummyModelLinear(nn.Module):
    """Dummy model for testing purposes."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = BayesLinear(1, 10, 1)

    def forward(self, x):
        return self.layer(x)


class DummyModelLinearProb(nn.Module):
    """Dummy model for testing purposes with probabilistic output."""

    def __init__(self) -> None:
        super().__init__()

        self.layer = NormalLinear(
            BayesLinear,
            10,
            in_features=1,
            prior_sigma_1=1,
        )

    def forward(self, x):
        return self.layer(x)


class DummyModelConv(nn.Module):
    """Dummy conv model for testing purposes."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = BayesConv2d(1, 10, 1)

    def forward(self, x):
        return self.layer(x)


class DummyModelMix(nn.Module):
    """Dummy mix model for testing purposes."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = BayesConv2d(1, 10, 1, bias=False)
        self.relu = nn.ReLU()
        self.layer2 = nn.Conv2d(10, 1, 1)

    def forward(self, x):
        y = self.relu(self.layer(x))
        return self.layer2(y)


class TestStochasticModel:
    """Testing the StochasticModel decorator."""

    def test_main(self) -> None:
        model = StochasticModel(DummyModelLinear(), 2)
        model.freeze()
        model(torch.randn(1, 1))
        assert model.core_model.layer.frozen
        model.unfreeze()
        assert not model.core_model.layer.frozen
        model.eval()
        out = model(torch.randn(1, 1))
        assert out.shape == (2, 10)

        model = StochasticModel(DummyModelConv(), 2)
        model.freeze()
        assert model.core_model.layer.frozen
        model.unfreeze()
        assert not model.core_model.layer.frozen

    def test_probabilistic(self) -> None:
        model = StochasticModel(DummyModelLinearProb(), 2, probabilistic=True)
        model.freeze()
        model(torch.randn(1, 1))
        assert model.core_model.layer.base_layer.frozen
        model.unfreeze()
        assert not model.core_model.layer.base_layer.frozen
        model.eval()
        out = model(torch.randn(1, 1))
        assert isinstance(out, dict)
        assert "loc" in out
        assert "scale" in out
        assert out["loc"].shape == (2, 10)
        assert out["scale"].shape == (2, 10)

    def test_mix(self) -> None:
        model = StochasticModel(DummyModelMix(), 2)
        model.freeze()
        assert model.core_model.layer.frozen
        model.unfreeze()
        assert not model.core_model.layer.frozen

        state = model.sample()[0]
        keys = state.keys()
        print(list(keys))  # noqa: T201
        assert list(keys) == [
            "layer.weight",
            "layer2.weight",
            "layer2.bias",
        ]
