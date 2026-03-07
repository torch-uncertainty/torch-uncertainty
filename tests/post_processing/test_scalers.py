import pytest
import torch
from torch import nn, softmax
from torch.utils.data import DataLoader

from torch_uncertainty.post_processing import (
    MatrixScaler,
    TemperatureScaler,
    VectorScaler,
)


class TestTemperatureScaler:
    """Testing the TemperatureScaler class."""

    def test_main(self) -> None:
        scaler = TemperatureScaler(model=nn.Identity(), init_val=2)
        scaler.set_temperature(1)

        logits = torch.tensor([[1, 2, 3]], dtype=torch.float32)

        assert scaler.temperature[0].item() == 1.0
        assert torch.all(scaler(logits) == logits)

    def test_fit_biased(self) -> None:
        inputs = torch.as_tensor([0.6, 0.4]).repeat(10, 1)
        labels = torch.as_tensor([0.5, 0.5]).repeat(10, 1)

        calibration_set = list(zip(inputs, labels, strict=True))
        dl = DataLoader(calibration_set, batch_size=10)

        scaler = TemperatureScaler(model=nn.Identity(), init_val=2, lr=1, max_iter=10)
        assert scaler.temperature[0] == 2.0
        scaler.fit(dl)
        assert scaler.temperature[0] > 10  # best is +inf
        assert (
            torch.sum(
                softmax(scaler(torch.as_tensor([[0.6, 0.4]])).detach(), dim=1)
                - torch.as_tensor([[0.5, 0.5]])
            )
            ** 2
            < 0.001
        )
        scaler.fit_predict(dl, progress=False)

        inputs = torch.as_tensor([0.6]).repeat(10, 1)
        labels = torch.as_tensor([0.5]).repeat(10)
        calibration_set = list(zip(inputs, labels, strict=True))
        dl = DataLoader(calibration_set, batch_size=10)
        scaler = TemperatureScaler(model=nn.Identity(), init_val=2, lr=1, max_iter=10)
        scaler.fit(dl)

        inputs = torch.as_tensor([0.6]).repeat(10, 1)
        labels = torch.as_tensor([1]).repeat(10)
        calibration_set = list(zip(inputs, labels, strict=True))
        dl = DataLoader(calibration_set, batch_size=10)
        scaler = TemperatureScaler(model=nn.Identity(), init_val=2, lr=1, max_iter=10)
        scaler.fit(dl)

    def test_errors(self) -> None:
        with pytest.raises(ValueError):
            TemperatureScaler(model=nn.Identity(), init_val=-1)

        with pytest.raises(ValueError):
            TemperatureScaler(model=nn.Identity(), lr=-1)

        with pytest.raises(ValueError, match=r"Max iterations must be strictly positive. Got "):
            TemperatureScaler(model=nn.Identity(), max_iter=-1)

        with pytest.raises(ValueError, match=r"Eps must be strictly positive. Got "):
            TemperatureScaler(model=nn.Identity(), eps=-1)

        scaler = TemperatureScaler(
            model=nn.Identity(),
        )
        with pytest.raises(ValueError):
            scaler.set_temperature(val=-1)


class TestVectorScaler:
    """Testing the VectorScaler class."""

    def test_main(self) -> None:
        scaler = VectorScaler(model=nn.Identity(), num_classes=1, init_val=2)
        scaler.set_temperature(1)

        logits = torch.tensor([[1, 2, 3]], dtype=torch.float32)

        assert scaler.temp_w.mean() == 1.0
        assert torch.all(scaler(logits) == logits)

        _ = scaler.temperature

    def test_fit_biased(self) -> None:
        inputs = torch.as_tensor([0.45, 0.45, 0.05, 0.05]).repeat(10, 1)
        labels = torch.as_tensor([1 / 4, 1 / 4, 1 / 4, 1 / 4]).repeat(10, 1)

        calibration_set = list(zip(inputs, labels, strict=True))
        dl = DataLoader(calibration_set, batch_size=10)

        scaler = VectorScaler(num_classes=4, model=nn.Identity(), init_val=1, lr=1, max_iter=10)
        scaler.fit(dl)
        assert scaler.temperature[0][:2].mean() < 1.0
        assert scaler.temperature[0][2:].mean() > 1.0

    def test_errors(self) -> None:
        with pytest.raises(ValueError):
            VectorScaler(model=nn.Identity(), num_classes=-1)

        with pytest.raises(TypeError):
            VectorScaler(model=nn.Identity(), num_classes=1.8)


class TestMatrixScaler:
    """Testing the MatrixScaler class."""

    def test_main(self) -> None:
        scaler = MatrixScaler(model=nn.Identity(), num_classes=2, init_w=2)
        scaler.set_temperature(1, 0)

        logits = torch.tensor([[[0, 1], [0, 2], [0, 3]]], dtype=torch.float32)

        assert scaler.temp_w.mean() == 1.0 / 2  # Fills only the diagonal with ones
        assert scaler.temp_b.mean() == 0.0
        assert torch.all(scaler(logits) == logits)

        _ = scaler.temperature

    def test_errors(self) -> None:
        with pytest.raises(ValueError):
            MatrixScaler(model=nn.Identity(), num_classes=-1)

        with pytest.raises(TypeError):
            MatrixScaler(model=nn.Identity(), num_classes=1.8)
