import pytest
import torch
from torch import nn, softmax
from torch.utils.data import DataLoader

from torch_uncertainty.post_processing import (
    BBQScaler,
    DirichletScaler,
    HistogramBinningScaler,
    IsotonicRegressionScaler,
    MatrixScaler,
    TemperatureScaler,
    VectorScaler,
)


class TestTemperatureScaler:
    """Testing the TemperatureScaler class."""

    def test_main(self) -> None:
        scaler = TemperatureScaler(model=nn.Identity(), init_temperature=2)
        scaler.set_temperature(1)

        logits = torch.tensor([[1, 2, 3]], dtype=torch.float32)

        assert scaler.temperature[0].item() == 1.0
        assert torch.all(scaler(logits) == logits)

    def test_fit_biased(self) -> None:
        inputs = torch.as_tensor([0.6, 0.4]).repeat(10, 1)
        labels = torch.as_tensor([0.5, 0.5]).repeat(10, 1)

        calibration_set = list(zip(inputs, labels, strict=True))
        dl = DataLoader(calibration_set, batch_size=10)

        scaler = TemperatureScaler(model=nn.Identity(), init_temperature=2, lr=1, max_iter=10)
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
        scaler = TemperatureScaler(model=nn.Identity(), init_temperature=2, lr=1, max_iter=10)
        scaler.fit(dl)

        inputs = torch.as_tensor([0.6]).repeat(10, 1)
        labels = torch.as_tensor([1]).repeat(10)
        calibration_set = list(zip(inputs, labels, strict=True))
        dl = DataLoader(calibration_set, batch_size=10)
        scaler = TemperatureScaler(model=nn.Identity(), init_temperature=2, lr=1, max_iter=10)
        scaler.fit(dl)

    def test_errors(self) -> None:
        with pytest.raises(ValueError):
            TemperatureScaler(model=nn.Identity(), init_temperature=-1)

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
        scaler = VectorScaler(model=nn.Identity(), num_classes=1, init_temperature=2)
        scaler.set_temperature(torch.tensor([1]))
        scaler.set_temperature(1)

        logits = torch.tensor([[1, 2, 3]], dtype=torch.float32)

        assert scaler.temperature[0].mean() == 1.0
        assert torch.all(scaler(logits) == logits)

        _ = scaler.temperature

    def test_fit_biased(self) -> None:
        inputs = torch.as_tensor([0.45, 0.45, 0.05, 0.05]).repeat(10, 1)
        labels = torch.as_tensor([1 / 4, 1 / 4, 1 / 4, 1 / 4]).repeat(10, 1)

        calibration_set = list(zip(inputs, labels, strict=True))
        dl = DataLoader(calibration_set, batch_size=10)

        scaler = VectorScaler(
            num_classes=4, model=nn.Identity(), init_temperature=1, lr=1, max_iter=10
        )
        scaler.fit(dl)
        assert scaler.temperature[0][:2].mean().item() > 1.0
        assert scaler.temperature[0][2:].mean().item() < 1.0

    def test_errors(self) -> None:
        with pytest.raises(
            ValueError, match=r"The number of classes must be strictly positive. Got "
        ):
            VectorScaler(num_classes=0)

        with pytest.raises(TypeError):
            VectorScaler(num_classes=1.8)

        with pytest.raises(ValueError, match=r"Temperature value must be strictly positive. Got "):
            VectorScaler(num_classes=2, init_temperature=0)

        with pytest.raises(ValueError, match=r"Temperature value must be strictly positive. Got "):
            VectorScaler(num_classes=2, init_temperature=torch.tensor([0]))

        vs = VectorScaler(num_classes=2)
        with pytest.raises(ValueError, match=r"val should be a float or a Tensor. Got "):
            vs.set_temperature([0, 0, 0])


class TestMatrixScaler:
    """Testing the MatrixScaler class."""

    def test_main(self) -> None:
        scaler = MatrixScaler(model=nn.Identity(), num_classes=2, init_weight_temperature=1)
        scaler.set_temperature(1, 1)
        scaler.set_temperature(1, None)

        logits = torch.tensor([[[0, 1], [0, 2], [0, 3]]], dtype=torch.float32)

        assert scaler.inv_temperature_weight.mean() == 1.0 / 2  # Fills only the diagonal with ones
        assert scaler.inv_temperature_bias.mean() == 0.0
        assert torch.all(scaler(logits) == logits)

        _ = scaler.temperature
        _ = scaler.inv_temperature


class TestDirichletScaler:
    """Testing the DirichletScaler class."""

    def test_main(self) -> None:
        scaler = DirichletScaler(model=nn.Identity(), num_classes=2)
        logits = torch.tensor([[[0, 1], [0, 2], [0, 3]]], dtype=torch.float32)

        assert scaler.inv_temperature_weight.mean() == 1.0 / 2  # Fills only the diagonal with ones
        assert scaler.inv_temperature_bias.mean() == 0.0
        assert torch.all(scaler(logits) == torch.log_softmax(logits, dim=1))

        _ = scaler.temperature

        inputs = torch.as_tensor([0.45, 0.45, 0.05, 0.05]).repeat(10, 1)
        labels = torch.as_tensor([1 / 4, 1 / 4, 1 / 4, 1 / 4]).repeat(10, 1)

        calibration_set = list(zip(inputs, labels, strict=True))
        dl = DataLoader(calibration_set, batch_size=10)

        scaler = DirichletScaler(num_classes=4, init_weight_temperature=1, lr=1, max_iter=10)
        scaler.fit(dl, save_logits=True)

        scaler = DirichletScaler(
            num_classes=4,
            model=nn.Identity(),
            init_weight_temperature=1,
            lr=1,
            max_iter=10,
            lambda_reg=1e-3,
            mu_reg=1e-3,
        )
        scaler.fit(dl)

    def test_errors(self) -> None:
        with pytest.raises(ValueError, match=r"lambda_reg must be None or positive. Got "):
            DirichletScaler(model=nn.Identity(), num_classes=2, lambda_reg=-1e8)
        with pytest.raises(ValueError, match=r"mu_reg must be None or positive. Got "):
            DirichletScaler(model=nn.Identity(), num_classes=2, mu_reg=-1e8)


@pytest.fixture
def binary_dataloader():
    """Returns a DataLoader for binary classification (1D logits)."""
    inputs = torch.tensor([1.5, -1.0, 2.0, -2.5]).repeat(10)
    labels = torch.tensor([1, 0, 1, 0]).repeat(10)
    dataset = list(zip(inputs, labels, strict=False))
    return DataLoader(dataset, batch_size=10)


@pytest.fixture
def multiclass_dataloader():
    """Returns a DataLoader for multiclass classification (2D logits)."""
    # 3 classes: Class 0, 1, and 2 dominant respectively
    inputs = torch.tensor([[2.0, 0.5, 0.1], [0.1, 2.0, 0.5], [0.5, 0.1, 2.0]]).repeat(10, 1)
    labels = torch.tensor([0, 1, 2]).repeat(10)
    dataset = list(zip(inputs, labels, strict=False))
    return DataLoader(dataset, batch_size=10)


class TestIsotonicRegressionScaler:
    """Testing the IsotonicRegressionScaler class."""

    def test_main(self) -> None:
        scaler = IsotonicRegressionScaler(model=nn.Identity())
        logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
        # Untrained check
        assert not scaler.trained
        assert torch.all(scaler(logits) == logits)

    def test_fit_binary(self, binary_dataloader) -> None:
        scaler = IsotonicRegressionScaler(model=nn.Identity())
        scaler.fit(binary_dataloader, progress=False)

        assert scaler.trained
        assert scaler.num_classes == 1

        # Test inference on a single batch from the fixture
        inputs, _ = next(iter(binary_dataloader))
        calib_logits = scaler(inputs)
        assert calib_logits.shape == inputs.shape
        assert not torch.isnan(calib_logits).any()

    def test_fit_multiclass(self, multiclass_dataloader) -> None:
        scaler = IsotonicRegressionScaler(model=nn.Identity())
        scaler.fit(multiclass_dataloader, progress=False)

        assert scaler.trained
        assert scaler.num_classes == 3

        inputs, _ = next(iter(multiclass_dataloader))
        calib_logits = scaler(inputs)

        # Ensure probabilities sum to 1
        calib_probs = torch.softmax(calib_logits, dim=-1)
        torch.testing.assert_close(calib_probs.sum(dim=-1), torch.ones(calib_probs.shape[0]))


class TestHistogramBinningScaler:
    """Testing the HistogramBinningScaler class."""

    def test_main(self) -> None:
        scaler = HistogramBinningScaler(model=nn.Identity(), num_bins=10)
        logits = torch.tensor([[1.0, 2.0, 3.0]])

        assert not scaler.trained
        assert scaler.num_bins == 10
        assert torch.all(scaler(logits) == logits)

    def test_invalid_bins(self) -> None:
        with pytest.raises(ValueError, match="Number of bins must be strictly positive"):
            HistogramBinningScaler(model=nn.Identity(), num_bins=0)

    def test_fit_binary(self, binary_dataloader) -> None:
        scaler = HistogramBinningScaler(model=nn.Identity(), num_bins=5)
        scaler.fit(binary_dataloader, progress=False)

        assert scaler.trained
        assert scaler.num_classes == 1
        assert scaler.bin_values.shape == (5,)

        # Test inference
        inputs, _ = next(iter(binary_dataloader))
        calib_logits = scaler(inputs)
        assert calib_logits.shape == inputs.shape
        assert not torch.isnan(calib_logits).any()

    def test_fit_multiclass(self, multiclass_dataloader) -> None:
        scaler = HistogramBinningScaler(model=nn.Identity(), num_bins=5)
        scaler.fit(multiclass_dataloader, progress=False)

        assert scaler.trained
        assert scaler.num_classes == 3
        assert scaler.bin_values.shape == (3, 5)  # (num_classes, num_bins)

        inputs, _ = next(iter(multiclass_dataloader))
        calib_logits = scaler(inputs)

        # Output should be stable and normalized probabilities
        calib_probs = torch.softmax(calib_logits, dim=-1)
        torch.testing.assert_close(calib_probs.sum(dim=-1), torch.ones(len(inputs)))


class TestBBQScaler:
    """Testing the BBQScaler class."""

    def test_main(self) -> None:
        scaler = BBQScaler(model=nn.Identity(), max_bins=10)
        logits = torch.tensor([[1.0, 2.0, 3.0]])

        assert not scaler.trained
        assert scaler.max_bins == 10
        assert torch.all(scaler(logits) == logits)

    def test_invalid_bins(self) -> None:
        with pytest.raises(ValueError, match="max_bins must be at least 2"):
            BBQScaler(model=nn.Identity(), max_bins=1)

    def test_fit_binary(self, binary_dataloader) -> None:
        scaler = BBQScaler(model=nn.Identity(), max_bins=5)
        scaler.fit(binary_dataloader, progress=False)

        assert scaler.trained
        assert scaler.num_classes == 1
        assert len(scaler.bbq_models) == 1

        # Test inference
        inputs, _ = next(iter(binary_dataloader))
        calib_logits = scaler(inputs)
        assert calib_logits.shape == inputs.shape
        assert not torch.isnan(calib_logits).any()

    def test_fit_multiclass(self, multiclass_dataloader) -> None:
        scaler = BBQScaler(model=nn.Identity(), max_bins=5)
        scaler.fit(multiclass_dataloader, progress=False)

        assert scaler.trained
        assert scaler.num_classes == 3

        inputs, _ = next(iter(multiclass_dataloader))
        calib_logits = scaler(inputs)

        # Output should be stable and normalized probabilities
        calib_probs = torch.softmax(calib_logits, dim=-1)
        torch.testing.assert_close(calib_probs.sum(dim=-1), torch.ones(len(inputs)))
        assert len(scaler.bbq_models) == 3  # OvR structure: one ensemble per class

        # Validate weights sum to ~1 for each class
        for c in range(scaler.num_classes):
            _, weights = scaler.bbq_models[c]
            torch.testing.assert_close(weights.sum(), torch.tensor(1.0, device=scaler.device))

        # Test valid normalized output
        inputs, _ = next(iter(multiclass_dataloader))
        calib_logits = scaler(inputs)
        calib_probs = torch.softmax(calib_logits, dim=-1)

        torch.testing.assert_close(calib_probs.sum(dim=-1), torch.ones(len(inputs)))

    def test_degenerate_data(self) -> None:
        """Test BBQ behavior when input logits are entirely identical."""
        scaler = BBQScaler(model=nn.Identity(), max_bins=5)

        # Dataset where the model predicts the exact same logit for everything
        inputs = torch.tensor([[1.0, 0.5]]).repeat(20, 1)
        labels = torch.cat([torch.ones(10), torch.zeros(10)]).long()
        dataset = list(zip(inputs, labels, strict=True))
        degenerate_loader = torch.utils.data.DataLoader(dataset, batch_size=20)

        scaler.fit(degenerate_loader, progress=False)
        assert scaler.trained

        # Should gracefully fall back to a single bin model
        inputs, _ = next(iter(degenerate_loader))
        calib_logits = scaler(inputs)
        assert not torch.isnan(calib_logits).any()
