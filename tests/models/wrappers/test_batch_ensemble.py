import pytest
import torch
from torch import nn

from torch_uncertainty.layers import BatchConv2d, BatchLinear
from torch_uncertainty.methods import BatchEnsemble, batch_ensemble


@pytest.fixture
def img_input() -> torch.Tensor:
    return torch.rand((5, 6, 3, 3))


# Define a simple model for testing wrapper functionality (disregarding the actual BatchEnsemble architecture)
class _DummyModel(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, 3)
        self.fc = nn.Linear(out_features, out_features)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)


class _DummyBEModel(nn.Module):
    def __init__(self, in_features, out_features, num_estimators) -> None:
        super().__init__()
        self.conv = BatchConv2d(in_features, out_features, 3, num_estimators)
        self.fc = BatchLinear(out_features, out_features, num_estimators=num_estimators)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)


class TestBatchEnsembleModel:
    def test_convert_layers(self) -> None:
        in_features = 6
        out_features = 4
        num_estimators = 3

        model = _DummyModel(in_features, out_features)
        wrapped_model = batch_ensemble(model, num_estimators, convert_layers=True)
        assert wrapped_model.num_estimators == num_estimators
        assert isinstance(wrapped_model.core_model.conv, BatchConv2d)
        assert isinstance(wrapped_model.core_model.fc, BatchLinear)

    def test_forward_pass(self, img_input) -> None:
        batch_size = img_input.size(0)
        in_features = img_input.size(1)
        out_features = 4
        num_estimators = 3
        model = _DummyBEModel(in_features, out_features, num_estimators)
        # with repeat_training_inputs=False
        wrapped_model = BatchEnsemble(model, num_estimators, repeat_training_inputs=False)
        # test forward pass for training
        logits = wrapped_model(img_input)
        assert logits.shape == (img_input.size(0), out_features)
        # test forward pass for evaluation
        wrapped_model.eval()
        logits = wrapped_model(img_input)
        assert logits.shape == (batch_size * num_estimators, out_features)
        # with repeat_training_inputs=True
        wrapped_model = BatchEnsemble(model, num_estimators, repeat_training_inputs=True)
        # test forward pass for training
        logits = wrapped_model(img_input)
        assert logits.shape == (batch_size * num_estimators, out_features)
        # test forward pass for evaluation
        wrapped_model.eval()
        logits = wrapped_model(img_input)
        assert logits.shape == (batch_size * num_estimators, out_features)

    def test_errors(self) -> None:
        with pytest.raises(ValueError):
            BatchEnsemble(_DummyBEModel(10, 5, 1), 0)
        with pytest.raises(ValueError):
            BatchEnsemble(_DummyModel(10, 5), 1)
        with pytest.raises(ValueError):
            BatchEnsemble(nn.Identity(), 2, convert_layers=True)
