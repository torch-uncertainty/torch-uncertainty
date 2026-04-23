import pytest
import torch
from torch import nn

from tests._dummies.model import dummy_model
from torch_uncertainty.methods import EMA


class TestEMA:
    """Testing the EMA class."""

    def test_training(self) -> None:
        ema = EMA(dummy_model(1, 10), momentum=0.99)
        ema.eval()
        ema(torch.randn(1, 1))
        ema.train()
        ema.update_wrapper(0)

    def test_failures(self) -> None:
        with pytest.raises(ValueError, match=r"must be in"):
            EMA(nn.Module(), momentum=-1)
