import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tests._dummies.model import dummy_model
from torch_uncertainty.methods import SWA, SWAG


class TestSWA:
    """Testing the SWA class."""

    def test_training(self) -> None:
        dl = DataLoader(TensorDataset(torch.randn(1, 1)), batch_size=1)
        swa = SWA(dummy_model(1, 10), cycle_start=1, cycle_length=1)
        swa.eval()
        swa(torch.randn(1, 1))

        swa.train()
        swa(torch.randn(1, 1))
        swa.update_wrapper(0)
        swa.bn_update(dl, "cpu")

        swa.update_wrapper(1)
        swa.bn_update(dl, "cpu")

        swa.eval()
        swa(torch.randn(1, 1))

    def test_failures(self) -> None:
        with pytest.raises(ValueError, match=r"`cycle_start` must be non-negative."):
            SWA(nn.Module(), cycle_start=-1, cycle_length=1)
        with pytest.raises(ValueError, match=r"`cycle_length` must be strictly positive."):
            SWA(nn.Module(), cycle_start=1, cycle_length=0)


class TestSWAG:
    """Testing the SWAG class."""

    def test_training(self) -> None:
        dl = DataLoader(TensorDataset(torch.randn(1, 1)), batch_size=1)
        swag = SWAG(
            dummy_model(1, 10),
            cycle_start=1,
            cycle_length=1,
            max_num_models=3,
            num_estimators=2,
        )
        assert swag.num_avgd_models == 0
        swag.eval()
        swag(torch.randn(1, 1))

        swag.train()
        swag(torch.randn(1, 1))
        swag.update_wrapper(0)
        assert swag.swag_stats["model.swag_stats.linear.weight_covariance_sqrt"].shape == (0, 10)
        swag.bn_update(dl, "cpu")
        swag(torch.randn(1, 1))

        swag.update_wrapper(1)
        assert swag.swag_stats["model.swag_stats.linear.weight_covariance_sqrt"].shape == (0, 10)
        assert swag.num_avgd_models == 0
        swag.bn_update(dl, "cpu")

        swag.update_wrapper(2)
        assert swag.swag_stats["model.swag_stats.linear.weight_covariance_sqrt"].shape == (1, 10)
        swag.bn_update(dl, "cpu")
        swag(torch.randn(1, 1))
        swag.update_wrapper(3)
        assert swag.swag_stats["model.swag_stats.linear.weight_covariance_sqrt"].shape == (2, 10)
        swag.update_wrapper(4)
        assert swag.num_avgd_models == 3
        assert swag.swag_stats["model.swag_stats.linear.weight_covariance_sqrt"].shape == (3, 10)
        swag.update_wrapper(5)
        assert swag.num_avgd_models == 4
        assert swag.swag_stats["model.swag_stats.linear.weight_covariance_sqrt"].shape == (3, 10)
        swag.eval()
        swag(torch.randn(1, 1))

        swag = SWAG(
            dummy_model(1, 10),
            cycle_start=1,
            cycle_length=1,
            diag_covariance=True,
        )
        swag.train()
        swag.update_wrapper(2)
        swag.sample(1, True, False, seed=1)

    def test_state_dict(self) -> None:
        mod = dummy_model(1, 10)
        swag = SWAG(mod, cycle_start=1, cycle_length=1, num_estimators=3)
        print(swag.state_dict())  # noqa: T201
        swag.load_state_dict(swag.state_dict())

    def test_failures(self) -> None:
        swag = SWAG(nn.Module(), scale=1, cycle_start=1, cycle_length=1)
        with pytest.raises(NotImplementedError, match=r"Raise an issue if you need this feature"):
            swag.sample(scale=1, block=True)
        with pytest.raises(ValueError, match=r"`scale` must be non-negative."):
            SWAG(nn.Module(), scale=-1, cycle_start=1, cycle_length=1)
        with pytest.raises(ValueError, match=r"`max_num_models` must be non-negative."):
            SWAG(nn.Module(), max_num_models=-1, cycle_start=1, cycle_length=1)
        with pytest.raises(ValueError, match=r"`var_clamp` must be non-negative. "):
            SWAG(nn.Module(), var_clamp=-1, cycle_start=1, cycle_length=1)
        swag = SWAG(nn.Module(), cycle_start=1, cycle_length=1, diag_covariance=True)
        with pytest.raises(
            ValueError,
            match=r"Cannot sample full rank from diagonal covariance matrix.",
        ):
            swag.sample(scale=1, diag_covariance=False)
