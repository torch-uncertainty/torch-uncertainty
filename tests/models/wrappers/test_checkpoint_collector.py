import pytest
import torch

from tests._dummies.model import dummy_model
from torch_uncertainty.methods import CheckpointCollector


class TestCheckpointCollector:
    """Testing the CheckpointCollector class."""

    def test_training(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        ens = CheckpointCollector(dummy_model(1, 10), store_on_cpu=True)
        assert ens.mode == "all"
        ens.to(device)
        ens.eval()
        ens(torch.randn(1, 1, device=device))

        ens.train()
        ens(torch.randn(1, 1, device=device))
        ens.update_wrapper(0)
        ens.eval()
        ens(torch.randn(1, 1, device=device))

        ens = CheckpointCollector(dummy_model(1, 10), use_final_model=False)
        ens.train()
        ens(torch.randn(1, 1))
        ens.update_wrapper(0)
        ens.eval()
        ens(torch.randn(1, 1))

        ens = CheckpointCollector(dummy_model(1, 10), cycle_start=1, cycle_length=3)
        assert ens.mode == "cycle"
        ens.train()
        ens(torch.randn(1, 1))
        ens.update_wrapper(0)
        ens(torch.randn(1, 1))
        ens.update_wrapper(1)
        ens.eval()
        ens(torch.randn(1, 1))

        ens = CheckpointCollector(dummy_model(1, 10), save_schedule=[2, 5], store_on_cpu=True)
        assert ens.mode == "schedule"
        ens.to(device)
        ens.train()
        ens(torch.randn(1, 1, device=device))
        ens.update_wrapper(0)
        ens.eval()
        ens(torch.randn(1, 1, device=device))

    def test_failures(self) -> None:
        with pytest.raises(ValueError):
            CheckpointCollector(dummy_model(1, 10), cycle_start=0)

        with pytest.raises(ValueError):
            CheckpointCollector(dummy_model(1, 10), cycle_length=0)

        with pytest.raises(ValueError):
            CheckpointCollector(
                dummy_model(1, 10), cycle_start=2, cycle_length=1, save_schedule=[1, 2]
            )
