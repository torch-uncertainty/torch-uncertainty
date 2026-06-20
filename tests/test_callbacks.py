from unittest.mock import MagicMock

import lightning.pytorch.callbacks.model_checkpoint as _mc
import torch

from torch_uncertainty.callbacks import TUClsCheckpoint, TURegCheckpoint, TUSegCheckpoint
from torch_uncertainty.callbacks.compound_checkpoint import CompoundCheckpoint


def _make_trainer_mock():
    return MagicMock()


class TestTUClsCheckpoint:
    def test_init_default(self) -> None:
        ckpt = TUClsCheckpoint()
        assert "acc" in ckpt.callbacks
        assert "brier" in ckpt.callbacks
        assert "nll" in ckpt.callbacks

    def test_init_save_last(self) -> None:
        TUClsCheckpoint(save_last=True)

    def test_best_model_path(self) -> None:
        ckpt = TUClsCheckpoint()
        ckpt.best_model_path  # noqa: B018

    def test_state_dict_and_load(self) -> None:
        ckpt = TUClsCheckpoint()
        sd = ckpt.state_dict()
        assert set(sd.keys()) == {"acc", "brier", "nll"}
        ckpt.load_state_dict(sd)

    def test_trainer_hooks(self) -> None:
        trainer, module = _make_trainer_mock(), MagicMock()
        ckpt = TUClsCheckpoint()
        ckpt.setup(trainer, module, "fit")
        ckpt.on_train_start(trainer, module)
        ckpt.on_train_batch_end(trainer, module, {}, None, 0)
        ckpt.on_train_epoch_end(trainer, module)
        ckpt.on_validation_epoch_end(trainer, module)


class TestTUSegCheckpoint:
    def test_init_default(self) -> None:
        ckpt = TUSegCheckpoint()
        assert "miou" in ckpt.callbacks
        assert "brier" in ckpt.callbacks
        assert "nll" in ckpt.callbacks

    def test_best_model_path(self) -> None:
        ckpt = TUSegCheckpoint()
        ckpt.best_model_path  # noqa: B018

    def test_state_dict(self) -> None:
        ckpt = TUSegCheckpoint()
        sd = ckpt.state_dict()
        assert set(sd.keys()) == {"miou", "brier", "nll"}


class TestTURegCheckpoint:
    def test_init_non_probabilistic(self) -> None:
        ckpt = TURegCheckpoint(probabilistic=False)
        assert "mse" in ckpt.callbacks
        assert "nll" not in ckpt.callbacks

    def test_init_probabilistic(self) -> None:
        ckpt = TURegCheckpoint(probabilistic=True)
        assert "mse" in ckpt.callbacks
        assert "nll" in ckpt.callbacks
        assert "qce" in ckpt.callbacks

    def test_best_model_path(self) -> None:
        ckpt = TURegCheckpoint()
        ckpt.best_model_path  # noqa: B018

    def test_state_dict(self) -> None:
        ckpt = TURegCheckpoint()
        sd = ckpt.state_dict()
        assert "mse" in sd


class TestCompoundCheckpoint:
    def test_init(self) -> None:
        CompoundCheckpoint(
            compound_metric_dict={"val/loss": 1.0, "val/acc": -1.0},
            mode="min",
        )

    def test_init_with_options(self) -> None:
        CompoundCheckpoint(
            compound_metric_dict={"val/loss": 0.5},
            save_last=True,
            save_top_k=3,
            mode="max",
        )

    def test_monitor_candidates(self) -> None:
        metrics = {"val/loss": torch.tensor(0.5), "val/acc": torch.tensor(0.9)}
        orig = _mc.ModelCheckpoint._monitor_candidates
        _mc.ModelCheckpoint._monitor_candidates = lambda _, __: metrics
        try:
            ckpt = CompoundCheckpoint(compound_metric_dict={"val/loss": 1.0, "val/acc": -1.0})
            result = ckpt._monitor_candidates(_make_trainer_mock())
            assert "compound_metric" in result
        finally:
            _mc.ModelCheckpoint._monitor_candidates = orig
