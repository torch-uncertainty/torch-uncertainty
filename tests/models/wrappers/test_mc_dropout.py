import pytest
import torch

from tests._dummies.model import dummy_model, dummy_segmentation_model
from torch_uncertainty.methods import mc_dropout


class TestMCDropout:
    """Testing the MC Dropout class."""

    def test_mc_dropout_train(self) -> None:
        # classification task
        model = dummy_model(10, 5, 0.1)
        dropout_model = mc_dropout(model, num_estimators=5)
        dropout_model.train()
        assert dropout_model.training
        dropout_model(torch.rand(1, 10))

        dropout_model = mc_dropout(model, num_estimators=5, last_layer=True)
        dropout_model.train()
        assert dropout_model.training
        dropout_model(torch.rand(1, 10))

        # regression task
        dropout_model = mc_dropout(model, num_estimators=5, task="regression", probabilistic=False)
        dropout_model.train()
        assert dropout_model.training
        dropout_model(torch.rand(1, 10))

        # segmentation task
        model = dummy_segmentation_model(10, 5, 3, 0.1)
        dropout_model = mc_dropout(model, num_estimators=5, task="segmentation")
        dropout_model.train()
        assert dropout_model.training
        dropout_model(torch.rand(1, 10, 3, 3))

        # pixel regression task
        dropout_model = mc_dropout(
            model, num_estimators=5, task="pixel_regression", probabilistic=False
        )
        dropout_model.train()
        assert dropout_model.training
        dropout_model(torch.rand(1, 10, 3, 3))

    def test_mc_dropout_eval_cls(self) -> None:
        model = dummy_model(10, 5, 0.1)
        dropout_model = mc_dropout(model, num_estimators=5)
        dropout_model.eval()
        assert not dropout_model.training
        assert dropout_model(torch.rand(1, 10)).size() == torch.Size([5, 5])

        dropout_model = mc_dropout(model, num_estimators=5, on_batch=False)
        dropout_model.eval()
        assert not dropout_model.training
        assert dropout_model(torch.rand(1, 10)).size() == torch.Size([5, 5])

    def test_mc_dropout_eval_reg(self) -> None:
        model = dummy_model(10, 5, 0.1)
        dropout_model = mc_dropout(model, num_estimators=5, task="regression", probabilistic=False)
        dropout_model.eval()
        assert not dropout_model.training
        assert dropout_model(torch.rand(1, 10)).size() == torch.Size([5, 5])

        dropout_model = mc_dropout(
            model, num_estimators=5, task="regression", on_batch=False, probabilistic=False
        )
        dropout_model.eval()
        assert not dropout_model.training
        assert dropout_model(torch.rand(1, 10)).size() == torch.Size([5, 5])

        model = dummy_model(10, 5, 0.1, dist_family="normal")
        dropout_model = mc_dropout(
            model, num_estimators=5, task="regression", on_batch=False, probabilistic=True
        )
        dropout_model.eval()
        assert not dropout_model.training
        out = dropout_model(torch.rand(1, 10))
        assert isinstance(out, dict)
        assert "loc" in out
        assert "scale" in out
        assert out["loc"].size() == torch.Size([5, 5])
        assert out["scale"].size() == torch.Size([5, 5])

    def test_mc_dropout_eval_seg(self) -> None:
        model = dummy_segmentation_model(10, 5, 3, 0.1)
        dropout_model = mc_dropout(model, num_estimators=5, task="segmentation")
        dropout_model.eval()
        assert not dropout_model.training
        assert dropout_model(torch.rand(1, 10, 3, 3)).size() == torch.Size([5, 5, 3, 3])

        dropout_model = mc_dropout(model, num_estimators=5, task="segmentation", on_batch=False)
        dropout_model.eval()
        assert not dropout_model.training
        assert dropout_model(torch.rand(1, 10, 3, 3)).size() == torch.Size([5, 5, 3, 3])

    def test_mc_dropout_eval_pixel_reg(self) -> None:
        model = dummy_segmentation_model(10, 5, 3, 0.1)
        dropout_model = mc_dropout(
            model, num_estimators=5, task="pixel_regression", probabilistic=False
        )
        dropout_model.eval()
        assert not dropout_model.training
        assert dropout_model(torch.rand(1, 10, 3, 3)).size() == torch.Size([5, 5, 3, 3])

        dropout_model = mc_dropout(
            model, num_estimators=5, task="pixel_regression", on_batch=False, probabilistic=False
        )
        dropout_model.eval()
        assert not dropout_model.training
        assert dropout_model(torch.rand(1, 10, 3, 3)).size() == torch.Size([5, 5, 3, 3])

        model = dummy_segmentation_model(10, 5, 3, 0.1, dist_family="normal")
        dropout_model = mc_dropout(
            model, num_estimators=5, task="pixel_regression", on_batch=False, probabilistic=True
        )
        dropout_model.eval()
        assert not dropout_model.training
        out = dropout_model(torch.rand(1, 10, 3, 3))
        assert isinstance(out, dict)
        assert "loc" in out
        assert "scale" in out
        assert out["loc"].size() == torch.Size([5, 5, 3, 3])
        assert out["scale"].size() == torch.Size([5, 5, 3, 3])

    def test_mc_dropout_errors(self) -> None:
        model = dummy_model(10, 5, 0.1)

        with pytest.raises(
            ValueError,
            match=r"Task invalid not supported. Supported tasks are: `classification`, `regression`, `segmentation`, `pixel_regression`.",
        ):
            mc_dropout(model, num_estimators=5, task="invalid")

        with pytest.raises(ValueError, match=r"`probabilistic` must be set for regression tasks."):
            mc_dropout(model, num_estimators=5, task="regression")

        with pytest.raises(ValueError, match=r"`num_estimators` must be strictly positive"):
            mc_dropout(core_model=model, num_estimators=-1, last_layer=True, on_batch=True)

        dropout_model = mc_dropout(model, 5)
        with pytest.raises(TypeError, match=r"Training mode is expected to be boolean"):
            dropout_model.train(mode=1)

        with pytest.raises(TypeError, match=r"Training mode is expected to be boolean"):
            dropout_model.train(mode=None)

        model = dummy_model(10, 5, 0.0)
        with pytest.raises(
            ValueError,
            match=r"At least one dropout module must have a dropout rate",
        ):
            dropout_model = mc_dropout(model, 5)

        model = dummy_model(10, 5, dropout_rate=0)
        del model.dropout
        with pytest.raises(ValueError):
            dropout_model = mc_dropout(model, 5)

        model = mc_dropout(
            dummy_model(10, 5, 0.1),
            num_estimators=5,
            task="regression",
            probabilistic=True,
            on_batch=False,
        )
        model.eval()
        with pytest.raises(
            ValueError,
            match=r"When `probabilistic=True`, the model must return a dictionary of distribution parameters.",
        ):
            model(torch.rand(1, 10))
