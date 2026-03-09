import pytest
import torch
from einops import repeat
from torch import nn
from torch.utils.data import DataLoader

from torch_uncertainty.post_processing import (
    Conformal,
    ConformalClsAPS,
    ConformalClsRAPS,
    ConformalClsTHR,
)


class TestConformal:
    """Testing the Conformal class."""

    def test_errors(self) -> None:
        Conformal.__abstractmethods__ = set()
        conformal = Conformal(
            model=None,
            alpha=0.1,
            ts_init_val=1,
            ts_lr=1,
            ts_max_iter=1,
            enable_ts=True,
            device="cpu",
        )
        assert conformal.model.model is None
        conformal.set_model(nn.Identity())
        assert isinstance(conformal.model.model, nn.Identity)
        conformal.fit(None)
        conformal.forward(None)
        conformal.conformal(None)


class TestConformalClsAPS:
    """Testing the ConformalClsRAPS class."""

    def test_fit(self) -> None:
        inputs = repeat(torch.tensor([0.6, 0.3, 0.1]), "c -> b c", b=10).log()
        labels = torch.tensor([0, 2] + [1] * 8)

        calibration_set = list(zip(inputs, labels, strict=True))
        dl = DataLoader(calibration_set, batch_size=10)

        conformal = ConformalClsAPS(alpha=0.1, model=nn.Identity(), randomized=False)
        conformal.fit(dl)
        out = conformal.conformal(inputs)
        assert out.shape == (10, 3)
        assert (
            out == repeat(torch.tensor([True, True, False]), "c -> b c", b=10).float() / 2
        ).all()

        conformal = ConformalClsAPS(
            alpha=0.1, model=nn.Identity(), randomized=True, enable_ts=False
        )
        conformal.fit(dl)
        out = conformal.conformal(inputs)
        assert out.shape == (10, 3)

    def test_failures(self) -> None:
        with pytest.raises(RuntimeError):
            _ = ConformalClsAPS(
                alpha=0.1,
            ).quantile


class TestConformalClsRAPS:
    """Testing the ConformalClsRAPS class."""

    def test_fit(self) -> None:
        inputs = repeat(torch.tensor([6.0, 4.0, 1.0]), "c -> b c", b=10)
        labels = torch.tensor([0, 2] + [1] * 8)

        calibration_set = list(zip(inputs, labels, strict=True))
        dl = DataLoader(calibration_set, batch_size=10)

        conformal = ConformalClsRAPS(alpha=0.1, model=nn.Identity(), randomized=False)
        conformal.set_model(nn.Identity())
        conformal.fit(dl)
        out = conformal.conformal(inputs)
        assert out.shape == (10, 3)
        assert (
            out == repeat(torch.tensor([True, True, False]), "c -> b c", b=10).float() / 2
        ).all()

        conformal = ConformalClsRAPS(
            alpha=0.1, model=nn.Identity(), randomized=True, enable_ts=False
        )
        conformal.fit(dl)
        out = conformal.conformal(inputs)
        assert out.shape == (10, 3)

    def test_failures(self) -> None:
        with pytest.raises(RuntimeError):
            ConformalClsRAPS(alpha=0.1).quantile  # noqa: B018

        with pytest.raises(ValueError, match=r"penalty should be non-negative. Got "):
            _ = ConformalClsRAPS(
                alpha=0.1,
                penalty=-0.1,
            )

        with pytest.raises(TypeError, match=r"regularization_rank should be an integer. Got"):
            _ = ConformalClsRAPS(
                alpha=0.1,
                regularization_rank=0.1,
            )

        with pytest.raises(ValueError, match=r"regularization_rank should be non-negative. Got "):
            _ = ConformalClsRAPS(
                alpha=0.1,
                regularization_rank=-1,
            )
        conformal = ConformalClsRAPS(alpha=0.1, model=nn.Identity(), randomized=True)
        with pytest.raises(
            RuntimeError, match=r"Cannot return temperature when enable_ts is False."
        ):
            _ = conformal.temperature


class TestConformalClsTHR:
    """Testing the ConformalClsTHR class."""

    def test_main(self) -> None:
        conformal = ConformalClsTHR(alpha=0.1, model=None, ts_init_val=2)
        assert conformal.temperature == 2.0
        conformal.set_model(nn.Identity())
        assert isinstance(conformal.model.model, nn.Identity)

    def test_fit(self) -> None:
        inputs = repeat(torch.tensor([0.6, 0.3, 0.1]), "c -> b c", b=10).log()
        labels = torch.tensor([0, 2] + [1] * 8)

        calibration_set = list(zip(inputs, labels, strict=True))
        dl = DataLoader(calibration_set, batch_size=10)

        conformal = ConformalClsTHR(
            alpha=0.1, model=nn.Identity(), ts_init_val=1, ts_lr=1, ts_max_iter=10, enable_ts=True
        )
        conformal.fit(dl)
        out = conformal.conformal(inputs)
        assert out.shape == (10, 3)
        assert (
            out == repeat(torch.tensor([True, True, False]), "c -> b c", b=10).float() / 2
        ).all()

        conformal = ConformalClsTHR(alpha=0.1, model=nn.Identity(), enable_ts=False)
        conformal.fit(dl)

    def test_failures(self) -> None:
        with pytest.raises(RuntimeError):
            _ = ConformalClsTHR(
                alpha=0.1,
            ).quantile
