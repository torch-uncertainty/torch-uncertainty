import matplotlib.pyplot as plt
import pytest
import torch
from torch import Tensor

from torch_uncertainty.metrics.classification import (
    AdaptiveCalibrationError,
    CalibrationError,
    SmoothCalibrationError,
)


class TestCalibrationError:
    """Testing the CalibrationError metric class."""

    def test_plot_binary(self) -> None:

        metric = CalibrationError(task="binary", num_bins=2, norm="l1")
        metric.update(
            torch.as_tensor([0.25, 0.25, 0.55, 0.75, 0.75]),
            torch.as_tensor([0, 0, 1, 1, 1]),
        )
        fig, ax = metric.plot()
        assert isinstance(fig, plt.Figure)
        assert ax[0].get_xlabel() == "Top-class Confidence (%)"
        assert ax[0].get_ylabel() == "Success Rate (%)"
        assert ax[1].get_xlabel() == "Top-class Confidence (%)"
        assert ax[1].get_ylabel() == "Density (%)"

        plt.close(fig)

    def test_plot_multiclass(
        self,
    ) -> None:
        metric = CalibrationError(task="multiclass", num_bins=3, norm="l1", num_classes=3)
        metric.update(
            torch.as_tensor(
                [
                    [0.25, 0.20, 0.55],
                    [0.55, 0.05, 0.40],
                    [0.10, 0.30, 0.60],
                    [0.90, 0.05, 0.05],
                ]
            ),
            torch.as_tensor([0, 1, 2, 0]),
        )
        fig, ax = metric.plot()
        assert isinstance(fig, plt.Figure)
        assert ax[0].get_xlabel() == "Top-class Confidence (%)"
        assert ax[0].get_ylabel() == "Success Rate (%)"
        assert ax[1].get_xlabel() == "Top-class Confidence (%)"
        assert ax[1].get_ylabel() == "Density (%)"
        plt.close(fig)

    def test_errors(self) -> None:
        with pytest.raises(TypeError, match=r"is expected to be `int`"):
            CalibrationError(task="multiclass", num_classes=None)
        with pytest.raises(
            ValueError, match=r"`n_bins` does not exist in TorchUncertainty, use `num_bins`."
        ):
            CalibrationError(task="multiclass", num_classes=2, n_bins=1)


class TestAdaptiveCalibrationError:
    """Testing the AdaptiveCalibrationError metric class."""

    def test_main(self) -> None:
        ace = AdaptiveCalibrationError(task="binary", num_bins=2, norm="l1", validate_args=True)

        ace = AdaptiveCalibrationError(task="binary", num_bins=2, norm="l1", validate_args=False)
        ece = CalibrationError(task="binary", num_bins=2, norm="l1")
        ace.update(
            torch.as_tensor([0.35, 0.35, 0.75, 0.75]),
            torch.as_tensor([0, 0, 1, 1]),
        )
        ece.update(
            torch.as_tensor([0.35, 0.35, 0.75, 0.75]),
            torch.as_tensor([0, 0, 1, 1]),
        )
        assert ace.compute().item() == ece.compute().item()

        ace.reset()
        ace.update(
            torch.as_tensor([0.3, 0.24, 0.25, 0.2, 0.8]),
            torch.as_tensor([0, 0, 0, 1, 1]),
        )
        assert ace.compute().item() == pytest.approx(
            3 / 5 * (1 - 1 / 3 * (0.7 + 0.76 + 0.75)) + 2 / 5 * (0.8 - 0.5)
        )

        ace = AdaptiveCalibrationError(
            task="multiclass",
            num_classes=2,
            num_bins=2,
            norm="l2",
            validate_args=True,
        )
        ace.update(
            torch.as_tensor([[0.7, 0.3], [0.76, 0.24], [0.75, 0.25], [0.2, 0.8], [0.8, 0.2]]),
            torch.as_tensor([0, 0, 0, 1, 1]),
        )
        assert ace.compute().item() ** 2 == pytest.approx(
            3 / 5 * (1 - 1 / 3 * (0.7 + 0.76 + 0.75)) ** 2 + 2 / 5 * (0.8 - 0.5) ** 2
        )

        ace = AdaptiveCalibrationError(
            task="multiclass",
            num_classes=2,
            num_bins=2,
            norm="max",
            validate_args=False,
        )
        ace.update(
            torch.as_tensor([[0.7, 0.3], [0.76, 0.24], [0.75, 0.25], [0.2, 0.8], [0.8, 0.2]]),
            torch.as_tensor([0, 0, 0, 1, 1]),
        )
        assert ace.compute().item() == pytest.approx(0.8 - 0.5)

        ace = AdaptiveCalibrationError(task="binary", num_bins=3, norm="l2")
        ece = CalibrationError(task="binary", num_bins=3, norm="l2")

        ace.update(
            torch.as_tensor([0.12, 0.26, 0.70, 0.71, 0.91, 0.92]),
            torch.as_tensor([0, 1, 0, 0, 1, 1]),
        )
        ece.update(
            torch.as_tensor([0.12, 0.26, 0.70, 0.71, 0.91, 0.92]),
            torch.as_tensor([0, 1, 0, 0, 1, 1]),
        )
        assert ace.compute().item() > ece.compute().item()

    def test_errors(self) -> None:
        with pytest.raises(TypeError, match=r"is expected to be `int`"):
            AdaptiveCalibrationError(task="multiclass", num_classes=None)


@pytest.fixture
def sce_logit() -> SmoothCalibrationError:
    return SmoothCalibrationError(kernel_type="logit", bandwidth=0.1)


@pytest.fixture
def sce_reflected() -> SmoothCalibrationError:
    return SmoothCalibrationError(kernel_type="reflected", bandwidth=0.1)


class TestSmoothCalibrationError:
    """Testing the SmoothCalibrationError metric class."""

    def test_perfect_calibration(self, sce_logit: SmoothCalibrationError) -> None:
        """If accuracy perfectly matches confidence, SmECE should be near zero."""
        # 100 samples where confidence is 0.8 and they are all 'correct'
        # (Note: In a perfectly calibrated model, 80% should be correct,
        # but for a local window, if acc matches conf, error is 0).
        conf = torch.full((100,), 0.8)
        acc = torch.full((100,), 0.8)

        # We bypass update() logic to test the core smoother
        val = sce_logit._compute_smooth_ece(conf, acc, h=0.1)
        assert val < 1e-4

    def test_total_miscalibration(self, sce_logit: SmoothCalibrationError) -> None:
        """If model is 100% confident but 0% accurate, SmECE should be high."""
        conf = torch.full((100,), 1.0)
        acc = torch.full((100,), 0.0)

        val = sce_logit._compute_smooth_ece(conf, acc, h=0.1)
        # The error |1.0 - 0.0| = 1.0
        assert torch.isclose(val, torch.tensor(1.0), atol=1e-2)

    def test_binary_input_shapes(self) -> None:
        """Test that (N,) and (N, 1) inputs yield the same result."""
        metric = SmoothCalibrationError(kernel_type="reflected", bandwidth=0.1)

        preds = torch.tensor([0.9, 0.1, 0.8, 0.2])
        target = torch.tensor([1, 0, 1, 1])

        metric.update(preds, target)
        res1 = metric.compute()

        metric.reset()
        metric.update(preds.unsqueeze(1), target)
        res2 = metric.compute()

        assert torch.isclose(res1, res2)

    def test_multiclass_input(self) -> None:
        """Test standard multiclass (N, C) logits."""
        metric = SmoothCalibrationError(kernel_type="logit", bandwidth="auto")

        # High confidence on class 1, but target is class 0
        preds = torch.tensor([[-5.0, 5.0], [5.0, -5.0]])
        target = torch.tensor([0, 1])

        metric.update(preds, target)
        ece = metric.compute()

        assert ece > 0.5  # Should be very high error
        assert isinstance(ece, Tensor)

    def test_auto_bandwidth_search(self) -> None:
        """Ensure 'auto' bandwidth selection doesn't crash and returns a valid value."""
        metric = SmoothCalibrationError(kernel_type="logit", bandwidth="auto")

        preds = torch.rand(100, 5)
        target = torch.randint(0, 5, (100,))

        metric.update(preds, target)
        ece = metric.compute()

        assert 0.0 <= ece <= 1.0

    @pytest.mark.parametrize("kernel", ["logit", "reflected"])
    def test_different_kernels(self, kernel: str) -> None:
        metric = SmoothCalibrationError(kernel_type=kernel, bandwidth=0.1)
        preds = torch.tensor([0.7, 0.8])
        target = torch.tensor([1, 1])
        metric.update(preds, target)
        assert metric.compute() >= 0
