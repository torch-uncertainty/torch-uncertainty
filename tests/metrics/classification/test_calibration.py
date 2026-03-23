from typing import Literal

import matplotlib.pyplot as plt
import pytest
import torch
from torch import Tensor

from torch_uncertainty.metrics.classification import (
    AdaptiveCalibrationError,
    CalibrationError,
    ClasswiseCalibrationError,
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
        val = sce_logit._compute_smooth_ece(conf, acc, bandwidth=0.1)
        assert val < 1e-4

    def test_total_miscalibration(self, sce_logit: SmoothCalibrationError) -> None:
        """If model is 100% confident but 0% accurate, SmECE should be high."""
        conf = torch.full((100,), 1.0)
        acc = torch.full((100,), 0.0)

        val = sce_logit._compute_smooth_ece(conf, acc, bandwidth=0.1)
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
    def test_different_kernels(self, kernel: Literal["logit", "reflected"]) -> None:
        metric = SmoothCalibrationError(kernel_type=kernel, bandwidth=0.1)
        preds = torch.tensor([0.7, 0.8])
        target = torch.tensor([1, 1])
        metric.update(preds, target)
        assert metric.compute() >= 0

    def test_errors(self) -> None:
        with pytest.raises(ValueError, match=r"kernel_type must be 'logit' or 'reflected'. Got"):
            SmoothCalibrationError(kernel_type="gaussian")
        with pytest.raises(ValueError, match=r"Invalid bandwidth: "):
            SmoothCalibrationError(bandwidth="automatic")


class TestClasswiseCalibrationError:
    def test_init_valid(self):
        """Test initialization with valid arguments."""
        metric = ClasswiseCalibrationError(num_classes=3, num_bins=10, norm="l2", reduction="sum")
        assert metric.num_classes == 3
        assert metric.num_bins == 10
        assert metric.norm == "l2"
        assert metric.reduction == "sum"

    def test_init_invalid_reduction(self):
        """Test that invalid reduction raises ValueError."""
        with pytest.raises(ValueError, match="Expected argument `reduction`"):
            ClasswiseCalibrationError(num_classes=2, reduction="invalid")

    def test_init_invalid_norm(self):
        """Test that invalid norm raises ValueError."""
        with pytest.raises(ValueError, match="Expected argument `norm`"):
            ClasswiseCalibrationError(num_classes=2, norm="l3")

    def test_update_invalid_shape(self):
        """Test that non-2D probs raise ValueError."""
        metric = ClasswiseCalibrationError(num_classes=3)
        probs = torch.randn(2, 2, 3)  # 3D should fail now
        target = torch.tensor([0, 1])
        with pytest.raises(ValueError, match="Expected `probs` to be of shape"):
            metric.update(probs, target)

    def test_compute_none_reduction(self):
        """Test compute with reduction=None returns per-class scores."""
        num_classes = 3
        metric = ClasswiseCalibrationError(num_classes=num_classes, reduction=None)

        # Perfect calibration for class 0, poor for others
        probs = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        target = torch.tensor([0, 1])

        metric.update(probs, target)
        res = metric.compute()

        assert res.shape == (num_classes,)
        assert isinstance(res, torch.Tensor)

    def test_compute_mean_reduction(self):
        """Test compute with reduction='mean'."""
        metric = ClasswiseCalibrationError(num_classes=2, reduction="mean")
        probs = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
        target = torch.tensor([0, 1])

        metric.update(probs, target)
        res = metric.compute()

        assert res.ndim == 0  # Scalar
        assert res >= 0

    def test_compute_sum_reduction(self):
        """Test compute with reduction='sum'."""
        metric = ClasswiseCalibrationError(num_classes=2, reduction="sum")
        probs = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        target = torch.tensor([0, 1])

        metric.update(probs, target)
        res_sum = metric.compute()

        # Compare with none reduction to ensure sum is correct
        metric_none = ClasswiseCalibrationError(num_classes=2, reduction=None)
        metric_none.update(probs, target)
        res_none = metric_none.compute()

        torch.testing.assert_close(res_sum, res_none.sum())

    def test_update_one_hot_targets(self):
        """Test update with one-hot encoded targets (2D)."""
        num_classes = 3
        metric = ClasswiseCalibrationError(num_classes=num_classes)
        probs = torch.softmax(torch.randn(4, num_classes), dim=1)
        target = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]]).float()

        # Should not raise error
        metric.update(probs, target)
        res = metric.compute()
        assert not torch.isnan(res)

    def test_multiple_updates(self):
        """Test that states are correctly concatenated across updates."""
        metric = ClasswiseCalibrationError(num_classes=2, reduction=None)

        # Update 1
        metric.update(torch.tensor([[0.9, 0.1]]), torch.tensor([0]))
        # Update 2
        metric.update(torch.tensor([[0.2, 0.8]]), torch.tensor([1]))

        res = metric.compute()
        assert res.shape == (2,)
