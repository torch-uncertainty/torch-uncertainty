import pytest
import torch

from torch_uncertainty.losses.quantile import PinballLoss, RQRLoss


class TestPinballLoss:
    """Testing the PinballLoss class."""

    def test_main(self) -> None:
        # At τ=0.5 and perfect prediction the loss is zero.
        loss = PinballLoss(quantile=0.5)
        pred = torch.tensor([1.0])
        target = torch.tensor([1.0])
        assert loss(pred, target) == pytest.approx(0.0)

        # tau 0.5 is MAE/2: underestimate and overestimate are symmetric.
        pred_low = torch.tensor([0.0])
        pred_high = torch.tensor([1.0])
        target_one = torch.tensor([1.0])
        target_zero = torch.tensor([0.0])
        assert loss(pred_low, target_one) == pytest.approx(0.5)
        assert loss(pred_high, target_zero) == pytest.approx(0.5)

        # tau 0.9 penalises underestimation more heavily than overestimation.
        loss_q90 = PinballLoss(quantile=0.9)
        assert loss_q90(pred_low, target_one) == pytest.approx(0.9)
        assert loss_q90(pred_high, target_zero) == pytest.approx(0.1)

        # reduction "sum"
        loss_sum = PinballLoss(quantile=0.5, reduction="sum")
        preds = torch.tensor([0.0, 1.0])
        targets = torch.tensor([1.0, 0.0])
        assert loss_sum(preds, targets) == pytest.approx(1.0)

        # reduction "none"
        loss_none = PinballLoss(quantile=0.5, reduction="none")
        result = loss_none(preds, targets)
        assert result.tolist() == pytest.approx([0.5, 0.5])

    def test_failures(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"The quantile parameter should be in \(0, 1\)",
        ):
            PinballLoss(quantile=0.0)

        with pytest.raises(
            ValueError,
            match=r"The quantile parameter should be in \(0, 1\)",
        ):
            PinballLoss(quantile=1.0)

        with pytest.raises(ValueError, match=r"is not a valid value for reduction."):
            PinballLoss(quantile=0.5, reduction="median")


class TestRQRLoss:
    """Testing the RQRLoss class."""

    def test_values_and_reductions(self) -> None:
        predictions = torch.tensor([[0.0, 2.0], [0.0, 2.0], [0.0, 2.0]])
        targets = torch.tensor([1.0, 3.0, 0.0])

        loss = RQRLoss(coverage_level=0.8, reduction="none")
        assert loss(predictions, targets).tolist() == pytest.approx([0.2, 2.4, 0.0])

        assert RQRLoss(coverage_level=0.8)(predictions, targets) == pytest.approx(2.6 / 3)
        assert RQRLoss(coverage_level=0.8, reduction="sum")(predictions, targets) == pytest.approx(
            2.6
        )

    def test_permutation_invariance(self) -> None:
        predictions = torch.tensor([[0.0, 2.0], [4.0, 1.0]])
        targets = torch.tensor([1.0, 5.0])
        loss = RQRLoss(coverage_level=0.8, reduction="none")

        assert torch.equal(loss(predictions, targets), loss(predictions.flip(-1), targets))

    def test_width_regularization(self) -> None:
        predictions = torch.tensor([[0.0, 2.0], [0.0, 2.0]])
        targets = torch.tensor([1.0, 3.0])

        loss = RQRLoss(coverage_level=0.8, width_weight=0.05, reduction="none")

        # The corrected coverage level is 0.8 + 2 * 0.05 = 0.9 and the
        # squared-width penalty is 0.05 * (2 - 0) ** 2 / 2 = 0.1.
        assert loss(predictions, targets).tolist() == pytest.approx([0.2, 2.8])

        # The largest valid width weight remains accepted despite floating-point
        # representation of (1 - coverage_level) / 2.
        RQRLoss(coverage_level=0.9, width_weight=0.05)

    def test_gradients(self) -> None:
        predictions = torch.tensor([[0.0, 2.0], [0.0, 2.0]], requires_grad=True)
        targets = torch.tensor([1.0, 3.0])

        RQRLoss(coverage_level=0.8, reduction="sum")(predictions, targets).backward()

        assert predictions.grad is not None
        torch.testing.assert_close(
            predictions.grad,
            torch.tensor([[-0.2, 0.2], [-0.8, -2.4]]),
        )

    @pytest.mark.parametrize("coverage_level", [0.0, 1.0])
    def test_invalid_coverage_level(self, coverage_level: float) -> None:
        with pytest.raises(ValueError, match=r"The coverage level should be in \(0, 1\)"):
            RQRLoss(coverage_level=coverage_level)

    @pytest.mark.parametrize("width_weight", [-0.1, 0.0501, float("nan")])
    def test_invalid_width_weight(self, width_weight: float) -> None:
        with pytest.raises(ValueError, match="The width weight should be in"):
            RQRLoss(coverage_level=0.9, width_weight=width_weight)

    def test_invalid_reduction(self) -> None:
        with pytest.raises(ValueError, match=r"is not a valid value for reduction."):
            RQRLoss(coverage_level=0.9, reduction="median")

    @pytest.mark.parametrize(
        ("predictions", "targets", "message"),
        [
            (torch.tensor(1.0), torch.tensor(1.0), "exactly two interval endpoints"),
            (torch.ones(2, 3), torch.ones(2), "exactly two interval endpoints"),
            (torch.ones(2, 2), torch.ones(2, 1), "same shape"),
        ],
    )
    def test_invalid_shapes(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            RQRLoss(coverage_level=0.9)(predictions, targets)
