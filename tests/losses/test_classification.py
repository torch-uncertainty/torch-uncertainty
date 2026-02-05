import pytest
import torch

from torch_uncertainty.losses import (
    BCEWithLogitsLSLoss,
    ConfidencePenaltyLoss,
    ConflictualLoss,
    CrossEntropyMaxSupLoss,
    DECLoss,
    FocalLoss,
    MixupMPLoss,
)


class TestDECLoss:
    """Testing the DECLoss class."""

    def test_main(self) -> None:
        loss = DECLoss(loss_type="mse", reg_weight=1e-2, annealing_step=1, reduction="sum")
        loss(torch.tensor([[0.0, 0.0]]), torch.tensor([0]), current_epoch=1)
        loss = DECLoss(loss_type="mse", reg_weight=1e-2, annealing_step=1)
        loss(torch.tensor([[0.0, 0.0]]), torch.tensor([0]), current_epoch=0)
        loss = DECLoss(loss_type="log", reg_weight=1e-2, reduction="none")
        loss(torch.tensor([[0.0, 0.0]]), torch.tensor([0]))
        loss = DECLoss(loss_type="digamma")
        loss(torch.tensor([[0.0, 0.0]]), torch.tensor([0]))

    def test_failures(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"The regularization weight should be non-negative, but got",
        ):
            DECLoss(reg_weight=-1)

        with pytest.raises(ValueError, match=r"The annealing step should be positive, but got "):
            DECLoss(annealing_step=0)

        loss = DECLoss(annealing_step=10)
        with pytest.raises(ValueError):
            loss(
                torch.tensor([[0.0, 0.0]]),
                torch.tensor([0]),
                current_epoch=None,
            )

        with pytest.raises(ValueError, match=r" is not a valid value for reduction."):
            DECLoss(reduction="median")

        with pytest.raises(ValueError, match=r"is not a valid value for mse/log/digamma loss."):
            DECLoss(loss_type="regression")


class TestConfidencePenaltyLoss:
    """Testing the ConfidencePenaltyLoss class."""

    def test_main(self) -> None:
        loss = ConfidencePenaltyLoss(reg_weight=1e-2, reduction="sum")
        loss(torch.tensor([[0.0, 0.0]]), torch.tensor([0]))
        loss = ConfidencePenaltyLoss(reg_weight=1e-2)
        loss(torch.tensor([[0.0, 0.0]]), torch.tensor([0]))
        loss = ConfidencePenaltyLoss(reg_weight=1e-2, reduction=None)
        loss(torch.tensor([[0.0, 0.0]]), torch.tensor([0]))

    def test_failures(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"The regularization weight should be non-negative, but got",
        ):
            ConfidencePenaltyLoss(reg_weight=-1)

        with pytest.raises(ValueError, match=r"is not a valid value for reduction."):
            ConfidencePenaltyLoss(reduction="median")

        with pytest.raises(
            ValueError,
            match=r"The epsilon value should be non-negative, but got",
        ):
            ConfidencePenaltyLoss(eps=-1)


class TestConflictualLoss:
    """Testing the ConflictualLoss class."""

    def test_main(self) -> None:
        loss = ConflictualLoss(reg_weight=1e-2, reduction="sum")
        loss(torch.tensor([[0.0, 0.0]]), torch.tensor([0]))
        loss = ConflictualLoss(reg_weight=1e-2)
        loss(torch.tensor([[0.0, 0.0]]), torch.tensor([0]))
        loss = ConflictualLoss(reg_weight=1e-2, reduction=None)
        loss(torch.tensor([[0.0, 0.0]]), torch.tensor([0]))

    def test_failures(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"The regularization weight should be non-negative, but got",
        ):
            ConflictualLoss(reg_weight=-1)

        with pytest.raises(ValueError, match=r"is not a valid value for reduction."):
            ConflictualLoss(reduction="median")


class TestFocalLoss:
    """Testing the FocalLoss class."""

    def test_main(self) -> None:
        loss = FocalLoss(gamma=1, reduction="sum")
        loss(torch.tensor([[0.0, 0.0]]), torch.tensor([0]))
        loss = FocalLoss(gamma=0.5)
        loss(torch.tensor([[0.0, 0.0]]), torch.tensor([0]))
        loss = FocalLoss(gamma=0.5, reduction=None)
        loss(torch.tensor([[0.0, 0.0]]), torch.tensor([0]))

    def test_failures(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"The gamma term of the focal loss should be non-negative, but got",
        ):
            FocalLoss(gamma=-1)

        with pytest.raises(ValueError, match=r"is not a valid value for reduction."):
            FocalLoss(gamma=1, reduction="median")


class TestBCEWithLogitsLSLoss:
    """Testing the BCEWithLogitsLSLoss class."""

    def test_main(self) -> None:
        loss = BCEWithLogitsLSLoss(reduction="sum", label_smoothing=0.1, weight=torch.Tensor([1]))
        loss(torch.tensor([0.0]), torch.tensor([0]))
        loss = BCEWithLogitsLSLoss(reduction="mean", label_smoothing=0.6)
        loss(torch.tensor([0.0]), torch.tensor([0]))
        loss = BCEWithLogitsLSLoss(reduction="none", label_smoothing=0.1)
        loss(torch.tensor([0.0]), torch.tensor([0]))
        loss = BCEWithLogitsLSLoss(reduction="none")
        loss(torch.tensor([0.0]), torch.tensor([0]))

    def test_failures(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"The label smoothing term of the BCE loss should be non-negative, but got",
        ):
            BCEWithLogitsLSLoss(label_smoothing=-1)


class TestCrossEntropyMaxSupLoss:
    """Testing the CrossEntropyMaxSupLoss class."""

    def test_main(self) -> None:
        loss = CrossEntropyMaxSupLoss(max_sup=0.0, reduction="mean")
        loss(torch.tensor([[0.0, 0.0]]), torch.tensor([0], dtype=torch.long))
        loss = CrossEntropyMaxSupLoss(max_sup=0.1, weight=torch.tensor([1, 1.2]), reduction="sum")
        loss(torch.tensor([[0.0, 0.0]]), torch.tensor([0], dtype=torch.long))
        loss = CrossEntropyMaxSupLoss(max_sup=0.1)
        loss(torch.tensor([[0.0, 0.0]]), torch.tensor([0], dtype=torch.long))
        loss = CrossEntropyMaxSupLoss(max_sup=0.1, reduction=None)
        loss(torch.tensor([[0.0, 0.0]]), torch.tensor([0], dtype=torch.long))


class TestMixupMPLoss:
    """Testing the MixupMPLoss class."""

    def test_main(self) -> None:
        loss = MixupMPLoss(mixup_ratio=2, reduction="mean")
        loss(torch.tensor([[0.0, 0.0]]), torch.tensor([0], dtype=torch.long))
        loss(torch.tensor([[0.0, 0.0]]), torch.tensor([0], dtype=torch.float32))

    def test_failures(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"mixup_ratio must be > 0. Got ",
        ):
            MixupMPLoss(mixup_ratio=-1)
