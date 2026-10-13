import math

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

    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4), (2, 3, 2, 4)])
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_class_axis(self, shape: tuple[int, ...], reduction: str, dtype: torch.dtype) -> None:
        logits = torch.linspace(-1.5, 2.0, math.prod(shape), dtype=dtype).reshape(shape)
        logits.requires_grad_()
        targets = torch.arange(logits[:, 0].numel()).reshape(logits[:, 0].shape) % shape[1]
        loss_fn = ConfidencePenaltyLoss(reg_weight=0.7, reduction=reduction)
        actual = loss_fn(logits, targets)
        log_probs = logits.log_softmax(dim=1)
        probs = log_probs.exp()
        cross_entropy = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        penalty = math.log(shape[1]) + (probs * (probs + loss_fn.eps).log()).sum(dim=1)
        expected = cross_entropy + 0.7 * penalty
        if reduction == "mean":
            expected = expected.mean()
        elif reduction == "sum":
            expected = expected.sum()
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        actual_grad = torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected.sum(), logits)[0]
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-6, atol=1e-7)

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

    @pytest.mark.parametrize("reduction", ["none", None, "mean", "sum"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_per_example_penalty(
        self, reduction: str | None, dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        logits = torch.tensor(
            [[1.0, 0.0, -1.0], [0.0, 2.0, -0.5], [-1.0, 0.5, 1.0]],
            dtype=dtype,
            requires_grad=True,
        )
        targets = torch.tensor([0, 1, 2])
        monkeypatch.setattr(
            torch, "randint", lambda *_args, **kwargs: torch.tensor([2], device=kwargs["device"])
        )
        actual = ConflictualLoss(reg_weight=0.4, reduction=reduction)(logits, targets)
        log_probs = logits.log_softmax(dim=1)
        expected = -log_probs[torch.arange(3), targets] - 0.4 * log_probs[:, 2]
        if reduction == "mean":
            expected = expected.mean()
        elif reduction == "sum":
            expected = expected.sum()
        torch.testing.assert_close(actual, expected)
        weights = torch.tensor([1.0, 2.0, 4.0], dtype=dtype) if expected.ndim else 1.0
        actual_grad = torch.autograd.grad((actual * weights).sum(), logits, retain_graph=True)[0]
        expected_grad = torch.autograd.grad((expected * weights).sum(), logits)[0]
        torch.testing.assert_close(actual_grad, expected_grad)

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

    @pytest.mark.parametrize("mixup_ratio", [0.5, 1.0, 2.0])
    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_hard_labels(self, mixup_ratio: float, reduction: str) -> None:
        """Class-index targets produce a finite scalar with a usable gradient."""
        loss = MixupMPLoss(mixup_ratio=mixup_ratio, reduction=reduction)
        logits = torch.randn(8, 3, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])

        out = loss(logits, targets)

        assert out.ndim == 0
        assert torch.isfinite(out)
        out.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()

    @pytest.mark.parametrize("mixup_ratio", [0.5, 1.0, 2.0])
    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_soft_labels(self, mixup_ratio: float, reduction: str) -> None:
        """Soft (float) targets route through KL-div on both branches."""
        loss = MixupMPLoss(mixup_ratio=mixup_ratio, reduction=reduction)
        logits = torch.randn(8, 3, requires_grad=True)
        targets = torch.softmax(torch.randn(8, 3), dim=-1)

        out = loss(logits, targets)

        assert out.ndim == 0
        assert torch.isfinite(out)
        out.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()

    def test_empty_norm_branch(self) -> None:
        """When ``mixup_ratio`` consumes the whole batch, the normal slice is
        empty and must contribute zero without invoking cross-entropy.

        Regression test for the torch >= 2.12 strict dtype validation on empty
        batches (previously raised ``RuntimeError: expected target dtype to be
        Long or Byte, but got Float``).
        """
        loss = MixupMPLoss(mixup_ratio=2.0)
        out_hard = loss(torch.tensor([[0.0, 0.0]]), torch.tensor([0], dtype=torch.long))
        out_soft = loss(torch.tensor([[0.0, 0.0]]), torch.tensor([[1.0, 0.0]]))

        assert torch.isfinite(out_hard)
        assert torch.isfinite(out_soft)

    def test_empty_mixup_branch(self) -> None:
        """A small ``mixup_ratio`` leaves the mixup slice empty; the normal
        branch alone must still produce a finite loss.
        """
        loss = MixupMPLoss(mixup_ratio=0.01)
        out = loss(torch.randn(4, 3), torch.tensor([0, 1, 2, 0]))

        assert torch.isfinite(out)

    def test_equivalent_to_ce_when_all_normal(self) -> None:
        """With a vanishingly small ratio the whole batch lands on the normal
        branch, so the loss should match plain cross-entropy (up to the mixup
        weighting on an empty branch, which is 0).
        """
        loss = MixupMPLoss(mixup_ratio=1e-6, reduction="mean")
        logits = torch.randn(16, 5)
        targets = torch.randint(0, 5, (16,))

        ours = loss(logits, targets)
        reference = torch.nn.functional.cross_entropy(logits, targets, reduction="mean")

        assert torch.allclose(ours, reference)

    def test_failures(self) -> None:
        with pytest.raises(ValueError, match=r"mixup_ratio must be > 0\. Got "):
            MixupMPLoss(mixup_ratio=-1)
        with pytest.raises(ValueError, match=r"mixup_ratio must be > 0\. Got "):
            MixupMPLoss(mixup_ratio=0)
