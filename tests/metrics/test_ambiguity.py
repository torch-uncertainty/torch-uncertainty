import pytest
import torch

from torch_uncertainty.metrics import Ambiguity


class TestAmbiguity:
    """Testing the Ambiguity metric class."""

    @pytest.mark.parametrize(
        ("reduction", "expected"),
        [("mean", 5 / 3), ("sum", 10 / 3), ("none", [2 / 3, 8 / 3]), (None, [2 / 3, 8 / 3])],
    )
    def test_reductions(self, reduction: str | None, expected: float | list[float]) -> None:
        preds = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
        result = Ambiguity(reduction=reduction)(preds)
        torch.testing.assert_close(result, torch.as_tensor(expected))

    def test_accumulation(self) -> None:
        metric = Ambiguity()
        metric.update(torch.tensor([[1.0, 3.0]]))
        metric.update(torch.tensor([[2.0, 6.0]]))
        torch.testing.assert_close(metric.compute(), torch.tensor(2.5))

    def test_multidimensional_outputs(self) -> None:
        preds = torch.tensor([[[1.0, 2.0], [3.0, 6.0]]])
        torch.testing.assert_close(Ambiguity()(preds), torch.tensor(2.5))

    def test_relative_ambiguity(self) -> None:
        preds = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
        result = Ambiguity(relative=True, reduction="none")(preds)
        torch.testing.assert_close(result, torch.tensor([1 / 6, 1 / 6]))

    def test_relative_ambiguity_multidimensional(self) -> None:
        preds = torch.tensor([[[1.0, -3.0], [3.0, -1.0]]])
        result = Ambiguity(relative=True)(preds)
        torch.testing.assert_close(result, torch.tensor(0.25))

    def test_relative_ambiguity_is_scale_invariant(self) -> None:
        preds = torch.tensor([[1.0, 2.0, 3.0]])
        metric = Ambiguity(relative=True)
        torch.testing.assert_close(metric(preds), metric(10 * preds))

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
    def test_relative_ambiguity_zero_ensemble_prediction(self, dtype: torch.dtype) -> None:
        result = Ambiguity(relative=True)(torch.tensor([[-1.0, 1.0]], dtype=dtype))
        assert torch.isfinite(result)
        assert result > 0

    def test_ambiguity_decomposition(self) -> None:
        preds = torch.tensor([[1.0, 2.0, 4.0], [2.0, 5.0, 8.0]])
        target = torch.tensor([3.0, 4.0])
        ensemble_error = (preds.mean(dim=1) - target).square().mean()
        mean_member_error = (preds - target.unsqueeze(1)).square().mean()

        torch.testing.assert_close(ensemble_error, mean_member_error - Ambiguity()(preds))

    @pytest.mark.parametrize(
        "preds",
        [torch.tensor([1.0, 2.0]), torch.tensor([[1.0]]), torch.tensor([[1, 2]])],
    )
    def test_invalid_predictions(self, preds: torch.Tensor) -> None:
        with pytest.raises(ValueError):
            Ambiguity().update(preds)

    def test_invalid_reduction(self) -> None:
        with pytest.raises(ValueError, match="reduction"):
            Ambiguity(reduction="geometric_mean")
