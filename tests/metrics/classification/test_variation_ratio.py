import pytest
import torch

from torch_uncertainty.metrics import VariationRatio


@pytest.fixture
def disagreement_probas_3est() -> torch.Tensor:
    """Return a vector with mean entropy ~ln(2) and entropy of mean =0."""
    return torch.as_tensor([[[0.2, 0.8], [0.7, 0.3], [0.6, 0.4]]])


@pytest.fixture
def agreement_probas() -> torch.Tensor:
    return torch.as_tensor([[[0.9, 0.1], [0.9, 0.1]]])


@pytest.fixture
def agreement_probas_3est() -> torch.Tensor:
    """Return a vector with mean entropy ~ln(2) and entropy of mean =0."""
    return torch.as_tensor([[[0.2, 0.8], [0.3, 0.7], [0.4, 0.6]]])


class TestVariationRatio:
    """Testing the VariationRatio metric class."""

    def test_compute_agreement(self, agreement_probas: torch.Tensor) -> None:
        metric = VariationRatio(probabilistic=True)
        metric.update(agreement_probas)
        res = metric.compute()
        assert res == pytest.approx(0.1, 1e-6)

        metric = VariationRatio(probabilistic=False)
        metric.update(agreement_probas)
        res = metric.compute()
        assert res == 0.0

        metric = VariationRatio(probabilistic=False, reduction="none")
        metric.update(agreement_probas)
        res = metric.compute()
        assert res == 0.0

    def test_compute_disagreement(
        self,
        agreement_probas_3est: torch.Tensor,
        disagreement_probas_3est: torch.Tensor,
    ) -> None:
        metric = VariationRatio(probabilistic=True, reduction="sum")
        metric.update(disagreement_probas_3est)
        metric.update(agreement_probas_3est)
        res = metric.compute()
        assert res == pytest.approx(0.8, 1e-6)

    def test_bad_argument(self) -> None:
        with pytest.raises(ValueError):
            _ = VariationRatio(reduction="geometric_mean")


@pytest.mark.parametrize("probabilistic", [False, True])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none", None])
@pytest.mark.parametrize("streaming", [False, True])
def test_variation_ratio_batch_estimator_axes(probabilistic, reduction, streaming):
    # The majority vote and the argmax of mean probabilities differ in sample 0.
    probs = torch.tensor(
        [[[0.51, 0.49], [0.51, 0.49], [0.0, 1.0]], [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]]],
        dtype=torch.float64,
    )
    expected = torch.tensor([0.34, 0.2], dtype=torch.float64)
    if not probabilistic:
        expected = torch.tensor([1 / 3, 0.0])
    if reduction == "mean":
        expected = expected.mean()
    elif reduction == "sum":
        expected = expected.sum()
    metric = VariationRatio(probabilistic=probabilistic, reduction=reduction)
    if streaming:
        metric.update(probs[:1])
        metric.update(probs[1:])
    else:
        metric.update(probs)
    torch.testing.assert_close(metric.compute(), expected)


def test_variation_ratio_criterion_shape_and_majority():
    from torch_uncertainty.ood_criteria import VariationRatioCriterion

    probs = torch.tensor(
        [[[0.51, 0.49], [0.9, 0.1]], [[0.51, 0.49], [0.8, 0.2]], [[0.0, 1.0], [0.7, 0.3]]]
    )
    result = VariationRatioCriterion()(probs)
    torch.testing.assert_close(result, torch.tensor([1 / 3, 0.0]))


@pytest.mark.parametrize("probabilistic", [False, True])
def test_variation_ratio_unequal_batches(probabilistic):
    probs = torch.tensor([[[0.8, 0.2], [0.6, 0.4]]]).expand(3, -1, -1)
    metric = VariationRatio(probabilistic=probabilistic, reduction="none")
    metric.update(probs[:1])
    metric.update(probs[1:])
    expected = torch.full((3,), 0.3 if probabilistic else 0.0)
    torch.testing.assert_close(metric.compute(), expected)
