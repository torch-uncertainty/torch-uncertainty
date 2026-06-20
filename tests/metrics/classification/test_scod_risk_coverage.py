import pytest
import torch

from torch_uncertainty.metrics.classification import (
    SCODAUGRC,
    SCODAURC,
    SCODCovAt5Risk,
    SCODRiskAt80Cov,
)


class TestSCODAURC:
    def test_compute_binary_extremes(self) -> None:
        ood_scores = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.2])
        labels = torch.zeros(5)
        metric = SCODAURC()
        assert metric(ood_scores, labels).item() == pytest.approx(0)

        labels = torch.ones(5)
        metric = SCODAURC()
        assert metric(ood_scores, labels).item() == pytest.approx(1)

        metric = SCODAURC()
        assert metric(torch.tensor([0.0]), torch.tensor([1.0])).isnan()


class TestSCODAUGRC:
    def test_compute_binary_extremes(self) -> None:
        ood_scores = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.2])
        labels = torch.zeros(5)
        metric = SCODAUGRC()
        assert metric(ood_scores, labels).item() == pytest.approx(0)

        labels = torch.ones(5)
        metric = SCODAUGRC()
        assert metric(ood_scores, labels).item() == pytest.approx(0.6)


class TestSCODCovAt5Risk:
    def test_compute(self) -> None:
        ood_scores = torch.tensor([0.05, 0.1, 0.2, 0.3, 0.4])
        labels = torch.zeros(5)
        metric = SCODCovAt5Risk()
        assert metric(ood_scores, labels) == 1


class TestSCODRiskAt80Cov:
    def test_compute(self) -> None:
        ood_scores = torch.tensor([0.05, 0.1, 0.8, 0.9, 0.95])
        labels = torch.tensor([0, 0, 1, 1, 1])
        metric = SCODRiskAt80Cov()
        assert metric(ood_scores, labels) == 0.5
