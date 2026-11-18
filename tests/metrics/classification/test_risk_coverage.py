import matplotlib.pyplot as plt
import pytest
import torch

from torch_uncertainty.metrics.classification import (
    AUGRC,
    AURC,
    CovAtxRisk,
    RiskAtxCov,
)


class TestAURC:
    """Testing the AURC metric class."""

    def test_compute_binary(self) -> None:
        probs = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.2])
        targets = torch.Tensor([1, 1, 1, 1, 1])
        metric = AURC()
        assert metric(probs, targets).item() == pytest.approx(1)
        targets = torch.Tensor([0, 0, 0, 0, 0])
        metric = AURC()
        assert metric(probs, targets).item() == pytest.approx(0)
        targets = torch.Tensor([0, 0, 1, 1, 0])
        metric = AURC()
        value = (0 * 0.4 + 0.25 * 0.2 / 2 + 0.25 * 0.2 + 0.15 * 0.2 / 2) / 0.8
        assert metric(probs, targets).item() == pytest.approx(value)

        metric = AURC()
        assert torch.isnan(metric(torch.Tensor([0.0]), torch.Tensor([1])))

    def test_compute_multiclass(self) -> None:
        probs = torch.Tensor([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.2, 0.8]])
        targets = torch.Tensor([1, 1, 1, 1, 1]).long()
        metric = AURC()
        assert metric(probs, targets).item() == pytest.approx(0)
        targets = torch.Tensor([0, 0, 0, 0, 0])
        metric = AURC()
        assert metric(probs, targets).item() == pytest.approx(1)
        targets = torch.Tensor([1, 1, 0, 0, 1])
        metric = AURC()
        value = (0 * 0.4 + 0.25 * 0.2 / 2 + 0.25 * 0.2 + 0.15 * 0.2 / 2) / 0.8
        assert metric(probs, targets).item() == pytest.approx(value)

    def test_compute_nan(self) -> None:
        probs = torch.Tensor([[0.1, 0.9]])
        targets = torch.Tensor([1]).long()
        metric = AURC()
        assert torch.isnan(metric(probs, targets)).all()

    def test_tied_scores_do_not_depend_on_order(self) -> None:
        # The last two samples have the same score, and only one of them is an error.
        # No threshold can separate them, so the result must not depend on their order.
        probs = torch.Tensor([[0.9, 0.1], [0.6, 0.4], [0.6, 0.4]])
        # Risks at coverage 1/3, 2/3 and 1 are 0, 1/4 and 1/3: the tied pair adds
        # half an error at each of its two positions.
        value = (1 / 3) * ((0 + 1 / 4) / 2 + (1 / 4 + 1 / 3) / 2) / (1 - 1 / 3)
        for targets in (torch.Tensor([0, 1, 0]), torch.Tensor([0, 0, 1])):
            metric = AURC()
            assert metric(probs, targets.long()).item() == pytest.approx(value)

    def test_plot(self) -> None:
        scores = torch.Tensor([0.2, 0.1, 0.5, 0.3, 0.4])
        values = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        metric = AURC()
        metric.update(scores, values)
        fig, ax = metric.plot()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert ax.get_xlabel() == "Coverage (%)"
        assert ax.get_ylabel() == "Risk - Error Rate (%)"
        plt.close(fig)

        metric = AURC()
        assert metric(torch.zeros(1), torch.zeros(1)).isnan()

        metric = AURC()
        metric.update(scores, values)
        fig, ax = metric.plot(plot_value=False)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert ax.get_xlabel() == "Coverage (%)"
        assert ax.get_ylabel() == "Risk - Error Rate (%)"
        plt.close(fig)

        # existing ax: fig should be None
        metric = AURC()
        metric.update(scores, values)
        _, existing_ax = plt.subplots()
        fig, ax = metric.plot(ax=existing_ax)
        assert fig is None
        assert ax is existing_ax
        plt.close("all")


class TestAUGRC:
    """Testing the AUGRC metric class."""

    def test_tied_scores_do_not_depend_on_order(self) -> None:
        probs = torch.Tensor([[0.9, 0.1], [0.6, 0.4], [0.6, 0.4]])
        # Generalized risks at coverage 1/3, 2/3 and 1 are 0, 1/6 and 1/3.
        value = (1 / 3) * ((0 + 1 / 6) / 2 + (1 / 6 + 1 / 3) / 2) / (1 - 1 / 3)
        for targets in (torch.Tensor([0, 1, 0]), torch.Tensor([0, 0, 1])):
            metric = AUGRC()
            assert metric(probs, targets.long()).item() == pytest.approx(value)

    def test_plot(self) -> None:
        scores = torch.Tensor([0.2, 0.1, 0.5, 0.3, 0.4])
        values = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        metric = AUGRC()
        metric.update(scores, values)
        fig, ax = metric.plot()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

        # existing ax: fig should be None
        metric = AUGRC()
        metric.update(scores, values)
        _, existing_ax = plt.subplots()
        fig, ax = metric.plot(ax=existing_ax)
        assert fig is None
        assert ax is existing_ax
        plt.close("all")


class TestCovAtxRisk:
    """Testing the CovAtxRisk metric class."""

    def test_compute_zero(self) -> None:
        probs = torch.Tensor([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.8, 0.2]])
        targets = torch.Tensor([1, 1, 1, 1, 1])
        metric = CovAtxRisk(risk_threshold=0.5)
        # no cov for given risk
        assert torch.isnan(metric(probs, targets))

        probs = torch.Tensor([0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.48, 0.49])
        targets = torch.Tensor([1, 0, 1, 1, 1, 0, 0, 0, 1])
        metric = CovAtxRisk(risk_threshold=0.55)
        # multiple cov for given risk
        assert metric(probs, targets) == pytest.approx(8 / 9)

        probs = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.2])
        targets = torch.Tensor([0, 0, 1, 1, 1])
        metric = CovAtxRisk(risk_threshold=0.5)
        assert metric(probs, targets) == pytest.approx(4 / 5)

        targets = torch.Tensor([0, 0, 1, 1, 0])
        metric = CovAtxRisk(risk_threshold=0.5)
        assert metric(probs, targets) == 1

        metric = CovAtxRisk(risk_threshold=0.5)
        assert metric(torch.zeros(0), torch.zeros(0)).isnan()

    def test_errors(self) -> None:
        with pytest.raises(TypeError, match=r"Expected threshold to be of type float"):
            CovAtxRisk(risk_threshold="0.5")
        with pytest.raises(ValueError, match=r"Threshold should be in the range"):
            CovAtxRisk(risk_threshold=-0.5)


class TestRiskAtxCov:
    """Testing the RiskAtxCov metric class."""

    def test_compute_zero(self) -> None:
        probs = torch.Tensor([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.8, 0.2]])
        targets = torch.Tensor([1, 1, 1, 1, 1])
        metric = RiskAtxCov(cov_threshold=0.5)
        assert metric(probs, targets) == 1

        probs = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.2])
        targets = torch.Tensor([0, 0, 1, 1, 1])
        metric = RiskAtxCov(cov_threshold=0.5)
        assert metric(probs, targets) == pytest.approx(1 / 3)

        probs = torch.Tensor([0.1, 0.19, 0.3, 0.15, 0.4, 0.2])
        targets = torch.Tensor([0, 0, 1, 0, 1, 1])
        metric = RiskAtxCov(cov_threshold=0.5)
        assert metric(probs, targets) == 0

        probs = torch.Tensor([0.1, 0.2, 0.3, 0.15, 0.4, 0.2])
        targets = torch.Tensor([0, 0, 1, 0, 1, 1])
        metric = RiskAtxCov(cov_threshold=0.55)
        assert metric(probs, targets) == 1 / 4

    def test_coverage_inside_tied_scores(self) -> None:
        # At 50% coverage, two of the three samples are accepted: the first one and
        # one of the tied pair. On average half of an error is accepted, so the risk is 1/4.
        probs = torch.Tensor([[0.9, 0.1], [0.6, 0.4], [0.6, 0.4]])
        for targets in (torch.Tensor([0, 1, 0]), torch.Tensor([0, 0, 1])):
            metric = RiskAtxCov(cov_threshold=0.5)
            assert metric(probs, targets.long()) == pytest.approx(1 / 4)
