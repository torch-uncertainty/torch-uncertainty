import contextlib
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
from huggingface_hub.errors import (
    HfHubHTTPError,
    RepositoryNotFoundError,
)

from torch_uncertainty.utils import (
    csv_writer,
    distributions,
    get_version,
    hub,
    plot_hist,
    plot_per_class_accuracy,
)
from torch_uncertainty.utils.distributions import TUStudentT


class TestUtils:
    """Testing utils methods."""

    def test_get_version_log_success(self) -> None:
        get_version("tests/testlog", version=42)
        get_version(Path("tests/testlog"), version=42)

        get_version("tests/testlog", version=42, checkpoint=45)

    def test_getversion_log_failure(self) -> None:
        with pytest.raises(FileNotFoundError):
            get_version("tests/testlog", version=52)


class TestHub:
    """Testing hub methods."""

    def test_hub_exists(self) -> None:
        hub.load_hf("test")
        hub.load_hf("test", version=1)
        hub.load_hf("test", version=2)

    def test_hub_notexists(self) -> None:
        with (
            contextlib.suppress(ValueError),
            pytest.raises((RepositoryNotFoundError, HfHubHTTPError)),
        ):
            hub.load_hf("tests")

        with contextlib.suppress(ValueError), pytest.raises((ValueError, HfHubHTTPError)):
            hub.load_hf("test", version=42)


class TestMisc:
    """Testing misc methods."""

    def test_csv_writer(self) -> None:
        root = Path(__file__).parent.resolve()
        csv_writer(root / "testlog" / "results.csv", {"a": 1.0, "b": 2.0})
        csv_writer(root / "testlog" / "results.csv", {"a": 1.0, "b": 2.0, "c": 3.0})

    def test_plot_hist(self) -> None:
        conf = [torch.rand(20), torch.rand(20)]
        plot_hist(conf, bins=10, title="test")

    def test_plot_per_class_accuracy_default_names(self) -> None:
        acc = torch.tensor([0.9, 0.7, 0.5])
        fig, ax = plot_per_class_accuracy(acc)
        assert isinstance(fig, plt.Figure)
        assert ax.get_xlabel() == "Accuracy"
        plt.close(fig)

    def test_plot_per_class_accuracy_custom_names(self) -> None:
        acc = torch.tensor([0.8, 0.6])
        fig, ax = plot_per_class_accuracy(acc, class_names=["cat", "dog"])
        assert isinstance(fig, plt.Figure)
        assert [t.get_text() for t in ax.get_yticklabels()] == ["cat", "dog"]
        plt.close(fig)


class TestDistributions:
    """Testing distributions methods."""

    def test_nig(self) -> None:
        dist = distributions.NormalInverseGamma(
            0.0,
            1.1,
            1.1,
            1.1,
        )
        dist = distributions.NormalInverseGamma(
            torch.tensor(0.0),
            torch.tensor(1.1),
            torch.tensor(1.1),
            torch.tensor(1.1),
        )
        _ = dist.mean, dist.mean_loc, dist.mean_variance, dist.variance_loc

    def test_get_dist_class(self) -> None:
        dist = distributions.get_dist_class("normal")
        assert dist == torch.distributions.Normal
        dist = distributions.get_dist_class("laplace")
        assert dist == torch.distributions.Laplace
        dist = distributions.get_dist_class("nig")
        assert dist == distributions.NormalInverseGamma
        dist = distributions.get_dist_class("cauchy")
        assert dist == torch.distributions.Cauchy
        dist = distributions.get_dist_class("student")
        assert dist == TUStudentT

    def test_get_dist_estimate(self) -> None:
        dist = torch.distributions.Normal(0.0, 1.0)
        mean = distributions.get_dist_estimate(dist, "mean")
        mode = distributions.get_dist_estimate(dist, "mode")
        assert mean == mode

    def test_tu_student_t(self) -> None:
        dist = TUStudentT(df=2.0, loc=0.0, scale=1.0)
        assert torch.allclose(dist.cdf(torch.tensor(0.0)), torch.tensor(0.5))
        assert torch.allclose(dist.icdf(torch.tensor(0.5)), torch.tensor(0.0))
        assert dist.mean == torch.tensor(0.0)
        assert dist.mode == torch.tensor(0.0)
        assert dist.variance == torch.tensor(float("inf"))

        dist = TUStudentT(df=1.0, loc=0.0, scale=1.0)
        assert dist.variance.isnan().all()

        dist = TUStudentT(df=3.0, loc=0.0, scale=1.0)
        assert dist.variance == torch.tensor(3.0)
