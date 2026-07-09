import contextlib
from pathlib import Path
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from huggingface_hub.errors import (
    HfHubHTTPError,
    RepositoryNotFoundError,
)
from lightning.pytorch.loggers import MLFlowLogger
from matplotlib.figure import Figure

from torch_uncertainty.utils import (
    csv_writer,
    distributions,
    get_logger_dir,
    get_version,
    hub,
    log_figure,
    log_image_array,
    plot_hist,
    plot_per_class_accuracy,
)
from torch_uncertainty.utils.distributions import TUStudentT


def _make_mlflow_logger(save_dir: str | None = "mlruns") -> MagicMock:
    """Return a MagicMock that passes isinstance(…, MLFlowLogger) checks."""
    logger = MagicMock()
    logger.__class__ = MLFlowLogger
    logger.save_dir = save_dir
    logger.name = "0"
    logger.version = "abc123def456"
    logger.run_id = "abc123def456"
    logger.experiment = MagicMock()
    return logger


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
        assert "Per-Class Accuracy" in ax.get_title()
        plt.close(fig)

    def test_plot_per_class_accuracy_custom_names(self) -> None:
        acc = torch.tensor([0.8, 0.6])
        fig, ax = plot_per_class_accuracy(acc, class_names=["cat", "dog"])
        assert isinstance(fig, plt.Figure)
        assert "Per-Class Accuracy" in ax.get_title()
        plt.close(fig)

    def test_plot_per_class_accuracy_topk(self) -> None:
        acc = torch.rand(50)
        fig, ax = plot_per_class_accuracy(acc, top_k=10)
        assert "10/50" in ax.get_title()
        plt.close(fig)

    def test_plot_per_class_accuracy_topk_larger_than_classes(self) -> None:
        acc = torch.tensor([0.9, 0.7, 0.5])
        fig, ax = plot_per_class_accuracy(acc, top_k=100)
        assert "Per-Class Accuracy" in ax.get_title()
        assert "100/" not in ax.get_title()
        plt.close(fig)

    def test_plot_per_class_accuracy_topk_none(self) -> None:
        acc = torch.rand(50)
        fig, ax = plot_per_class_accuracy(acc, top_k=None)
        assert "Per-Class Accuracy" in ax.get_title()
        assert "/" not in ax.get_title()
        plt.close(fig)


class TestMiscLoggers:
    """Tests for the logger-aware helper functions in misc.py."""

    # --- get_logger_dir ---

    def test_get_logger_dir_mlflow_local(self, tmp_path) -> None:
        logger = _make_mlflow_logger(save_dir=str(tmp_path))
        result = get_logger_dir(logger)
        assert result == tmp_path / "0" / "abc123def456" / "artifacts"

    def test_get_logger_dir_mlflow_remote(self) -> None:
        logger = _make_mlflow_logger(save_dir=None)
        assert get_logger_dir(logger) is None

    def test_get_logger_dir_log_dir(self, tmp_path) -> None:
        logger = MagicMock(spec=["log_dir", "experiment"])
        logger.log_dir = str(tmp_path / "tb_logs")
        assert get_logger_dir(logger) == tmp_path / "tb_logs"

    def test_get_logger_dir_save_dir_fallback(self, tmp_path) -> None:
        logger = MagicMock(spec=["save_dir", "experiment"])
        logger.save_dir = str(tmp_path / "save_logs")
        assert get_logger_dir(logger) == tmp_path / "save_logs"

    def test_get_logger_dir_no_dir(self) -> None:
        logger = MagicMock(spec=["experiment"])
        assert get_logger_dir(logger) is None

    # --- log_figure ---

    def test_log_figure_mlflow(self) -> None:
        logger = _make_mlflow_logger()
        fig = MagicMock(spec=Figure)
        log_figure(logger, "Reliability diagram", fig)
        logger.experiment.log_figure.assert_called_once_with(
            "abc123def456", fig, "Reliability diagram.png"
        )

    def test_log_figure_tensorboard(self) -> None:
        logger = MagicMock()
        logger.experiment = MagicMock()
        fig = MagicMock(spec=Figure)
        log_figure(logger, "some tag", fig)
        logger.experiment.add_figure.assert_called_once_with("some tag", fig)

    def test_log_figure_noop(self) -> None:
        logger = MagicMock()
        del logger.experiment.add_figure
        logger.experiment = MagicMock(spec=[])
        fig = MagicMock(spec=Figure)
        log_figure(logger, "tag", fig)  # must not raise

    # --- log_image_array ---

    def test_log_image_array_mlflow(self) -> None:
        logger = _make_mlflow_logger()
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        log_image_array(logger, "depth/samples", img, step=0)
        logger.experiment.log_image.assert_called_once_with(
            "abc123def456", img, "depth/samples.png"
        )

    def test_log_image_array_tensorboard(self) -> None:
        logger = MagicMock()
        logger.experiment = MagicMock()
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        log_image_array(logger, "depth/samples", img, step=2)
        call_args = logger.experiment.add_image.call_args
        assert call_args[0][0] == "depth/samples"
        assert call_args[1]["global_step"] == 2
        logged = call_args[0][1]
        assert logged.shape == (3, 4, 4)  # transposed to (C, H, W)

    def test_log_image_array_noop(self) -> None:
        logger = MagicMock()
        logger.experiment = MagicMock(spec=[])
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        log_image_array(logger, "tag", img)  # must not raise


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
