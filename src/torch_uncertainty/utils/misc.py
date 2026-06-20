import csv
from pathlib import Path

import numpy as np
from lightning.pytorch.loggers import Logger, MLFlowLogger
from matplotlib.figure import Figure


def csv_writer(path: Path, metrics_dict: dict) -> None:
    """Write a dictionary of metric values to a csv file.

    Args:
        path: Path to the csv file.
        metrics_dict: Dictionary to write.
    """
    if path.is_file():  # Check that the file already exists
        append_mode = True
        rw_mode = "a"
    else:
        append_mode = False
        rw_mode = "w"
    # Write metrics_dict
    with path.open(rw_mode) as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        # Do not write header in append mode
        if append_mode is False:
            writer.writerow(metrics_dict.keys())
        writer.writerow([f"{elem:.4f}" for elem in metrics_dict.values()])


def get_logger_dir(logger: Logger) -> Path | None:
    """Return the per-run artifact directory for a Lightning logger.

    For ``MLFlowLogger`` with a local file tracking store, this is the artifact
    directory for the current run (creating the run if needed). For other
    loggers, falls back to ``logger.log_dir`` or ``logger.save_dir``. Returns
    ``None`` for non-filesystem MLflow tracking backends.
    """
    if isinstance(logger, MLFlowLogger):
        if logger.save_dir is None:
            return None  # remote tracking backend, no local filesystem path
        # `logger.version` lazily creates the run; `name` is the experiment id.
        return Path(logger.save_dir) / logger.name / logger.version / "artifacts"
    log_dir = getattr(logger, "log_dir", None) or getattr(logger, "save_dir", None)
    return Path(log_dir) if log_dir is not None else None


def log_figure(logger: Logger, tag: str, figure: Figure) -> None:
    """Log a matplotlib figure to TensorBoard or MLflow transparently.

    Silently no-ops for loggers whose experiment exposes neither ``add_figure``
    (TensorBoard SummaryWriter) nor ``log_figure`` (MlflowClient).
    """
    experiment = logger.experiment
    if isinstance(logger, MLFlowLogger):
        experiment.log_figure(logger.run_id, figure, f"{tag}.png")
    elif hasattr(experiment, "add_figure"):
        experiment.add_figure(tag, figure)


def log_image_array(
    logger: Logger,
    tag: str,
    image: np.ndarray,
    step: int | None = None,
) -> None:
    """Log a uint8 ``(H, W, C)`` image array to TensorBoard or MLflow.

    Silently no-ops for loggers exposing neither ``add_image`` nor ``log_image``.
    """
    experiment = logger.experiment
    if isinstance(logger, MLFlowLogger):
        experiment.log_image(logger.run_id, image, f"{tag}.png")
    elif hasattr(experiment, "add_image"):
        # TensorBoard expects (C, H, W); convert from (H, W, C).
        experiment.add_image(tag, np.transpose(image, (2, 0, 1)), global_step=step)
