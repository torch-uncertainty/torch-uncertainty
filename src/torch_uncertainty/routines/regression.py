import warnings
from pathlib import Path

from einops import rearrange
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import Tensor, nn
from torch.distributions import (
    Distribution,
    Independent,
)
from torch.optim import Optimizer
from torch.utils.flop_counter import FlopCounterMode
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection

from torch_uncertainty.losses import ELBOLoss
from torch_uncertainty.metrics import (
    DistributionNLL,
    QuantileCalibrationError,
)
from torch_uncertainty.models import (
    EPOCH_UPDATE_MODEL,
    STEP_UPDATE_MODEL,
)
from torch_uncertainty.utils import csv_writer
from torch_uncertainty.utils.distributions import (
    get_dist_class,
    get_dist_estimate,
)


class RegressionRoutine(LightningModule):
    test_num_flops: int | None = None
    num_params: int | None = None

    def __init__(
        self,
        model: nn.Module,
        output_dim: int,
        loss: nn.Module | None = None,
        dist_family: str | None = None,
        dist_estimate: str = "mean",
        *,
        is_ensemble: bool = False,
        optim_recipe: dict | Optimizer | None = None,
        eval_shift: bool = False,
        format_batch_fn: nn.Module | None = None,
        log_plots: bool = False,
        num_bins_cal_err: int = 15,
        save_in_csv: bool = False,
        csv_filename: str = "results.csv",
    ) -> None:
        r"""Routine for training & testing on **regression** tasks.

        Args:
            model (torch.nn.Module): Model to train.
            output_dim (int): Number of outputs of the model.
            loss (torch.nn.Module): Loss function to optimize the :attr:`model`.
                Defaults to ``None``.
            dist_family (str, optional): The distribution family to use for probabilistic regression. If ``None`` then point-wise regression. Defaults to ``None``.
            dist_estimate (str, optional): The estimate to use when computing the point-wise metrics. Defaults to ``"mean"``.
            is_ensemble (bool, optional): Whether the model is an ensemble. Defaults to ``False``.
            optim_recipe (dict or torch.optim.Optimizer, optional): The optimizer and optionally the scheduler to use. Defaults to ``None``.
            eval_shift (bool, optional): Indicates whether to evaluate the Distribution shift performance. Defaults to ``False``.
            format_batch_fn (torch.nn.Module, optional): The function to format the batch. Defaults to ``None``.
            log_plots (bool, optional): Indicates whether to log figures in the logger.
                Defaults to ``False``.
            num_bins_cal_err (int, optional): Number of bins to compute calibration
                error metrics. Defaults to ``15``.
            save_in_csv (bool, optional): Save the results in csv. Defaults to ``False``.
            csv_filename (str, optional): Name of the csv file. Defaults to ``"results.csv"``. Note that this is only used if
                :attr:`save_in_csv` is ``True``.

        Warning:
            If :attr:`probabilistic` is True, the model must output a `PyTorch
            distribution <https://pytorch.org/docs/stable/distributions.html>`_.

        Warning:
            You must define :attr:`optim_recipe` if you do not use
            the CLI.

        Note:
            :attr:`optim_recipe` can be anything that can be returned by
            :meth:`LightningModule.configure_optimizers()`. Find more details
            `here <https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers>`_.
        """
        super().__init__()
        _regression_routine_checks(output_dim)
        if eval_shift:
            raise NotImplementedError(
                "Distribution shift evaluation not implemented yet. Raise an issue if needed."
            )

        self.model = model
        self.dist_family = dist_family
        self.dist_estimate = dist_estimate
        self.probabilistic = dist_family is not None
        self.output_dim = output_dim
        self.loss = loss
        self.is_ensemble = is_ensemble
        self.log_plots = log_plots
        self.save_in_csv = save_in_csv
        self.csv_filename = csv_filename
        self.needs_epoch_update = isinstance(model, EPOCH_UPDATE_MODEL)
        self.needs_step_update = isinstance(model, STEP_UPDATE_MODEL)
        self.num_bins_cal_err = num_bins_cal_err

        if format_batch_fn is None:
            format_batch_fn = nn.Identity()

        self.is_elbo = isinstance(self.loss, ELBOLoss)
        if self.is_elbo:
            self.loss.set_model(self.model)

        self.optim_recipe = optim_recipe
        self.format_batch_fn = format_batch_fn
        self.one_dim_regression = output_dim == 1
        self._init_metrics()

    def _init_metrics(self) -> None:
        """Initialize the metrics depending on the exact task."""
        reg_metrics = MetricCollection(
            {
                "reg/MAE": MeanAbsoluteError(),
                "reg/MSE": MeanSquaredError(squared=True),
                "reg/RMSE": MeanSquaredError(squared=False),
            },
            compute_groups=[["reg/MAE"], ["reg/MSE", "reg/RMSE"]],
        )

        self.val_metrics = reg_metrics.clone(prefix="val/")
        self.test_metrics = reg_metrics.clone(prefix="test/")

        if self.probabilistic:
            reg_prob_metrics = MetricCollection(
                {
                    "reg/NLL": DistributionNLL(reduction="mean"),
                    "cal/QCE": QuantileCalibrationError(
                        num_bins=self.num_bins_cal_err,
                    ),
                }
            )
            self.val_prob_metrics = reg_prob_metrics.clone(prefix="val/")
            self.test_prob_metrics = reg_prob_metrics.clone(prefix="test/")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return self.optim_recipe

    def on_train_start(self) -> None:  # coverage: ignore
        """Put the hyperparameters in tensorboard."""
        if self.loss is None:
            raise ValueError(
                "To train a model, you must specify the `loss` argument in the routine. Got None."
            )
        if self.logger is not None:
            self.logger.log_hyperparams(
                self.hparams,
            )

    def on_validation_start(self) -> None:
        """Prepare the validation step.

        Update the model's wrapper and the batchnorms if needed.
        """
        if self.needs_epoch_update and not self.trainer.sanity_checking:
            self.model.update_wrapper(self.current_epoch)
            if hasattr(self.model, "need_bn_update"):
                self.model.bn_update(self.trainer.train_dataloader, device=self.device)

    def on_test_start(self) -> None:
        """Prepare the test step.

        Update the batchnorms if needed.
        """
        if hasattr(self.model, "need_bn_update"):
            self.model.bn_update(self.trainer.train_dataloader, device=self.device)

        if self.num_params is None:
            self.num_params = sum(p.numel() for p in self.model.parameters())

    def forward(self, inputs: Tensor) -> Tensor | dict[str, Tensor]:
        """Forward pass of the routine.

        The forward pass automatically squeezes the output if the regression
        is one-dimensional and if the routine contains a single model.

        Args:
            inputs (Tensor): The input tensor.

        Returns:
            Tensor | dict[str, Tensor]: The output tensor or the parameters of the output
                distribution.
        """
        pred = self.model(inputs)
        if self.probabilistic:
            if isinstance(pred, dict):
                if self.one_dim_regression:
                    pred = {k: v.squeeze(-1) for k, v in pred.items()}
                if not self.is_ensemble:
                    pred = {k: v.squeeze(-1) for k, v in pred.items()}
            else:
                raise TypeError(
                    "The model is probabilistic: the output must be a dictionary ",
                    "of PyTorch distribution parameters.",
                )
        else:
            if self.one_dim_regression:
                pred = pred.squeeze(-1)
            if not self.is_ensemble:
                pred = pred.squeeze(-1)
        return pred

    def training_step(self, batch: tuple[Tensor, Tensor]) -> STEP_OUTPUT:
        """Perform a single training step based on the input tensors.

        Args:
            batch (tuple[Tensor, Tensor]): the training data and their corresponding targets

        Returns:
            Tensor: the loss corresponding to this training step.
        """
        inputs, targets = self.format_batch_fn(batch)

        if self.one_dim_regression:
            targets = targets.unsqueeze(-1)

        if isinstance(self.loss, ELBOLoss):
            loss = self.loss(inputs, targets)
        else:
            out = self.model(inputs)
            if self.probabilistic:
                # Adding the Independent wrapper to the distribution to compute correctly the
                # log-likelihood given a target. Here the last dimension is the event dimension.
                # When computing the log-likelihood, the values are summed over the event
                # dimension.
                dists = Independent(get_dist_class(self.dist_family)(**out), 1)
                loss = self.loss(dists, targets)
            else:
                loss = self.loss(out, targets)

        if self.needs_step_update:
            self.model.update_wrapper(self.current_epoch)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def evaluation_forward(self, inputs: Tensor) -> tuple[Tensor, Distribution | None]:
        """Get the prediction and handle predicted eventual distribution parameters.

        Args:
            inputs (Tensor): the input data.

        Returns:
            tuple[Tensor, Distribution | None]: the prediction as a Tensor and a distribution.
        """
        batch_size = inputs.size(0)
        preds = self.model(inputs)

        if self.probabilistic:
            dist_params = {
                k: rearrange(v, "(m b) c -> b m c", b=batch_size).mean(1) for k, v in preds.items()
            }
            dist = Independent(get_dist_class(self.dist_family)(**dist_params), 1)
            preds = get_dist_estimate(dist, self.dist_estimate)
            return preds, dist

        preds = rearrange(preds, "(m b) c -> b m c", b=batch_size)
        return preds.mean(dim=1), None

    def validation_step(self, batch: tuple[Tensor, Tensor]) -> None:
        """Perform a single validation step based on the input tensors.

        Compute the prediction of the model and the value of the metrics on the validation batch.

        Args:
            batch (tuple[Tensor, Tensor]): the validation data and their corresponding targets.
        """
        inputs, targets = batch
        if self.one_dim_regression:
            targets = targets.unsqueeze(-1)
        preds, dist = self.evaluation_forward(inputs)

        self.val_metrics.update(preds, targets)
        if isinstance(dist, Distribution):
            self.val_prob_metrics.update(dist, targets)

    def test_step(
        self,
        batch: tuple[Tensor, Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Perform a single test step based on the input tensors.

        Compute the prediction of the model and the value of the metrics on the test batch. Also
        handle OOD and distribution-shifted images.

        Args:
            batch (tuple[Tensor, Tensor]): the test data and their corresponding targets.
            batch_idx (int): the number of the current batch (unused).
            dataloader_idx (int): 0 if in-distribution, 1 if out-of-distribution.
        """
        if dataloader_idx != 0:
            raise NotImplementedError(
                "Regression OOD detection not implemented yet. Raise an issue if needed."
            )

        inputs, targets = batch

        if self.test_num_flops is None:
            flop_counter = FlopCounterMode(display=False)
            with flop_counter:
                self.forward(inputs)
            self.test_num_flops = flop_counter.get_total_flops()

        if self.one_dim_regression:
            targets = targets.unsqueeze(-1)
        preds, dist = self.evaluation_forward(inputs)

        self.test_metrics.update(preds, targets)
        if isinstance(dist, Distribution):
            self.test_prob_metrics.update(dist, targets)

    def on_validation_epoch_end(self) -> None:
        """Compute and log the values of the collected metrics in `validation_step`."""
        res_dict = self.val_metrics.compute()
        self.log_dict(res_dict, logger=True, sync_dist=True)
        self.log(
            "RMSE",
            res_dict["val/reg/RMSE"],
            prog_bar=True,
            logger=False,
            sync_dist=True,
        )
        self.val_metrics.reset()
        if self.probabilistic:
            prob_dict = self.val_prob_metrics.compute()
            self.log_dict(prob_dict, logger=True, sync_dist=True)
            self.log(
                "NLL",
                prob_dict["val/reg/NLL"],
                prog_bar=True,
                logger=False,
                sync_dist=True,
            )
            self.val_prob_metrics.reset()

    def on_test_epoch_end(self) -> None:
        """Compute and log the values of the collected metrics in `test_step`."""
        result_dict = self.test_metrics.compute() | {
            "test/cplx/flops": self.test_num_flops,
            "test/cplx/params": self.num_params,
        }
        self.test_metrics.reset()

        if self.probabilistic:
            result_dict |= self.test_prob_metrics.compute()

            if isinstance(self.logger, Logger) and self.log_plots:
                try:
                    self.logger.experiment.add_figure(
                        "Calibration/Reliability diagram",
                        self.test_prob_metrics["cal/QCE"].plot()[0],
                    )
                except NotImplementedError:
                    warnings.warn(
                        "The distribution does not support the `icdf()` method. "
                        "This metric will therefore return `nan` values. "
                        "Please use a distribution that implements `icdf()`.",
                        UserWarning,
                        stacklevel=2,
                    )

        self.log_dict(result_dict, sync_dist=True)

        self.test_metrics.reset()
        if self.probabilistic:
            self.test_prob_metrics.reset()

        if self.save_in_csv and self.logger is not None:
            csv_writer(
                Path(self.logger.log_dir) / self.csv_filename,
                result_dict,
            )


def _regression_routine_checks(output_dim: int) -> None:
    """Check the domains of the routine's parameters.

    Args:
        output_dim (int): the dimension of the output of the regression task.
    """
    if output_dim < 1:
        raise ValueError(f"output_dim must be positive, got {output_dim}.")
