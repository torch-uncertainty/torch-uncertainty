from pathlib import Path
from typing import Literal

import matplotlib.cm as cm
import torch
from einops import rearrange
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor, nn
from torch.distributions import (
    Categorical,
    Distribution,
    Independent,
    MixtureSameFamily,
)
from torch.optim import Optimizer
from torch.utils.flop_counter import FlopCounterMode
from torchmetrics import MeanSquaredError, MetricCollection
from torchvision.transforms.v2 import functional as F
from torchvision.utils import make_grid

from torch_uncertainty.metrics import (
    DistributionNLL,
    Log10,
    MeanAbsoluteErrorInverse,
    MeanGTRelativeAbsoluteError,
    MeanGTRelativeSquaredError,
    MeanSquaredErrorInverse,
    MeanSquaredLogError,
    SILog,
    ThresholdAccuracy,
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


class PixelRegressionRoutine(LightningModule):
    inv_norm_params = {
        "mean": [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
        "std": [1 / 0.229, 1 / 0.224, 1 / 0.255],
    }
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
        format_batch_fn: nn.Module | None = None,
        optim_recipe: dict | Optimizer | None = None,
        eval_shift: bool = False,
        num_image_plot: int = 4,
        log_plots: bool = False,
        save_in_csv: bool = False,
        csv_filename: str = "results.csv",
    ) -> None:
        r"""Routine for training & testing on **pixel regression** tasks.

        Args:
            model (nn.Module): Model to train.
            output_dim (int): Number of outputs of the model.
            loss (nn.Module): Loss function to optimize the :attr:`model`.
                Defaults to ``None``.
            dist_family (str, optional): The distribution family to use for
                probabilistic pixel regression. If ``None`` then point-wise regression.
                Defaults to ``None``.
            dist_estimate (str, optional): The estimate to use when computing the
                point-wise metrics. Defaults to ``"mean"``.
            is_ensemble (bool, optional): Whether the model is an ensemble.
                Defaults to ``False``.
            optim_recipe (dict or Optimizer, optional): The optimizer and
                optionally the scheduler to use. Defaults to ``None``.
            eval_shift (bool, optional): Indicates whether to evaluate the Distribution
                shift performance. Defaults to ``False``.
            format_batch_fn (nn.Module, optional): The function to format the
                batch. Defaults to ``None``.
            num_image_plot (int, optional): Number of images to plot. Defaults to ``4``.
            log_plots (bool, optional): Indicates whether to log plots from
                metrics. Defaults to ``False``.
            save_in_csv (bool, optional): Save the results in csv. Defaults to
                ``False``.
            csv_filename (str, optional): Name of the csv file. Defaults to
                ``"results.csv"``. Note that this is only used if
                :attr:`save_in_csv` is ``True``.
        """
        super().__init__()
        _depth_routine_checks(output_dim, num_image_plot, log_plots)
        if eval_shift:
            raise NotImplementedError(
                "Distribution shift evaluation not implemented yet. Raise an issue if needed."
            )

        self.model = model
        self.output_dim = output_dim
        self.one_dim_depth = output_dim == 1
        self.dist_family = dist_family
        self.dist_estimate = dist_estimate
        self.probabilistic = dist_family is not None
        self.loss = loss
        self.save_in_csv = save_in_csv
        self.csv_filename = csv_filename
        self.num_image_plot = num_image_plot
        self.is_ensemble = is_ensemble
        self.log_plots = log_plots

        self.needs_epoch_update = isinstance(model, EPOCH_UPDATE_MODEL)
        self.needs_step_update = isinstance(model, STEP_UPDATE_MODEL)

        if format_batch_fn is None:
            format_batch_fn = nn.Identity()

        self.optim_recipe = optim_recipe
        self.format_batch_fn = format_batch_fn
        self._init_metrics()

    def _init_metrics(self) -> None:
        """Initialize the metrics depending on the exact task."""
        depth_metrics = MetricCollection(
            {
                "reg/SILog": SILog(),
                "reg/log10": Log10(),
                "reg/ARE": MeanGTRelativeAbsoluteError(),
                "reg/RSRE": MeanGTRelativeSquaredError(squared=False),
                "reg/RMSE": MeanSquaredError(squared=False),
                "reg/RMSELog": MeanSquaredLogError(squared=False),
                "reg/iMAE": MeanAbsoluteErrorInverse(),
                "reg/iRMSE": MeanSquaredErrorInverse(squared=False),
                "reg/d1": ThresholdAccuracy(power=1),
                "reg/d2": ThresholdAccuracy(power=2),
                "reg/d3": ThresholdAccuracy(power=3),
            },
            compute_groups=False,
        )

        self.val_metrics = depth_metrics.clone(prefix="val/")
        self.test_metrics = depth_metrics.clone(prefix="test/")

        if self.probabilistic:
            depth_prob_metrics = MetricCollection({"reg/NLL": DistributionNLL(reduction="mean")})
            self.val_prob_metrics = depth_prob_metrics.clone(prefix="val/")
            self.test_prob_metrics = depth_prob_metrics.clone(prefix="test/")

    def configure_optimizers(self) -> Optimizer | dict:
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

    def forward(self, inputs: Tensor) -> Tensor | Distribution:
        """Forward pass of the routine.

        The forward pass automatically squeezes the output if the regression
        is one-dimensional and if the routine contains a single model.

        Args:
            inputs (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        pred = self.model(inputs)
        if self.probabilistic:
            if not self.is_ensemble:
                pred = {k: v.squeeze(-1) for k, v in pred.items()}
        else:
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
        inputs, target = self.format_batch_fn(batch)
        if self.one_dim_depth:
            target = target.unsqueeze(1)

        out = self.model(inputs)
        out_shape = out[next(iter(out))].shape[-2:] if self.probabilistic else out.shape[-2:]
        target = F.resize(target, out_shape, interpolation=F.InterpolationMode.NEAREST)
        target = rearrange(target, "b c h w -> b h w c")
        padding_mask = torch.isnan(target).any(dim=-1)
        if self.probabilistic:
            dist_params = {k: rearrange(v, "b c h w -> b h w c") for k, v in out.items()}
            # Adding the Independent wrapper to the distribution to compute correctly the
            # log-likelihood given a target. Here the last dimension is the event dimension.
            # When computing the log-likelihood, the values are summed over the event dimension.
            dists = Independent(get_dist_class(self.dist_family)(**dist_params), 1)
            loss = self.loss(dists, target, padding_mask)
        else:
            out = rearrange(out, "b c h w -> b h w c")
            loss = self.loss(out[padding_mask], target[padding_mask])

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
                k: rearrange(v, "(m b) c h w -> b h w m c", b=batch_size) for k, v in preds.items()
            }
            # Adding the Independent wrapper to the distribution to create a MixtureSameFamily.
            # As required by the torch.distributions API, the last dimension is the event dimension.
            comp = Independent(get_dist_class(self.dist_family)(**dist_params), 1)
            mix = Categorical(torch.ones(comp.batch_shape, device=self.device))
            mixture = MixtureSameFamily(mix, comp)
            preds = get_dist_estimate(comp, self.dist_estimate).mean(-2)
            return preds, mixture

        preds = rearrange(preds, "(m b) c h w -> b m h w c", b=batch_size)
        return preds.mean(dim=1), None

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Perform a single validation step based on the input tensors.

        Compute the prediction of the model and the value of the metrics on the validation batch.

        Args:
            batch (tuple[Tensor, Tensor]): the validation images and their corresponding targets.
            batch_idx (int): the id of the batch. Optionally plot images and the predictions with
                the first batch.
        """
        inputs, targets = batch
        if self.one_dim_depth:
            targets = targets.unsqueeze(1)
        targets = rearrange(targets, "b c h w -> b h w c")
        preds, dist = self.evaluation_forward(inputs)

        if batch_idx == 0 and self.log_plots:
            self._plot_pixel_regression(
                inputs[: self.num_image_plot, ...],
                preds[: self.num_image_plot, ...],
                targets[: self.num_image_plot, ...],
                stage="val",
            )

        padding_mask = torch.isnan(targets).any(dim=-1)
        self.val_metrics.update(preds[padding_mask], targets[padding_mask])
        if isinstance(dist, Distribution):
            self.val_prob_metrics.update(dist, targets, padding_mask)

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
                "Depth OOD detection not implemented yet. Raise an issue if needed."
            )
        inputs, targets = batch
        if self.test_num_flops is None:
            flop_counter = FlopCounterMode(display=False)
            with flop_counter:
                self.forward(inputs)
            self.test_num_flops = flop_counter.get_total_flops()
        if self.one_dim_depth:
            targets = targets.unsqueeze(1)
        targets = rearrange(targets, "b c h w -> b h w c")
        preds, dist = self.evaluation_forward(inputs)

        if batch_idx == 0 and self.log_plots:
            num_images = min(inputs.size(0), self.num_image_plot)
            self._plot_pixel_regression(
                inputs[:num_images, ...],
                preds[:num_images, ...],
                targets[:num_images, ...],
                stage="test",
            )

        padding_mask = torch.isnan(targets).any(dim=-1)
        self.test_metrics.update(preds[padding_mask], targets[padding_mask])
        if isinstance(dist, Distribution):
            self.test_prob_metrics.update(dist, targets, padding_mask)

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
            self.log_dict(
                self.val_prob_metrics.compute(),
                sync_dist=True,
            )
            self.val_prob_metrics.reset()

    def on_test_epoch_end(self) -> None:
        """Compute and log the values of the collected metrics in `test_step`."""
        result_dict = self.test_metrics.compute()
        result_dict |= {
            "test/cplx/flops": self.test_num_flops,
            "test/cplx/params": self.num_params,
        }

        if self.probabilistic:
            result_dict |= self.test_prob_metrics.compute()

        self.log_dict(result_dict, sync_dist=True)

        self.test_metrics.reset()
        if self.probabilistic:
            self.test_prob_metrics.reset()

        if self.save_in_csv and self.logger is not None:
            csv_writer(
                Path(self.logger.log_dir) / self.csv_filename,
                result_dict,
            )

    def _plot_pixel_regression(
        self,
        inputs: Tensor,
        preds: Tensor,
        target: Tensor,
        stage: Literal["val", "test"],
    ) -> None:
        if (
            self.logger is not None
            and isinstance(self.logger, TensorBoardLogger)
            and self.one_dim_depth
        ):
            all_imgs = []
            for i in range(inputs.size(0)):
                img = F.normalize(inputs[i, ...].cpu(), **self.inv_norm_params)
                pred = colorize(preds[i, 0, ...].cpu(), vmin=0, vmax=self.model.max_depth)
                tgt = colorize(target[i, 0, ...].cpu(), vmin=0, vmax=self.model.max_depth)
                all_imgs.extend([img, pred, tgt])

            self.logger.experiment.add_image(
                f"{stage}/samples",
                make_grid(torch.stack(all_imgs, dim=0), nrow=3),
                self.current_epoch,
            )


def colorize(
    value: Tensor,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "magma",
) -> Tensor:
    """Colorize a tensor of depth values.

    Args:
        value (Tensor): The tensor of depth values.
        vmin (float, optional): The minimum depth value. Defaults to None.
        vmax (float, optional): The maximum depth value. Defaults to None.
        cmap (str, optional): The colormap to use. Defaults to 'magma'.
    """
    vmin = value.min().item() if vmin is None else vmin
    vmax = value.max().item() if vmax is None else vmax
    if vmin == vmax:
        return torch.zeros_like(value)
    value = (value - vmin) / (vmax - vmin)
    cmapper = cm.get_cmap(cmap)
    value = cmapper(value.numpy(), bytes=True)
    img = value[:, :, :3]
    return torch.as_tensor(img).permute(2, 0, 1).float() / 255.0


def _depth_routine_checks(output_dim: int, num_image_plot: int, log_plots: bool) -> None:
    """Check the domains of the routine's parameters.

    Args:
        output_dim (int): the dimension of the output of the regression task.
        num_image_plot (int): the number of images to plot at evaluation time.
        log_plots (bool): whether to plot images and predictions during evaluation.
    """
    if output_dim < 1:
        raise ValueError(f"output_dim must be positive, got {output_dim}.")
    if num_image_plot < 1 and log_plots:
        raise ValueError(f"num_image_plot must be positive, got {num_image_plot}.")
