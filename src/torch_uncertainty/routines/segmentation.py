import logging
from pathlib import Path

import torch
from einops import rearrange
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.flop_counter import FlopCounterMode
from torchmetrics import Accuracy, MetricCollection
from torchvision.transforms.v2 import ToDtype
from torchvision.transforms.v2 import functional as F
from torchvision.utils import draw_segmentation_masks

from torch_uncertainty.metrics import (
    AUGRC,
    AURC,
    BrierScore,
    CalibrationError,
    CategoricalNLL,
    CovAt5Risk,
    MeanIntersectionOverUnion,
    RiskAt80Cov,
    SegmentationBinaryAUROC,
    SegmentationBinaryAveragePrecision,
    SegmentationFPR95,
)
from torch_uncertainty.models import (
    EPOCH_UPDATE_MODEL,
    STEP_UPDATE_MODEL,
)
from torch_uncertainty.ood_criteria import (
    OODCriterionInputType,
    TUOODCriterion,
    get_ood_criterion,
)
from torch_uncertainty.post_processing import PostProcessing
from torch_uncertainty.utils import csv_writer
from torch_uncertainty.utils.plotting import show_segmentation_predictions


class SegmentationRoutine(LightningModule):
    test_num_flops: int | None = None
    num_params: int | None = None

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        loss: nn.Module | None = None,
        *,
        optim_recipe: dict | Optimizer | None = None,
        eval_shift: bool = False,
        format_batch_fn: nn.Module | None = None,
        metric_subsampling_rate: float = 1e-2,
        eval_ood: bool = False,
        ood_criterion: TUOODCriterion | str = "msp",
        post_processing: PostProcessing | None = None,
        log_plots: bool = False,
        num_samples_to_plot: int = 3,
        num_bins_cal_err: int = 15,
        save_in_csv: bool = False,
        csv_filename: str = "results.csv",
    ) -> None:
        r"""Routine for training & testing on **segmentation** tasks.

        Args:
            model (torch.nn.Module): Model to train.
            num_classes (int): Number of classes in the segmentation task.
            loss (torch.nn.Module): Loss function to optimize the :attr:`model`.
                Defaults to ``None``.
            optim_recipe (dict or Optimizer, optional): The optimizer and
                optionally the scheduler to use. Defaults to ``None``.
            eval_shift (bool, optional): Indicates whether to evaluate the Distribution
                shift performance. Defaults to ``False``.
            format_batch_fn (torch.nn.Module, optional): The function to format the
                batch. Defaults to ``None``.
            metric_subsampling_rate (float, optional): The rate of subsampling for the
                memory consuming metrics. Defaults to ``1e-2``.
            eval_ood (bool, optional): Indicates whether to evaluate the OOD
                performance. Defaults to ``False``.
            ood_criterion (TUOODCriterion, optional): Criterion for the binary OOD detection task.
                Defaults to ``"msp"`` which amounts to the maximum softmax probability score (MSP).
            post_processing (PostProcessing, optional): The post-processing
                technique to use. Defaults to ``None``. Warning: There is no
                post-processing technique implemented yet for segmentation tasks.
            log_plots (bool, optional): Indicates whether to log figures in the logger.
                Defaults to ``False``.
            num_samples_to_plot (int, optional): Number of segmentation prediction and
                target to plot in the logger. Note that this is only used if
                :attr:`log_plots` is set to ``True``. Defaults to ``3``.
            num_bins_cal_err (int, optional): Number of bins to compute calibration
                error metrics. Defaults to ``15``.
            save_in_csv (bool, optional): Save the results in csv. Defaults to
                ``False``.
            csv_filename (str, optional): The name of the csv file to save the results in.
                Defaults to ``"results.csv"``.

        Warning:
            You must define :attr:`optim_recipe` if you do not use the CLI.

        Note:
            :attr:`optim_recipe` can be anything that can be returned by
            :meth:`LightningModule.configure_optimizers()`. Find more details
            `here <https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers>`_.
        """
        super().__init__()
        _segmentation_routine_checks(
            num_classes=num_classes,
            metric_subsampling_rate=metric_subsampling_rate,
            num_bins_cal_err=num_bins_cal_err,
        )
        if eval_shift:
            raise NotImplementedError(
                "Distribution shift evaluation not implemented yet. Raise an issue if needed."
            )

        self.model = model
        self.num_classes = num_classes
        self.num_bins_cal_err = num_bins_cal_err
        self.loss = loss
        self.needs_epoch_update = isinstance(model, EPOCH_UPDATE_MODEL)
        self.needs_step_update = isinstance(model, STEP_UPDATE_MODEL)

        if format_batch_fn is None:
            format_batch_fn = nn.Identity()

        self.optim_recipe = optim_recipe
        self.format_batch_fn = format_batch_fn
        self.metric_subsampling_rate = metric_subsampling_rate
        self.log_plots = log_plots
        self.save_in_csv = save_in_csv
        self.csv_filename = csv_filename
        self.ood_criterion = get_ood_criterion(ood_criterion)
        self.eval_ood = eval_ood

        self.post_processing = post_processing
        if self.post_processing is not None:
            self.post_processing.set_model(self.model)

        self._init_metrics()

        if log_plots:
            self.num_samples_to_plot = num_samples_to_plot
            self.sample_buffer = []

    def _init_metrics(self) -> None:
        """Initialize the metrics depending on the exact task."""
        seg_metrics = MetricCollection(
            {
                "seg/mIoU": MeanIntersectionOverUnion(num_classes=self.num_classes),
                "seg/mAcc": Accuracy(
                    task="multiclass", average="macro", num_classes=self.num_classes
                ),
                "seg/pixAcc": Accuracy(task="multiclass", num_classes=self.num_classes),
            },
            compute_groups=[["seg/mIoU", "seg/mAcc", "seg/pixAcc"]],
        )
        sbsmpl_seg_metrics = MetricCollection(
            {
                "seg/Brier": BrierScore(num_classes=self.num_classes),
                "seg/NLL": CategoricalNLL(),
                "cal/ECE": CalibrationError(
                    task="multiclass",
                    num_classes=self.num_classes,
                    num_bins=self.num_bins_cal_err,
                ),
                "cal/aECE": CalibrationError(
                    task="multiclass",
                    adaptive=True,
                    num_classes=self.num_classes,
                    num_bins=self.num_bins_cal_err,
                ),
                "sc/AURC": AURC(),
                "sc/AUGRC": AUGRC(),
                "sc/Cov@5Risk": CovAt5Risk(),
                "sc/Risk@80Cov": RiskAt80Cov(),
            },
            compute_groups=[
                ["seg/Brier"],
                ["seg/NLL"],
                ["cal/ECE", "cal/aECE"],
                ["sc/AURC", "sc/AUGRC", "sc/Cov@5Risk", "sc/Risk@80Cov"],
            ],
        )

        self.val_seg_metrics = seg_metrics.clone(prefix="val/")
        self.val_sbsmpl_seg_metrics = sbsmpl_seg_metrics.clone(prefix="val/")
        self.test_seg_metrics = seg_metrics.clone(prefix="test/")
        self.test_sbsmpl_seg_metrics = sbsmpl_seg_metrics.clone(prefix="test/")

        if self.eval_ood:
            ood_metrics = MetricCollection(
                {
                    "AUROC": SegmentationBinaryAUROC(),
                    "AUPR": SegmentationBinaryAveragePrecision(),
                    "FPR95": SegmentationFPR95(pos_label=1),
                }
            )
            self.test_ood_metrics = ood_metrics.clone(prefix="ood/")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return self.optim_recipe

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            inputs (Tensor): input tensor.

        Returns:
            Tensor: the prediction of the model.
        """
        return self.model(inputs)

    def on_train_start(self) -> None:  # coverage: ignore
        if self.loss is None:
            raise ValueError(
                "To train a model, you must specify the `loss` argument in the routine. Got None."
            )
        if self.logger is not None:
            self.logger.log_hyperparams(self.hparams)

    def on_validation_start(self) -> None:
        if self.needs_epoch_update and not self.trainer.sanity_checking:
            self.model.update_wrapper(self.current_epoch)
            if hasattr(self.model, "need_bn_update"):
                self.model.bn_update(self.trainer.train_dataloader, device=self.device)

    def on_test_start(self) -> None:
        if self.post_processing is not None:
            with torch.inference_mode(False):
                self.post_processing.fit(self.trainer.datamodule.postprocess_dataloader())

        if self.eval_ood and self.log_plots and isinstance(self.logger, Logger):
            self.id_logit_storage = []
            self.ood_logit_storage = []

        if hasattr(self.model, "need_bn_update"):
            self.model.bn_update(self.trainer.train_dataloader, device=self.device)

        if self.num_params is None:
            self.num_params = sum(p.numel() for p in self.model.parameters())

    def training_step(self, batch: tuple[Tensor, Tensor]) -> STEP_OUTPUT:
        """Perform a single training step based on the input tensors.

        Args:
            batch (tuple[Tensor, Tensor]): the training images and their corresponding targets

        Returns:
            Tensor: the loss corresponding to this training step.
        """
        img, targets = self.format_batch_fn(batch)
        logits = self.forward(img)
        targets = F.resize(targets, logits.shape[-2:], interpolation=F.InterpolationMode.NEAREST)
        logits = rearrange(logits, "b c h w -> (b h w) c")
        targets = targets.flatten()
        valid_mask = (targets != 255) * (targets < self.num_classes)
        loss = self.loss(logits[valid_mask], targets[valid_mask])
        if self.needs_step_update:
            self.model.update_wrapper(self.current_epoch)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor]) -> None:
        """Perform a single validation step based on the input tensors.

        Compute the prediction of the model and the value of the metrics on the validation batch.

        Args:
            batch (tuple[Tensor, Tensor]): the validation images and their corresponding targets
        """
        img, targets = batch
        logits = self.forward(img)
        targets = F.resize(
            targets,
            logits.shape[-2:],
            interpolation=F.InterpolationMode.NEAREST,
        )
        logits = rearrange(logits, "(m b) c h w -> (b h w) m c", b=targets.size(0))
        probs_per_est = logits.softmax(dim=-1)
        probs = probs_per_est.mean(dim=1)
        targets = targets.flatten()
        valid_mask = (targets != 255) * (targets < self.num_classes)
        probs, targets = probs[valid_mask], targets[valid_mask]
        self.val_seg_metrics.update(probs, targets)
        self.val_sbsmpl_seg_metrics.update(*self._subsample(probs, targets))

    def test_step(
        self,
        batch: tuple[Tensor, Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Perform a single test step based on the input tensors.

        Compute the prediction of the model and the value of the metrics on the test batch.

        Args:
            batch (tuple[Tensor, Tensor]): the test images and their corresponding targets
            batch_idx (int): the index of the batch in the test dataloader.
            dataloader_idx (int, optional): the index of the dataloader. Defaults to ``0``.
        """
        img, targets = batch

        if self.test_num_flops is None:
            flop_counter = FlopCounterMode(display=False)
            with flop_counter:
                self.forward(img)
            self.test_num_flops = flop_counter.get_total_flops()

        logits = self.forward(img)
        targets = F.resize(
            targets,
            logits.shape[-2:],
            interpolation=F.InterpolationMode.NEAREST,
        )

        logits = rearrange(logits, "(m b) c h w -> b m c h w", b=targets.size(0))
        probs_per_est = logits.softmax(dim=2)
        probs = probs_per_est.mean(dim=1)

        if self.log_plots and len(self.sample_buffer) < self.num_samples_to_plot:
            max_count = self.num_samples_to_plot - len(self.sample_buffer)
            for i, (_img, _prb, _tgt) in enumerate(zip(img, probs, targets, strict=True)):
                if i >= max_count:
                    break
                _pred = _prb.argmax(dim=0, keepdim=True)
                self.sample_buffer.append((_img, _pred, _tgt))

        probs_per_est = rearrange(probs_per_est, "b m c h w -> (b h w) m c")
        probs = rearrange(probs, "b c h w -> (b h w) c")
        targets = targets.flatten()
        valid_mask = targets != 255
        probs, probs_per_est, targets = (
            probs[valid_mask],
            probs_per_est[valid_mask],
            targets[valid_mask],
        )
        id_mask = targets < self.num_classes
        ood_mask = targets >= self.num_classes

        if dataloader_idx == 0:
            id_probs, _, id_targets = probs[id_mask], probs_per_est[id_mask], targets[id_mask]
            self.test_seg_metrics.update(id_probs, id_targets)
            self.test_sbsmpl_seg_metrics.update(*self._subsample(id_probs, id_targets))

        if self.eval_ood and dataloader_idx == 1:
            if self.ood_criterion.input_type == OODCriterionInputType.PROB:
                ood_scores = self.ood_criterion(probs)
            elif self.ood_criterion.input_type == OODCriterionInputType.ESTIMATOR_PROB:
                ood_scores = self.ood_criterion(probs_per_est)
            else:
                raise ValueError(
                    f"Unsupported input type for OOD criterion: {self.ood_criterion.input_type}"
                )

            labels = torch.zeros_like(targets)
            labels[id_mask] = 0  # ID examples
            labels[ood_mask] = 1  # OOD examples

            self.test_ood_metrics.update(ood_scores, labels)

    def on_validation_epoch_end(self) -> None:
        """Compute and log the values of the collected metrics in `validation_step`."""
        res_dict = self.val_seg_metrics.compute()
        self.log_dict(res_dict, logger=True, sync_dist=True)
        self.log(
            "mIoU%",
            res_dict["val/seg/mIoU"] * 100,
            prog_bar=True,
            sync_dist=True,
        )
        self.log_dict(self.val_sbsmpl_seg_metrics.compute(), sync_dist=True)
        self.val_seg_metrics.reset()
        self.val_sbsmpl_seg_metrics.reset()

    def on_test_epoch_end(self) -> None:
        """Compute, log, and plot the values of the collected metrics in `test_step`."""
        result_dict = self.test_seg_metrics.compute()
        result_dict |= self.test_sbsmpl_seg_metrics.compute()
        result_dict |= {
            "test/cplx/flops": self.test_num_flops,
            "test/cplx/params": self.num_params,
        }
        if self.eval_ood:
            result_dict |= self.test_ood_metrics.compute()

        self.log_dict(result_dict, logger=True, sync_dist=True)

        if isinstance(self.logger, Logger) and self.log_plots:
            self._plot_results()

        self.test_seg_metrics.reset()
        self.test_sbsmpl_seg_metrics.reset()
        if self.eval_ood:
            self.test_ood_metrics.reset()

        if self.save_in_csv and self.logger is not None:
            csv_writer(
                Path(self.logger.log_dir) / self.csv_filename,
                result_dict,
            )

    def _plot_results(self):
        """Plot uncertainty quantification metrics and segmentation figures."""
        self.logger.experiment.add_figure(
            "Calibration/Reliabity diagram",
            self.test_sbsmpl_seg_metrics["cal/ECE"].plot()[0],
        )
        self.logger.experiment.add_figure(
            "Selective Classification/Risk-Coverage curve",
            self.test_sbsmpl_seg_metrics["sc/AURC"].plot()[0],
        )
        self.logger.experiment.add_figure(
            "Selective Classification/Generalized Risk-Coverage curve",
            self.test_sbsmpl_seg_metrics["sc/AUGRC"].plot()[0],
        )
        if self.trainer.datamodule is not None:
            self._log_segmentation_plots()
        else:
            logging.info("No datamodule found, skipping segmentation plots.")

    def _log_segmentation_plots(self) -> None:
        """Build and log examples of segmentation plots from the test set."""
        for i, (img, pred, tgt) in enumerate(self.sample_buffer):
            pred = pred == torch.arange(self.num_classes, device=pred.device)[:, None, None]
            tgt = tgt == torch.arange(self.num_classes, device=tgt.device)[:, None, None]

            # Undo normalization on the image and convert to uint8.
            mean = torch.tensor(self.trainer.datamodule.mean, device=img.device)
            std = torch.tensor(self.trainer.datamodule.std, device=img.device)
            img = img * std[:, None, None] + mean[:, None, None]
            img = ToDtype(torch.uint8, scale=True)(img)

            dataset = self.trainer.datamodule.test
            color_palette = dataset.color_palette if hasattr(dataset, "color_palette") else None

            pred_mask = draw_segmentation_masks(img, pred, alpha=0.7, colors=color_palette)
            gt_mask = draw_segmentation_masks(img, tgt, alpha=0.7, colors=color_palette)

            self.logger.experiment.add_figure(
                f"Segmentation results/{i}",
                show_segmentation_predictions(pred_mask, gt_mask),
            )

    def _subsample(self, pred: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        """Select a random sample of the data to compute the loss onto.

        Args:
            pred (Tensor): the prediction tensor.
            target (Tensor): the target tensor.

        Returns:
            Tuple[Tensor, Tensor]: the subsampled prediction and target tensors.
        """
        total_size = target.size(0)
        num_samples = max(1, int(total_size * self.metric_subsampling_rate))
        indices = torch.randperm(total_size, device=pred.device)[:num_samples]
        return pred[indices], target[indices]


def _segmentation_routine_checks(
    num_classes: int,
    metric_subsampling_rate: float,
    num_bins_cal_err: int,
) -> None:
    """Check the domains of the routine's parameters.

    Args:
        num_classes (int): the number of classes in the dataset.
        metric_subsampling_rate (float): the rate of subsampling to compute the metrics.
        num_bins_cal_err (int): the number of bins for the evaluation of the calibration.
    """
    if num_classes < 2:
        raise ValueError(f"num_classes must be at least 2, got {num_classes}.")

    if not 0 < metric_subsampling_rate <= 1:
        raise ValueError(
            f"metric_subsampling_rate must be in the range (0, 1], got {metric_subsampling_rate}."
        )

    if num_bins_cal_err < 2:
        raise ValueError(f"num_bins_cal_err must be at least 2, got {num_bins_cal_err}.")
