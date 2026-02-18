from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.flop_counter import FlopCounterMode
from torchmetrics import Accuracy, MetricCollection
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
)

from torch_uncertainty.losses import DECLoss, ELBOLoss
from torch_uncertainty.metrics import (
    AUGRC,
    AURC,
    FPR95,
    BrierScore,
    CalibrationError,
    CategoricalNLL,
    CovAt5Risk,
    CoverageRate,
    Disagreement,
    Entropy,
    GroupingLoss,
    MutualInformation,
    RiskAt80Cov,
    SetSize,
)
from torch_uncertainty.models import (
    EPOCH_UPDATE_MODEL,
    STEP_UPDATE_MODEL,
)
from torch_uncertainty.ood_criteria import (
    OODCriterionInputType,
    PostProcessingCriterion,
    TUOODCriterion,
    get_ood_criterion,
)
from torch_uncertainty.post_processing import Conformal, LaplaceApprox, PostProcessing
from torch_uncertainty.transforms import MIXUP_PARAMS, RepeatTarget, build_mixup
from torch_uncertainty.utils import csv_writer, plot_hist


class ClassificationRoutine(LightningModule):
    test_num_flops: int | None = None
    num_params: int | None = None

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        loss: nn.Module | None = None,
        *,
        is_ensemble: bool = False,
        num_tta: int = 1,
        format_batch_fn: nn.Module | None = None,
        optim_recipe: dict | Optimizer | None = None,
        mixup_params: dict | None = None,
        eval_ood: bool = False,
        eval_shift: bool = False,
        eval_grouping_loss: bool = False,
        ood_criterion: TUOODCriterion | str = "msp",
        post_processing: PostProcessing | None = None,
        num_bins_cal_err: int = 15,
        log_plots: bool = False,
        save_in_csv: bool = False,
        csv_filename: str = "results.csv",
    ) -> None:
        r"""Routine for training & testing on **classification** tasks.

        Args:
            model (torch.nn.Module): Model to train.
            num_classes (int): Number of classes.
            loss (torch.nn.Module): Loss function to optimize the :attr:`model`.
                Defaults to ``None``.
            is_ensemble (bool, optional): Indicates whether the model is an
                ensemble at test time or not. Defaults to ``False``.
            num_tta (int): Number of test-time augmentations (TTA). If ``1``: no TTA.
                Defaults to ``1``.
            format_batch_fn (torch.nn.Module, optional): Function to format the batch.
                Defaults to ``None``.
            optim_recipe (dict or torch.optim.Optimizer, optional): The optimizer and
                optionally the scheduler to use. Defaults to ``None``.
            mixup_params (dict, optional): Mixup parameters. Can include mixup type,
                mixup mode, distance similarity, kernel tau max, kernel tau std,
                mixup alpha, and cutmix alpha. If None, no mixup augmentations.
                Defaults to ``None``.
            eval_ood (bool, optional): Indicates whether to evaluate the OOD
                detection performance. Defaults to ``False``.
            eval_shift (bool, optional): Indicates whether to evaluate the Distribution
                shift performance. Defaults to ``False``.
            eval_grouping_loss (bool, optional): Indicates whether to evaluate the
                grouping loss or not. Defaults to ``False``.
            ood_criterion (TUOODCriterion | str, optional): Criterion for the binary OOD detection
                task. Defaults to ``msp``, the Maximum Softmax Probability score.
            post_processing (PostProcessing, optional): Post-processing method
                to train on the calibration set. No post-processing if None.
                Defaults to ``None``.
            num_bins_cal_err (int, optional): Number of bins to compute calibration
                error metrics. Defaults to ``15``.
            log_plots (bool, optional): Indicates whether to log plots from
                metrics. Defaults to ``False``.
            save_in_csv (bool, optional): Save the results in csv. Defaults to
                ``False``.
            csv_filename (str, optional): Name of the csv file. Defaults to
                ``"results.csv"``. Note that this is only used if
                :attr:`save_in_csv` is ``True``.

        Warning:
            You must define :attr:`optim_recipe` if you do not use the Lightning CLI.

        Warning:
            When using an ensemble model, you must:
            1. Set :attr:`is_ensemble` to ``True``.
            2. Set :attr:`format_batch_fn` to :class:`torch_uncertainty.transforms.RepeatTarget(num_repeats=num_estimators)`.
            3. Ensure that the model's forward pass outputs a tensor of shape :math:`(M \times B, C)`,
            where :math:`M` is the number of estimators, :math:`B` is the batch size, :math:`C` is the number of classes.

            For automated batch handling, consider using the available model wrappers in `torch_uncertainty.models.wrappers`.

        Note:
            If :attr:`eval_ood` is ``True``, we perform a binary classification and update the
            OOD-related metrics twice:
            - once during the test on ID values where the given binary label is 0 (for ID)
            - once during the test on OOD values where the given binary label is 1 (for OOD)

        Note:
            :attr:`optim_recipe` can be anything that can be returned by
            :meth:`LightningModule.configure_optimizers()`. Find more details
            `here <https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers>`_.
        """
        super().__init__()
        _classification_routine_checks(
            model=model,
            num_classes=num_classes,
            is_ensemble=is_ensemble,
            ood_criterion=ood_criterion,
            eval_grouping_loss=eval_grouping_loss,
            num_bins_cal_err=num_bins_cal_err,
            mixup_params=mixup_params,
            post_processing=post_processing,
            format_batch_fn=format_batch_fn,
        )

        if format_batch_fn is None:
            format_batch_fn = nn.Identity()

        self.num_classes = num_classes
        self.eval_ood = eval_ood
        self.eval_shift = eval_shift
        self.eval_grouping_loss = eval_grouping_loss
        self.num_tta = num_tta
        self.ood_criterion = get_ood_criterion(ood_criterion)
        self.log_plots = log_plots
        self.save_in_csv = save_in_csv
        self.csv_filename = csv_filename
        self.binary_cls = num_classes == 1
        self.needs_epoch_update = isinstance(model, EPOCH_UPDATE_MODEL)
        self.needs_step_update = isinstance(model, STEP_UPDATE_MODEL)
        self.num_bins_cal_err = num_bins_cal_err
        self.model = model
        self.loss = loss
        self.format_batch_fn = format_batch_fn
        self.optim_recipe = optim_recipe
        self.is_ensemble = is_ensemble

        self.post_processing = post_processing
        if self.post_processing is not None:
            self.post_processing.set_model(self.model)

        self._init_metrics()
        self.mixup = self._init_mixup(mixup_params)

        self.is_elbo = isinstance(self.loss, ELBOLoss)
        if self.is_elbo:
            self.loss.set_model(self.model)
        self.is_dec = isinstance(self.loss, DECLoss)

        self.id_score_storage = None
        self.ood_score_storage = None

    def _init_metrics(self) -> None:
        """Initialize the metrics depending on the exact task."""
        task = "binary" if self.binary_cls else "multiclass"

        metrics_dict = {
            "cls/Acc": Accuracy(task=task, num_classes=self.num_classes),
            "cls/Brier": BrierScore(num_classes=self.num_classes),
            "cls/NLL": CategoricalNLL(),
            "cal/ECE": CalibrationError(
                task=task,
                num_bins=self.num_bins_cal_err,
                num_classes=self.num_classes,
            ),
            "cal/aECE": CalibrationError(
                task=task,
                adaptive=True,
                num_bins=self.num_bins_cal_err,
                num_classes=self.num_classes,
            ),
            "sc/AURC": AURC(),
            "sc/AUGRC": AUGRC(),
            "sc/Cov@5Risk": CovAt5Risk(),
            "sc/Risk@80Cov": RiskAt80Cov(),
        }
        groups = [
            ["cls/Acc"],
            ["cls/Brier"],
            ["cls/NLL"],
            ["cal/ECE", "cal/aECE"],
            ["sc/AURC", "sc/AUGRC", "sc/Cov@5Risk", "sc/Risk@80Cov"],
        ]

        if self.binary_cls:
            metrics_dict |= {
                "cls/AUROC": BinaryAUROC(),
                "cls/AUPR": BinaryAveragePrecision(),
                "cls/FRP95": FPR95(pos_label=1),
            }
            groups.extend([["cls/AUROC", "cls/AUPR"], ["cls/FRP95"]])

        cls_metrics = MetricCollection(metrics_dict, compute_groups=groups)
        self.val_cls_metrics = cls_metrics.clone(prefix="val/")

        self.test_cls_metrics = cls_metrics.clone(prefix="test/")

        if self.post_processing is not None and isinstance(self.post_processing, Conformal):
            self.post_cls_metrics = MetricCollection(
                {
                    "test/post/CoverageRate": CoverageRate(),
                    "test/post/SetSize": SetSize(),
                },
            )
        elif self.post_processing is not None:
            self.post_cls_metrics = cls_metrics.clone(prefix="test/post/")

        self.test_id_entropy = Entropy()

        if self.eval_ood:
            ood_metrics = MetricCollection(
                {
                    "AUROC": BinaryAUROC(),
                    "AUPR": BinaryAveragePrecision(),
                    "FPR95": FPR95(pos_label=1),
                },
                compute_groups=[["AUROC", "AUPR"], ["FPR95"]],
            )
            self.test_ood_metrics = ood_metrics.clone(prefix="ood/")
            self.test_ood_entropy = Entropy()

        if self.eval_shift:
            self.test_shift_metrics = cls_metrics.clone(prefix="shift/")

        # metrics for ensembles only
        if self.is_ensemble:
            ens_metrics = MetricCollection(
                {
                    "Disagreement": Disagreement(),
                    "MI": MutualInformation(),
                    "Entropy": Entropy(),
                }
            )

            self.test_id_ens_metrics = ens_metrics.clone(prefix="test/ens_")

            if self.eval_ood:
                self.test_ood_ens_metrics = ens_metrics.clone(prefix="ood/ens_")

            if self.eval_shift:
                self.test_shift_ens_metrics = ens_metrics.clone(prefix="shift/ens_")

        if self.eval_grouping_loss:
            grouping_loss = MetricCollection({"cls/grouping_loss": GroupingLoss()})
            self.val_grouping_loss = grouping_loss.clone(prefix="val/")
            self.test_grouping_loss = grouping_loss.clone(prefix="test/")

    def _init_mixup(self, mixup_params: dict | None) -> Callable:
        """Setup the optional mixup augmentation based on the :attr:`mixup_params` dict.

        Args:
            mixup_params (dict | None): the detailed parameters of the mixup augmentation. None if
                unused.
        """
        if mixup_params is None:
            mixup_params = {}
        mixup_params = MIXUP_PARAMS | mixup_params
        self.has_kernel_warping_mixup = mixup_params["mixtype"] == "kernel_warping"
        self.kw_on_embeddings = mixup_params.pop("kw_on_embeddings")
        return build_mixup(num_classes=self.num_classes, **mixup_params)

    def _apply_mixup(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        """Apply the mixup augmentation on a :attr:`batch` of images.

        Args:
            batch (tuple[Tensor, Tensor]): the images and the corresponding targets.

        Returns:
            tuple[Tensor, Tensor]: the images and the corresponding targets transformed with mixup.
        """
        if not self.is_ensemble:
            if self.has_kernel_warping_mixup:
                if self.kw_on_embeddings:
                    with torch.no_grad():
                        feats = self.model.feats_forward(batch[0]).detach()
                    batch = self.mixup(*batch, feats)
                else:
                    batch = self.mixup(*batch, batch[0])
            else:
                batch = self.mixup(*batch)
        return batch

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

        Setup the post-processing dataset and fit the post-processing method if needed, prepares
        the storage lists for logit plotting and update the batchnorms if needed.
        """
        if self.post_processing is not None:
            with torch.inference_mode(False):
                self.post_processing.fit(self.trainer.datamodule.postprocess_dataloader())

        if self.eval_ood and self.log_plots and isinstance(self.logger, Logger):
            self.id_score_storage = []
            self.ood_score_storage = []

        if hasattr(self.model, "need_bn_update"):
            self.model.bn_update(self.trainer.train_dataloader, device=self.device)

        if self.num_params is None:
            self.num_params = sum(p.numel() for p in self.model.parameters())

    def forward(self, inputs: Tensor, save_feats: bool = False) -> Tensor:
        """Forward pass of the inner model.

        Args:
            inputs (Tensor): input tensor.
            save_feats (bool, optional): whether to store the features or
                not. Defaults to ``False``.

        Note:
            The features are stored in the :attr:`self.features` attribute.
        """
        if save_feats:
            self.features = self.model.feats_forward(inputs)
            if hasattr(self.model, "classification_head"):  # coverage: ignore
                logits = self.model.classification_head(self.features)
            else:
                logits = self.model.linear(self.features)
        else:
            self.features = None
            logits = self.model(inputs)
        return logits

    def training_step(self, batch: tuple[Tensor, Tensor]) -> STEP_OUTPUT:
        """Perform a single training step based on the input tensors.

        Args:
            batch (tuple[Tensor, Tensor]): the training data and their corresponding targets.

        Returns:
            Tensor: the loss corresponding to this training step.
        """
        batch = self._apply_mixup(batch)
        inputs, target = self.format_batch_fn(batch)

        if self.is_elbo:
            loss = self.loss(inputs, target)
        else:
            logits = self.forward(inputs)
            # BCEWithLogitsLoss expects float target
            if self.binary_cls and isinstance(self.loss, nn.BCEWithLogitsLoss):
                logits = logits.squeeze(-1)
                target = target.float()

            if not self.is_dec:
                loss = self.loss(logits, target)
            else:
                loss = self.loss(logits, target, self.current_epoch)
        if self.needs_step_update:
            self.model.update_wrapper(self.current_epoch)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor]) -> None:
        """Perform a single validation step based on the input tensors.

        Compute the prediction of the model and the value of the metrics on the validation batch.

        Args:
            batch (tuple[Tensor, Tensor]): the validation data and their corresponding targets
        """
        inputs, targets = batch
        # remove duplicates when doing TTA
        targets = targets[:: self.num_tta]
        logits = self.forward(inputs, save_feats=self.eval_grouping_loss)
        logits = rearrange(logits, "(m b) c -> b m c", b=targets.size(0))

        if self.binary_cls:
            probs_per_est = torch.sigmoid(logits).squeeze(-1)
        else:
            probs_per_est = F.softmax(logits, dim=-1)

        probs = probs_per_est.mean(dim=1)
        self.val_cls_metrics.update(probs, targets)

        if self.eval_grouping_loss:
            self.val_grouping_loss.update(probs, targets, self.features)

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
            dataloader_idx (int): 0 if in-distribution, 1 if out-of-distribution and 2 if
                distribution-shifted.
        """
        inputs, targets = batch

        if self.test_num_flops is None:
            flop_counter = FlopCounterMode(display=False)
            with flop_counter:
                self.forward(inputs)
            self.test_num_flops = flop_counter.get_total_flops()

        # remove duplicates when doing TTA
        targets = targets[:: self.num_tta]
        logits = self.forward(inputs, save_feats=self.eval_grouping_loss)
        logits = rearrange(logits, "(m b) c -> b m c", b=targets.size(0))
        probs_per_est = torch.sigmoid(logits) if self.binary_cls else F.softmax(logits, dim=-1)
        probs = probs_per_est.mean(dim=1)

        if self.post_processing is not None:
            pp_logits = self.post_processing(inputs)
            if isinstance(self.post_processing, LaplaceApprox | Conformal):
                pp_probs = pp_logits
            else:
                pp_probs = F.softmax(pp_logits, dim=-1)

        if self.ood_criterion.input_type == OODCriterionInputType.LOGIT:
            ood_scores = self.ood_criterion(logits)
        elif self.ood_criterion.input_type == OODCriterionInputType.PROB:
            ood_scores = self.ood_criterion(probs)
        elif self.ood_criterion.input_type == OODCriterionInputType.ESTIMATOR_PROB:
            ood_scores = self.ood_criterion(probs_per_est)
        elif self.ood_criterion.input_type == OODCriterionInputType.POST_PROCESSING:
            ood_scores = self.ood_criterion(pp_probs)

        if dataloader_idx == 0:
            # squeeze if binary classification only for binary metrics
            self.test_cls_metrics.update(
                probs.squeeze(-1) if self.binary_cls else probs,
                targets,
            )
            self.test_id_entropy.update(probs)

            if self.eval_grouping_loss:
                self.test_grouping_loss.update(probs, targets, self.features)

            if self.is_ensemble:
                self.test_id_ens_metrics.update(probs_per_est)

            if self.eval_ood:
                self.test_ood_entropy.update(probs)
                self.test_ood_metrics.update(ood_scores, torch.zeros_like(targets))

            if self.id_score_storage is not None:
                self.id_score_storage.append(ood_scores.detach().cpu())

            if self.post_processing is not None:
                self.post_cls_metrics.update(pp_probs, targets)

        if self.eval_ood and dataloader_idx == 1:
            self.test_ood_metrics.update(ood_scores, torch.ones_like(targets))

            if self.is_ensemble:
                self.test_ood_ens_metrics.update(probs_per_est)

            if self.ood_score_storage is not None:
                self.ood_score_storage.append(ood_scores.detach().cpu())

        if self.eval_shift and dataloader_idx == (2 if self.eval_ood else 1):
            self.test_shift_metrics.update(probs, targets)
            if self.is_ensemble:
                self.test_shift_ens_metrics.update(probs_per_est)

    def on_validation_epoch_end(self) -> None:
        """Compute and log the values of the collected metrics in `validation_step`."""
        res_dict = self.val_cls_metrics.compute()
        self.log_dict(res_dict, logger=True, sync_dist=True)
        # Progress bar only
        self.log(
            "Acc",
            res_dict["val/cls/Acc"] * 100,
            prog_bar=True,
            logger=False,
            sync_dist=True,
        )
        self.val_cls_metrics.reset()

        if self.eval_grouping_loss:
            self.log_dict(self.val_grouping_loss.compute(), sync_dist=True)
            self.val_grouping_loss.reset()

    def on_test_epoch_end(self) -> None:
        """Compute, log, and plot the values of the collected metrics in `test_step`."""
        result_dict = self.test_cls_metrics.compute() | {
            "test/cls/Entropy": self.test_id_entropy.compute(),
            "test/cplx/flops": self.test_num_flops,
            "test/cplx/params": self.num_params,
        }

        if self.post_processing is not None:
            result_dict |= self.post_cls_metrics.compute()

        if self.eval_grouping_loss:
            result_dict |= self.test_grouping_loss.compute()

        if self.is_ensemble:
            result_dict |= self.test_id_ens_metrics.compute()

        if self.eval_ood:
            result_dict |= self.test_ood_metrics.compute() | {
                "ood/Entropy": self.test_ood_entropy.compute()
            }
            if self.is_ensemble:
                result_dict |= self.test_ood_ens_metrics.compute()

        if self.eval_shift:
            result_dict |= self.test_shift_metrics.compute() | {
                "shift/severity": self.trainer.datamodule.shift_severity,
            }

            if self.is_ensemble:
                result_dict |= self.test_shift_ens_metrics.compute()

        self.log_dict(result_dict, sync_dist=True)

        if isinstance(self.logger, Logger) and self.log_plots:
            self._plot_results()

        # reset metrics
        self.test_cls_metrics.reset()
        self.test_id_entropy.reset()
        if self.post_processing is not None:
            self.post_cls_metrics.reset()
        if self.eval_grouping_loss:
            self.test_grouping_loss.reset()
        if self.is_ensemble:
            self.test_id_ens_metrics.reset()
        if self.eval_ood:
            self.test_ood_metrics.reset()
            self.test_ood_entropy.reset()
            if self.is_ensemble:
                self.test_ood_ens_metrics.reset()
        if self.eval_shift:
            self.test_shift_metrics.reset()
            if self.is_ensemble:
                self.test_shift_ens_metrics.reset()

        if self.save_in_csv and self.logger is not None:
            csv_writer(
                Path(self.logger.log_dir) / self.csv_filename,
                result_dict,
            )

    def _plot_results(self):
        """Plot uncertainty quantification metrics."""
        self.logger.experiment.add_figure(
            "Reliabity diagram", self.test_cls_metrics["cal/ECE"].plot()[0]
        )
        self.logger.experiment.add_figure(
            "Risk-Coverage curve",
            self.test_cls_metrics["sc/AURC"].plot()[0],
        )
        self.logger.experiment.add_figure(
            "Generalized Risk-Coverage curve",
            self.test_cls_metrics["sc/AUGRC"].plot()[0],
        )

        if self.post_processing is not None and not isinstance(self.post_processing, Conformal):
            self.logger.experiment.add_figure(
                "Reliabity diagram after calibration",
                self.post_cls_metrics["cal/ECE"].plot()[0],
            )

        # plot histograms of logits and likelihoods
        if self.eval_ood:
            id_scores = torch.cat(self.id_score_storage, dim=0)
            ood_scores = torch.cat(self.ood_score_storage, dim=0)

            score_fig = plot_hist(
                [id_scores, ood_scores],
                20,
                "Histogram of the OOD scores",
            )[0]
            self.logger.experiment.add_figure("OOD Score Histogram", score_fig)


def _classification_routine_checks(
    model: nn.Module,
    num_classes: int,
    is_ensemble: bool,
    ood_criterion: TUOODCriterion | str,
    eval_grouping_loss: bool,
    num_bins_cal_err: int,
    mixup_params: dict | None,
    post_processing: PostProcessing | None,
    format_batch_fn: nn.Module | None,
) -> None:
    """Check the domains of the arguments of the classification routine.

    Args:
        model (nn.Module): the model used to make classification predictions.
        num_classes (int): the number of classes in the dataset.
        is_ensemble (bool): whether the model is an ensemble or a single model.
        ood_criterion (TUOODCriterion, optional): OOD criterion for the binary OOD detection task.
        eval_grouping_loss (bool): whether to evaluate the grouping loss.
        num_bins_cal_err (int): the number of bins for the evaluation of the calibration.
        mixup_params (dict | None): the dictionary to setup the mixup augmentation.
        post_processing (PostProcessing | None): the post-processing module.
        format_batch_fn (nn.Module | None): the function for formatting the batch for ensembles.
    """
    ood_criterion = get_ood_criterion(ood_criterion)
    if not is_ensemble and ood_criterion.ensemble_only:
        raise ValueError(
            "You cannot use mutual information or variation ratio with a single model."
        )

    if is_ensemble and ood_criterion.single_only:
        raise NotImplementedError(
            "Logit-based criteria are not implemented for ensembles. Raise an issue if needed."
        )

    if isinstance(ood_criterion, PostProcessingCriterion) and post_processing is None:
        raise ValueError(
            "You cannot set ood_criterion=PostProcessingCriterion when post_processing is None."
        )

    if is_ensemble and eval_grouping_loss:
        raise NotImplementedError(
            "Grouping loss for ensembles is not yet implemented. Raise an issue if needed."
        )

    if num_classes < 1:
        raise ValueError(
            f"The number of classes must be a positive integer >= 1. Got {num_classes}."
        )

    if eval_grouping_loss and not hasattr(model, "feats_forward"):
        raise ValueError(
            "Your model must have a `feats_forward` method to compute the grouping loss."
        )

    if eval_grouping_loss and not (
        hasattr(model, "classification_head") or hasattr(model, "linear")
    ):
        raise ValueError(
            "Your model must have a `classification_head` or `linear` "
            "attribute to compute the grouping loss."
        )

    if num_bins_cal_err < 2:
        raise ValueError(f"num_bins_cal_err must be at least 2, got {num_bins_cal_err}.")

    if mixup_params is not None and isinstance(format_batch_fn, RepeatTarget):
        raise ValueError(
            "Mixup is not supported for ensembles at training time. Please set mixup_params to None."
        )

    if post_processing is not None and is_ensemble:
        raise ValueError(
            "Ensembles and post-processing methods cannot be used together. Raise an issue if needed."
        )
