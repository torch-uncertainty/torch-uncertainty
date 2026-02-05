import copy

from torch import nn

from torch_uncertainty.models import EMA, SWA, deep_ensembles
from torch_uncertainty.ood_criteria import TUOODCriterion
from torch_uncertainty.optim_recipes import optim_cifar10_resnet18
from torch_uncertainty.post_processing import TemperatureScaler
from torch_uncertainty.routines import (
    ClassificationRoutine,
    PixelRegressionRoutine,
    RegressionRoutine,
    SegmentationRoutine,
)
from torch_uncertainty.transforms import RepeatTarget

from .model import dummy_model, dummy_segmentation_model


class DummyClassificationBaseline:
    def __new__(
        cls,
        num_classes: int,
        in_channels: int,
        loss: nn.Module,
        baseline_type: str = "single",
        optim_recipe=optim_cifar10_resnet18,
        with_feats: bool = True,
        ood_criterion: TUOODCriterion | str = "msp",
        eval_ood: bool = False,
        eval_shift: bool = False,
        eval_grouping_loss: bool = False,
        calibrate: bool = False,
        save_in_csv: bool = False,
        mixtype: str = "mixup",
        isobatch: bool = False,
        kw_on_embeddings: bool = True,
        kernel_tau_max: float = 1,
        kernel_tau_std: float = 0.5,
        mixup_alpha: float = 1,
        cutmix_alpha: float = 0,
        mixup_ratio: float = 0,
        no_mixup_params: bool = False,
        ema: bool = False,
        swa: bool = False,
    ) -> ClassificationRoutine:
        model = dummy_model(
            in_channels=in_channels,
            num_classes=num_classes,
            with_feats=with_feats,
        )
        if ema:
            model = EMA(model, momentum=0.99)
        if swa:
            model = SWA(model, cycle_start=0, cycle_length=1)
        if not no_mixup_params:
            mixup_params = {
                "mixup_alpha": mixup_alpha,
                "cutmix_alpha": cutmix_alpha,
                "mixtype": mixtype,
                "isobatch": isobatch,
                "kw_on_embeddings": kw_on_embeddings,
                "kernel_tau_max": kernel_tau_max,
                "kernel_tau_std": kernel_tau_std,
                "mixupmp_ratio": mixup_ratio,
            }
        else:
            mixup_params = None
        if baseline_type == "single":
            return ClassificationRoutine(
                num_classes=num_classes,
                model=model,
                loss=loss,
                format_batch_fn=nn.Identity(),
                log_plots=True,
                optim_recipe=optim_recipe(model),
                is_ensemble=False,
                mixup_params=mixup_params,
                ood_criterion=ood_criterion,
                eval_ood=eval_ood,
                eval_shift=eval_shift,
                eval_grouping_loss=eval_grouping_loss,
                post_processing=TemperatureScaler() if calibrate else None,
                save_in_csv=save_in_csv,
            )
        # baseline_type == "ensemble":
        model = deep_ensembles(
            [model, copy.deepcopy(model)],
            task="classification",
        )
        return ClassificationRoutine(
            num_classes=num_classes,
            model=model,
            loss=loss,
            optim_recipe=optim_recipe(model),
            format_batch_fn=RepeatTarget(2),
            log_plots=True,
            is_ensemble=True,
            ood_criterion=ood_criterion,
            eval_ood=eval_ood,
            eval_shift=eval_shift,
            eval_grouping_loss=eval_grouping_loss,
            post_processing=TemperatureScaler() if calibrate else None,
            save_in_csv=save_in_csv,
        )


class DummyRegressionBaseline:
    def __new__(
        cls,
        in_features: int,
        output_dim: int,
        loss: nn.Module,
        baseline_type: str = "single",
        optim_recipe=None,
        dist_family: str | None = "normal",
        ema: bool = False,
        swa: bool = False,
    ) -> RegressionRoutine:
        model = dummy_model(
            in_channels=in_features,
            num_classes=output_dim,
            dist_family=dist_family,
        )
        if ema:
            model = EMA(model, momentum=0.99)
        if swa:
            model = SWA(model, cycle_start=0, cycle_length=1)

        if baseline_type == "single":
            return RegressionRoutine(
                output_dim=output_dim,
                model=model,
                loss=loss,
                optim_recipe=optim_recipe(model),
                dist_family=dist_family,
                log_plots=True,
            )
        # baseline_type == "ensemble":
        model = deep_ensembles(
            [model, copy.deepcopy(model)],
            task="regression",
            probabilistic=dist_family is not None,
        )
        return RegressionRoutine(
            output_dim=output_dim,
            model=model,
            loss=loss,
            is_ensemble=True,
            optim_recipe=optim_recipe(model),
            format_batch_fn=RepeatTarget(2),
            dist_family=dist_family,
            log_plots=True,
        )


class DummySegmentationBaseline:
    def __new__(
        cls,
        in_channels: int,
        num_classes: int,
        image_size: int,
        loss: nn.Module,
        baseline_type: str = "single",
        optim_recipe=None,
        metric_subsampling_rate: float = 1,
        log_plots: bool = False,
        ema: bool = False,
        swa: bool = False,
    ) -> SegmentationRoutine:
        model = dummy_segmentation_model(
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=image_size,
        )
        if ema:
            model = EMA(model, momentum=0.99)
        if swa:
            model = SWA(model, cycle_start=0, cycle_length=2)

        if baseline_type == "single":
            return SegmentationRoutine(
                num_classes=num_classes,
                model=model,
                loss=loss,
                format_batch_fn=None,
                optim_recipe=optim_recipe(model),
                metric_subsampling_rate=metric_subsampling_rate,
                log_plots=log_plots,
            )

        # baseline_type == "ensemble":
        model = deep_ensembles(
            [model, copy.deepcopy(model)],
            task="segmentation",
        )
        return SegmentationRoutine(
            num_classes=num_classes,
            model=model,
            loss=loss,
            format_batch_fn=RepeatTarget(2),
            optim_recipe=optim_recipe(model),
            metric_subsampling_rate=metric_subsampling_rate,
            log_plots=log_plots,
        )


class DummyPixelRegressionBaseline:
    def __new__(
        cls,
        in_channels: int,
        output_dim: int,
        image_size: int,
        loss: nn.Module,
        dist_family: str = "normal",
        baseline_type: str = "single",
        optim_recipe=None,
        ema: bool = False,
        swa: bool = False,
    ) -> PixelRegressionRoutine:
        model = dummy_segmentation_model(
            num_classes=output_dim,
            in_channels=in_channels,
            image_size=image_size,
            dist_family=dist_family,
        )
        if ema:
            model = EMA(model, momentum=0.99)
        if swa:
            model = SWA(model, cycle_start=0, cycle_length=1)

        if baseline_type == "single":
            return PixelRegressionRoutine(
                output_dim=output_dim,
                model=model,
                loss=loss,
                optim_recipe=optim_recipe(model),
                dist_family=dist_family,
            )

        # baseline_type == "ensemble":
        model = deep_ensembles(
            [model, copy.deepcopy(model)],
            task="pixel_regression",
            probabilistic=dist_family is not None,
        )
        return PixelRegressionRoutine(
            output_dim=output_dim,
            model=model,
            loss=loss,
            format_batch_fn=RepeatTarget(2),
            is_ensemble=True,
            optim_recipe=optim_recipe(model),
            dist_family=dist_family,
        )
