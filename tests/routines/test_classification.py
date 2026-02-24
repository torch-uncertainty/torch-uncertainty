from pathlib import Path

import pytest
from torch import nn

from tests._dummies import (
    DummyClassificationBaseline,
    DummyClassificationDataModule,
    dummy_model,
)
from torch_uncertainty import TUTrainer
from torch_uncertainty.losses import DECLoss, ELBOLoss
from torch_uncertainty.ood_criteria import (
    EntropyCriterion,
    PostProcessingCriterion,
)
from torch_uncertainty.post_processing import ConformalClsTHR
from torch_uncertainty.routines import ClassificationRoutine
from torch_uncertainty.transforms import RepeatTarget


class TestClassification:
    """Testing the classification routine."""

    def test_one_estimator_binary(self) -> None:
        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)

        dm = DummyClassificationDataModule(
            root=Path(),
            batch_size=16,
            num_classes=1,
            num_images=100,
        )
        model = DummyClassificationBaseline(
            in_channels=dm.num_channels,
            num_classes=dm.num_classes,
            loss=nn.BCEWithLogitsLoss(),
            baseline_type="single",
            ema=True,
            mixtype=None,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_two_estimators_binary(self) -> None:
        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)

        dm = DummyClassificationDataModule(
            root=Path(),
            batch_size=16,
            num_classes=1,
            num_images=100,
        )
        model = DummyClassificationBaseline(
            in_channels=dm.num_channels,
            num_classes=dm.num_classes,
            loss=nn.BCEWithLogitsLoss(),
            baseline_type="single",
            ood_criterion="logit",
            swa=True,
            mixtype=None,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_one_estimator_two_classes(self) -> None:
        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)

        dm = DummyClassificationDataModule(
            root=Path(),
            batch_size=16,
            num_classes=2,
            num_images=100,
            eval_ood=True,
            eval_shift=True,
        )
        model = DummyClassificationBaseline(
            num_classes=dm.num_classes,
            in_channels=dm.num_channels,
            loss=nn.CrossEntropyLoss(),
            baseline_type="single",
            ood_criterion=EntropyCriterion,
            eval_ood=True,
            eval_shift=True,
            no_mixup_params=True,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_one_estimator_two_classes_timm(self) -> None:
        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)

        dm = DummyClassificationDataModule(
            root=Path(),
            batch_size=16,
            num_classes=2,
            num_images=100,
            eval_ood=True,
        )
        model = DummyClassificationBaseline(
            num_classes=dm.num_classes,
            in_channels=dm.num_channels,
            loss=nn.CrossEntropyLoss(),
            baseline_type="single",
            ood_criterion="entropy",
            eval_ood=True,
            mixtype="timm",
            mixup_alpha=1.0,
            cutmix_alpha=0.5,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_one_estimator_two_classes_mixup(self) -> None:
        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)

        dm = DummyClassificationDataModule(
            root=Path(),
            batch_size=16,
            num_classes=2,
            num_images=100,
            eval_ood=True,
        )
        model = DummyClassificationBaseline(
            num_classes=dm.num_classes,
            in_channels=dm.num_channels,
            loss=nn.CrossEntropyLoss(),
            baseline_type="single",
            ood_criterion="entropy",
            eval_ood=True,
            mixtype="mixup",
            mixup_alpha=1.0,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_one_estimator_two_classes_mixupmp_ratio_lower_than_1(self) -> None:
        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)

        dm = DummyClassificationDataModule(
            root=Path(),
            batch_size=16,
            num_classes=2,
            num_images=100,
            eval_ood=True,
        )
        model = DummyClassificationBaseline(
            num_classes=dm.num_classes,
            in_channels=dm.num_channels,
            loss=nn.CrossEntropyLoss(),
            baseline_type="single",
            ood_criterion="entropy",
            eval_ood=True,
            mixtype="mixupmp",
            mixup_ratio=0.5,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_one_estimator_two_classes_mixupmp_ratio_greater_than_1(self) -> None:
        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)

        dm = DummyClassificationDataModule(
            root=Path(),
            batch_size=16,
            num_classes=2,
            num_images=100,
            eval_ood=True,
        )
        model = DummyClassificationBaseline(
            num_classes=dm.num_classes,
            in_channels=dm.num_channels,
            loss=nn.CrossEntropyLoss(),
            baseline_type="single",
            ood_criterion="entropy",
            eval_ood=True,
            mixtype="mixupmp",
            mixup_ratio=1.5,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_one_estimator_two_classes_mixup_io(self) -> None:
        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)

        dm = DummyClassificationDataModule(
            root=Path(),
            batch_size=16,
            num_classes=2,
            num_images=100,
            eval_ood=True,
        )
        model = DummyClassificationBaseline(
            num_classes=dm.num_classes,
            in_channels=dm.num_channels,
            loss=nn.CrossEntropyLoss(),
            baseline_type="single",
            ood_criterion="entropy",
            eval_ood=True,
            mixtype="mixupio",
            mixup_alpha=1.0,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_one_estimator_two_classes_regmixup(self) -> None:
        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)

        dm = DummyClassificationDataModule(
            root=Path(),
            batch_size=16,
            num_classes=2,
            num_images=100,
            eval_ood=True,
        )
        model = DummyClassificationBaseline(
            num_classes=dm.num_classes,
            in_channels=dm.num_channels,
            loss=nn.CrossEntropyLoss(),
            baseline_type="single",
            ood_criterion="entropy",
            eval_ood=True,
            mixtype="regmixup",
            mixup_alpha=1.0,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_one_estimator_two_classes_kernel_warping_emb(self) -> None:
        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)

        dm = DummyClassificationDataModule(
            root=Path(),
            batch_size=16,
            num_classes=2,
            num_images=100,
            eval_ood=True,
        )
        model = DummyClassificationBaseline(
            num_classes=dm.num_classes,
            in_channels=dm.num_channels,
            loss=nn.CrossEntropyLoss(),
            baseline_type="single",
            ood_criterion="entropy",
            eval_ood=True,
            mixtype="kernel_warping",
            mixup_alpha=0.5,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_one_estimator_two_classes_kernel_warping_inp(self) -> None:
        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)

        dm = DummyClassificationDataModule(
            root=Path(),
            batch_size=16,
            num_classes=2,
            num_images=100,
            eval_ood=True,
        )
        model = DummyClassificationBaseline(
            num_classes=dm.num_classes,
            in_channels=dm.num_channels,
            loss=nn.CrossEntropyLoss(),
            baseline_type="single",
            ood_criterion="entropy",
            eval_ood=True,
            mixtype="kernel_warping",
            kw_on_embeddings=False,
            mixup_alpha=0.5,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_one_estimator_two_classes_calibrated_with_ood(self) -> None:
        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True, logger=True)

        dm = DummyClassificationDataModule(
            root=Path(),
            batch_size=19,  # lower than 19 it doesn't work :'(
            num_classes=2,
            num_images=100,
            eval_ood=True,
        )
        model = DummyClassificationBaseline(
            num_classes=dm.num_classes,
            in_channels=dm.num_channels,
            loss=nn.CrossEntropyLoss(),
            baseline_type="single",
            ood_criterion="energy",
            eval_ood=True,
            eval_grouping_loss=True,
            calibrate=True,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_two_estimators_two_classes_mi(self) -> None:
        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)

        dm = DummyClassificationDataModule(
            root=Path(),
            batch_size=16,
            num_classes=2,
            num_images=100,
            eval_ood=True,
            eval_shift=True,
        )
        model = DummyClassificationBaseline(
            num_classes=dm.num_classes,
            in_channels=dm.num_channels,
            loss=DECLoss(1, 1e-2),
            baseline_type="ensemble",
            ood_criterion="mutual_information",
            eval_ood=True,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_two_estimator_two_classes_elbo_vr_logs(self) -> None:
        trainer = TUTrainer(
            accelerator="cpu",
            max_epochs=1,
            limit_train_batches=1,
            limit_val_batches=1,
            limit_test_batches=1,
            enable_checkpointing=False,
        )

        dm = DummyClassificationDataModule(
            root=Path(),
            batch_size=16,
            num_classes=2,
            num_images=100,
            eval_ood=True,
        )
        model = DummyClassificationBaseline(
            num_classes=dm.num_classes,
            in_channels=dm.num_channels,
            loss=ELBOLoss(None, nn.CrossEntropyLoss(), kl_weight=1.0, num_samples=4),
            baseline_type="ensemble",
            ood_criterion="variation_ratio",
            eval_ood=True,
            save_in_csv=True,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_one_estimator_conformal(self) -> None:
        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)

        dm = DummyClassificationDataModule(
            root=Path(),
            batch_size=16,
            num_classes=3,
            num_images=100,
            eval_ood=True,
        )

        model = dummy_model(
            in_channels=dm.num_channels,
            num_classes=dm.num_classes,
        )
        routine = ClassificationRoutine(
            model=model,
            loss=None,
            num_classes=3,
            post_processing=ConformalClsTHR(alpha=0.1),
            ood_criterion=PostProcessingCriterion(),
            eval_ood=True,
        )
        trainer.test(routine, dm)

        model = ConformalClsTHR(
            alpha=0.1,
            model=dummy_model(
                in_channels=dm.num_channels,
                num_classes=dm.num_classes,
            ),
        )
        model.fit(dm.postprocess_dataloader())

        routine = ClassificationRoutine(
            model=model,
            loss=None,
            num_classes=3,
            post_processing=None,
        )
        trainer.test(routine, dm)

    def test_classification_failures(self) -> None:
        # num_classes
        with pytest.raises(ValueError):
            ClassificationRoutine(num_classes=0, model=nn.Module(), loss=None)
        # single & MI
        with pytest.raises(ValueError):
            ClassificationRoutine(
                num_classes=10,
                model=nn.Module(),
                loss=None,
                is_ensemble=False,
                ood_criterion="mutual_information",
            )

        with pytest.raises(ValueError):
            ClassificationRoutine(
                num_classes=10,
                model=nn.Module(),
                loss=None,
                is_ensemble=False,
                ood_criterion=32,
            )

        with pytest.raises(ValueError):
            ClassificationRoutine(
                num_classes=10,
                model=nn.Module(),
                loss=None,
                is_ensemble=False,
                ood_criterion="other",
            )

        mixup_params = {"mixtype": "mixup", "cutmix_alpha": -1}
        with pytest.raises(ValueError):
            ClassificationRoutine(
                num_classes=10,
                model=nn.Module(),
                loss=None,
                mixup_params=mixup_params,
            )

        with pytest.raises(ValueError, match=r"num_bins_calibration_error must be at least 2, got"):
            ClassificationRoutine(
                model=nn.Identity(),
                num_classes=2,
                loss=nn.CrossEntropyLoss(),
                num_bins_calibration_error=0,
            )

        with pytest.raises(ValueError):
            ClassificationRoutine(
                num_classes=10,
                model=nn.Module(),
                loss=None,
                eval_grouping_loss=True,
            )

        with pytest.raises(NotImplementedError):
            ClassificationRoutine(
                num_classes=10,
                model=nn.Module(),
                loss=None,
                is_ensemble=True,
                eval_grouping_loss=True,
            )

        model = dummy_model(1, 1, 0, with_feats=False)
        with pytest.raises(ValueError):
            ClassificationRoutine(num_classes=10, model=model, loss=None, eval_grouping_loss=True)

        with pytest.raises(
            ValueError,
            match=r"Mixup is not supported for ensembles at training time",
        ):
            ClassificationRoutine(
                num_classes=10,
                model=nn.Module(),
                loss=None,
                mixup_params={"mixtype": "mixup"},
                format_batch_fn=RepeatTarget(2),
            )

        with pytest.raises(
            ValueError,
            match=r"Ensembles and post-processing methods cannot be used together. Raise an issue if needed.",
        ):
            ClassificationRoutine(
                num_classes=10,
                model=nn.Module(),
                loss=None,
                is_ensemble=True,
                post_processing=nn.Module(),
            )

        with pytest.raises(
            ValueError,
            match=r"You cannot set ood_criterion=PostProcessingCriterion when post_processing is None.",
        ):
            ClassificationRoutine(
                num_classes=10,
                model=nn.Module(),
                loss=None,
                post_processing=None,
                ood_criterion=PostProcessingCriterion(),
            )
