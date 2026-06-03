import logging
import types
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from tests._dummies import (
    DummyClassificationBaseline,
    DummyClassificationDataModule,
    dummy_model,
    dummy_ood_model,
)
from torch_uncertainty import TUTrainer
from torch_uncertainty.losses import DECLoss, ELBOLoss
from torch_uncertainty.ood.ood_criteria import (
    EntropyCriterion,
    MaxSoftmaxCriterion,
)
from torch_uncertainty.post_processing import ConformalClsTHR
from torch_uncertainty.routines import ClassificationRoutine
from torch_uncertainty.transforms import RepeatTarget
from torch_uncertainty.utils.evaluation_loop import TUEvaluationLoop


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
            mixtype="mixup_io",
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
            dist_sim="inp",
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

    OOD_CRITS = [
        "scale",
        "ash",
        "react",
        "adascale_a",
        "vim",
        "odin",
        "knn",
        "neco",
        "gen",
        "nnguide",
    ]

    @pytest.mark.parametrize("crit", OOD_CRITS)
    def test_all_other_ood_criteria_with_dummy_ood_model(self, crit, monkeypatch):
        device = torch.device("cpu")

        monkeypatch.setattr(
            torch.Tensor,
            "cuda",
            lambda self, *a, **k: self.to(device),  # noqa: ARG005
            raising=False,
        )
        monkeypatch.setattr(nn.Module, "cuda", lambda self, *a, **k: self.to(device), raising=False)  # noqa: ARG005

        trainer = TUTrainer(
            accelerator="cpu",
            devices=1,
            inference_mode=False,
            num_sanity_val_steps=0,
            max_epochs=1,
            limit_train_batches=1,
            limit_val_batches=1,
            limit_test_batches=1,
            enable_checkpointing=False,
            logger=False,
        )

        class _TensorDS(Dataset):
            def __init__(self, x, y, name="tensor_ds"):
                self.x, self.y, self.dataset_name = x, y, name

            def __len__(self):
                return self.x.shape[0]

            def __getitem__(self, i):
                return self.x[i], self.y[i]

        def _mk_split(n, c, h, w, num_classes, shift=0.0, seed=0):
            g = torch.Generator().manual_seed(seed)
            x = (torch.rand((n, c, h, w), generator=g) + shift).clamp(0, 1)
            y = (torch.arange(n) % num_classes).long()
            return x.float(), y

        dm = DummyClassificationDataModule(
            root=Path(),
            batch_size=8,
            num_classes=3,
            num_images=64,
            num_workers=0,
            eval_ood=True,
            persistent_workers=False,
        )

        def patched_setup(self, stage=None):
            self.num_channels = 3
            h = w = self.image_size
            n = self.num_images

            if stage in (None, "fit"):
                x_tr, y_tr = _mk_split(n, 3, h, w, self.num_classes, 0.0, 123)
                x_va, y_va = _mk_split(n, 3, h, w, self.num_classes, 0.0, 456)
                self.train = _TensorDS(x_tr, y_tr, "train")
                self.val = _TensorDS(x_va, y_va, "val")

            if stage in (None, "test"):
                x_te, y_te = _mk_split(n, 3, h, w, self.num_classes, 0.0, 789)
                self.test = _TensorDS(x_te, y_te, "test")
                if self.eval_ood:
                    x_vo, y_vo = _mk_split(n, 3, h, w, self.num_classes, 0.10, 321)
                    self.val_ood = _TensorDS(x_vo, y_vo, "val_ood")
                    x_near, y_near = _mk_split(n, 3, h, w, self.num_classes, 0.15, 654)
                    self.near_oods = [_TensorDS(x_near, y_near, "near")]
                    x_far, y_far = _mk_split(n, 3, h, w, self.num_classes, 0.35, 987)
                    self.far_oods = [_TensorDS(x_far, y_far, "far")]

            if self.eval_shift:
                x_sh, y_sh = _mk_split(n, 3, h, w, self.num_classes, 0.20, 111)
                self.shift = _TensorDS(x_sh, y_sh, "shift")
                self.shift_severity = 1

        monkeypatch.setattr(dm, "setup", patched_setup.__get__(dm, type(dm)), raising=True)

        model = dummy_ood_model(in_channels=3, feat_dim=4096, num_classes=dm.num_classes).to(device)

        routine = ClassificationRoutine(
            model=model,
            loss=None,
            num_classes=dm.num_classes,
            eval_ood=True,
            ood_criterion=crit,
            log_plots=False,
        )

        c = routine.ood_criterion
        if hasattr(c, "args_dict"):
            if crit in ("scale", "ash", "react"):
                c.args_dict = {"percentile": [70]}
            elif crit == "adascale_a":
                c.args_dict = {
                    "percentile": [(40, 60)],
                    "k1": [1],
                    "k2": [1],
                    "lmbda": [0.1],
                    "o": [0.05],
                }
            elif crit == "vim":
                safe_dim = min(64, getattr(model, "feature_size", 256) - 1)
                c.args_dict = {"dim": [safe_dim]}
                c.dim = safe_dim
            elif crit == "odin":
                c.args_dict = {"temperature": [1.0], "noise": [0.0014]}
            elif crit == "knn":
                c.args_dict = {"K": [5]}
            elif crit == "neco":
                # train split has dm.num_images (64) samples; PCA rank is capped by n_samples
                safe_dim = min(32, dm.num_images - 1, getattr(model, "feature_size", 256) - 1)
                c.args_dict = {"neco_dim": [safe_dim]}
                c.neco_dim = safe_dim
            elif crit == "gen":
                # ensure m ≤ num_classes to avoid degenerate slices
                c.gamma = getattr(c, "gamma", 0.1)
                c.m = min(getattr(c, "m", 10), dm.num_classes)
                c.args_dict = {"gamma": [c.gamma], "m": [c.m]}
            elif crit == "nnguide":
                c.args_dict = {"K": [5], "alpha": [0.5]}
            c.hyperparam_search_done = False

        trainer.test(routine, dm)

        if hasattr(c, "args_dict"):
            assert getattr(c, "hyperparam_search_done", False), (
                f"Hyperparam search did not complete for '{crit}'."
            )
        for needs_setup in {"react", "adascale_a", "vim", "knn", "neco", "nnguide"}:
            if crit == needs_setup:
                assert getattr(c, "setup_flag", False), f"Setup not executed for '{crit}'."

    def test_setup_logs_when_no_train_loader(self, caplog, monkeypatch):
        dm = DummyClassificationDataModule(
            root=Path(),
            batch_size=4,
            num_classes=3,
            num_images=16,
            eval_ood=True,
        )

        def _raise_train_loader(*_a, **_k):
            raise RuntimeError("no train loader")

        monkeypatch.setattr(
            ClassificationRoutine, "_hyperparam_search_ood", lambda _self: None, raising=True
        )
        monkeypatch.setattr(dm, "train_dataloader", _raise_train_loader, raising=True)

        model = dummy_ood_model(in_channels=3, feat_dim=64, num_classes=3)
        routine = ClassificationRoutine(
            model=model,
            loss=None,
            num_classes=3,
            eval_ood=True,
        )
        routine.ood_criterion = MaxSoftmaxCriterion()  # no setup() side-effects

        routine.trainer = types.SimpleNamespace(datamodule=dm)

        with caplog.at_level(logging.INFO):
            routine.setup("test")
        assert any("No train loader detected" in r.message for r in caplog.records)

    def test_ood_val_loader_for_hyperparam_search_prefers_val_ood(self):
        """HP search must use val_ood (e.g. Tiny ImageNet), not test_ood (same-domain CIFAR-100)."""

        class _TaggedLoader:
            def __init__(self, tag: str):
                self.tag = tag

            def __iter__(self):
                return iter([])

        loaders = [
            _TaggedLoader("id_test"),
            _TaggedLoader("test_ood"),
            _TaggedLoader("val_ood"),
            _TaggedLoader("near"),
        ]
        dm = types.SimpleNamespace(
            test_dataloader=lambda: loaders,
            get_indices=lambda: {
                "test": [0],
                "test_ood": [1],
                "val_ood": [2],
                "near_oods": [3],
                "far_oods": [],
                "shift": [],
            },
        )
        model = dummy_ood_model(in_channels=3, feat_dim=64, num_classes=3)
        routine = ClassificationRoutine(model=model, loss=None, num_classes=3, eval_ood=True)
        routine.trainer = types.SimpleNamespace(datamodule=dm)

        chosen = routine._ood_val_loader_for_hyperparam_search()
        assert chosen.tag == "val_ood"

    def test_ood_val_loader_for_hyperparam_search_falls_back_to_test_ood(self):
        class _TaggedLoader:
            def __init__(self, tag: str):
                self.tag = tag

            def __iter__(self):
                return iter([])

        loaders = [_TaggedLoader("id_test"), _TaggedLoader("test_ood")]
        dm = types.SimpleNamespace(
            test_dataloader=lambda: loaders,
            get_indices=lambda: {
                "test": [0],
                "test_ood": [1],
                "val_ood": [],
                "near_oods": [],
                "far_oods": [],
                "shift": [],
            },
        )
        model = dummy_ood_model(in_channels=3, feat_dim=64, num_classes=3)
        routine = ClassificationRoutine(model=model, loss=None, num_classes=3, eval_ood=True)
        routine.trainer = types.SimpleNamespace(datamodule=dm)

        chosen = routine._ood_val_loader_for_hyperparam_search()
        assert chosen.tag == "test_ood"

    def test_create_near_far_metric_dicts_non_ensemble(self, capsys):
        model = dummy_ood_model(in_channels=3, feat_dim=64, num_classes=3)
        routine = ClassificationRoutine(
            model=model, loss=None, num_classes=3, eval_ood=True, is_ensemble=False
        )
        routine.ood_criterion = MaxSoftmaxCriterion()

        x = torch.rand(4, 3, 8, 8)
        y = torch.tensor([0, 1, 2, 0])

        class _DS:
            def __init__(self, name):
                self.dataset_name = name

        routine.trainer = types.SimpleNamespace(
            datamodule=types.SimpleNamespace(
                get_indices=lambda: {"val_ood": 9, "near_oods": [2], "far_oods": [3], "shift": []},
                near_oods=[_DS("nearX")],
                far_oods=[_DS("farY")],
            )
        )
        routine.on_test_start()

        routine.test_step((x, y), batch_idx=0, dataloader_idx=2)  # near
        assert "nearX" in routine.test_ood_metrics_near

        routine.test_step((x, y), batch_idx=0, dataloader_idx=3)  # far
        assert "farY" in routine.test_ood_metrics_far

        routine.test_ood_metrics_near["stale_near"] = routine.ood_metrics_template.clone(
            prefix="ood_near_stale_near_"
        )
        routine.test_ood_metrics_far["stale_far"] = routine.ood_metrics_template.clone(
            prefix="ood_far_stale_far_"
        )

        routine.on_test_start()
        assert "stale_near" not in routine.test_ood_metrics_near
        assert "stale_far" not in routine.test_ood_metrics_far

        routine.test_step((x, y), batch_idx=0, dataloader_idx=2)
        routine.test_step((x, y), batch_idx=0, dataloader_idx=3)
        for near_metrics in routine.test_ood_metrics_near.values():
            near_metrics.compute()
        for far_metrics in routine.test_ood_metrics_far.values():
            far_metrics.compute()

        fake_results = [
            {
                "ood_near_nearX_auroc": torch.tensor(0.91),
                "ood_near_nearX_fpr95": torch.tensor(0.09),
                "ood_near_nearX_aupr": torch.tensor(0.88),
                "ood_near_dsB_auroc": torch.tensor(0.92),
                "ood_near_dsB_fpr95": torch.tensor(0.08),
                "ood_near_dsB_aupr": torch.tensor(0.86),
                "ood_far_farY_auroc": torch.tensor(0.81),
                "ood_far_farY_fpr95": torch.tensor(0.19),
                "ood_far_farY_aupr": torch.tensor(0.72),
                "ood_far_dsD_auroc": torch.tensor(0.79),
                "ood_far_dsD_fpr95": torch.tensor(0.21),
                "ood_far_dsD_aupr": torch.tensor(0.70),
            }
        ]
        TUEvaluationLoop._print_results(fake_results, stage="test")
        out = capsys.readouterr().out
        assert "OOD Results" in out
        assert "NearOOD Average" in out
        assert "FarOOD Average" in out

    def test_create_near_far_metric_dicts_ensemble_and_aggregator(self):
        model = dummy_ood_model(in_channels=3, feat_dim=64, num_classes=3)
        routine = ClassificationRoutine(
            model=model, loss=None, num_classes=3, eval_ood=True, is_ensemble=True
        )
        routine.ood_criterion = MaxSoftmaxCriterion()

        x = torch.rand(4, 3, 8, 8)
        y = torch.tensor([0, 1, 2, 0])

        class _DS:
            def __init__(self, name):
                self.dataset_name = name

        routine.trainer = types.SimpleNamespace(
            datamodule=types.SimpleNamespace(
                get_indices=lambda: {
                    "val_ood": 9,
                    "near_oods": [5],
                    "far_oods": [6],
                    "shift": [7],
                },
                near_oods=[_DS("n1")],
                far_oods=[_DS("f1")],
            )
        )

        routine.test_step((x, y), batch_idx=0, dataloader_idx=1)  # aggregator
        assert "n1" in routine.test_ood_ens_metrics_near
        assert "f1" in routine.test_ood_ens_metrics_far

        routine.test_step((x, y), batch_idx=0, dataloader_idx=5)  # near
        routine.test_step((x, y), batch_idx=0, dataloader_idx=6)  # far
        assert "n1" in routine.test_ood_ens_metrics_near
        assert "f1" in routine.test_ood_ens_metrics_far

    def test_skip_when_val_ood_loader(self):
        model = dummy_ood_model(in_channels=3, feat_dim=64, num_classes=3)
        routine = ClassificationRoutine(model=model, loss=None, num_classes=3, eval_ood=True)
        routine.ood_criterion = MaxSoftmaxCriterion()

        routine.trainer = types.SimpleNamespace(
            datamodule=types.SimpleNamespace(
                get_indices=lambda: {"val_ood": 4, "near_oods": [], "far_oods": [], "shift": []}
            )
        )
        x = torch.rand(2, 3, 8, 8)
        y = torch.tensor([0, 1])
        routine.test_step((x, y), batch_idx=0, dataloader_idx=4)

    def test_init_metrics_creates_shift_ens_metrics_when_ensemble_and_eval_shift(self):
        model = dummy_ood_model(in_channels=3, feat_dim=64, num_classes=3)
        routine = ClassificationRoutine(
            model=model, loss=None, num_classes=3, eval_shift=True, is_ensemble=True
        )
        assert hasattr(routine, "test_shift_ens_metrics")

    def test_shift_ens_update_path(self):
        model = dummy_ood_model(in_channels=3, feat_dim=64, num_classes=3)
        routine = ClassificationRoutine(
            model=model, loss=None, num_classes=3, eval_shift=True, is_ensemble=True
        )
        routine.ood_criterion = MaxSoftmaxCriterion()

        x = torch.rand(4, 3, 8, 8)
        y = torch.tensor([0, 1, 2, 0])

        routine.trainer = types.SimpleNamespace(
            datamodule=types.SimpleNamespace(
                get_indices=lambda: {"val_ood": 99, "near_oods": [], "far_oods": [], "shift": [7]}
            )
        )
        routine.test_step((x, y), batch_idx=0, dataloader_idx=7)

    def test_logs_when_eval_flags_mismatch_datamodule(self, caplog):
        model = dummy_ood_model(in_channels=3, feat_dim=64, num_classes=3)
        routine = ClassificationRoutine(
            model=model, loss=None, num_classes=3, eval_ood=False, eval_shift=False
        )
        routine.ood_criterion = MaxSoftmaxCriterion()

        class _DM:
            def get_indices(self):
                return {"val_ood": 9, "near_oods": [2], "far_oods": [3], "shift": [4]}

        routine._trainer = types.SimpleNamespace(barebones=True, datamodule=_DM())

        x = torch.rand(2, 3, 8, 8)
        y = torch.tensor([0, 1])

        with caplog.at_level(logging.INFO):
            routine.test_step((x, y), batch_idx=0, dataloader_idx=0)

        assert any(
            "`eval_ood` to `True` in the datamodule and not in the routine" in r.message
            for r in caplog.records
        )
        assert any(
            "`eval_shift` to `True` in the datamodule and not in the routine" in r.message
            for r in caplog.records
        )

    def test_guardrails__classification_routine_checks(self):
        # ---- Local dummy models (minimal & fast) ----

        class BareModel(nn.Module):
            """No feats_forward, no classifier attrs."""

            def __init__(self, num_classes=3):
                super().__init__()
                self.fc = nn.Linear(4, num_classes)

            def forward(self, x):
                b = x.size(0) if hasattr(x, "size") else 2
                return self.fc(torch.zeros((b, 4)))

        class FeatsWithHead(nn.Module):
            """Has feats_forward + classification_head."""

            def __init__(self, num_classes=3):
                super().__init__()
                self.backbone = nn.Linear(4, 8)
                self.classification_head = nn.Linear(8, num_classes)

            def feats_forward(self, x):
                b = x.size(0) if hasattr(x, "size") else 2
                return self.backbone(torch.zeros((b, 4)))

            def forward(self, x):
                return self.classification_head(self.feats_forward(x))

        class FeatsWithLinear(nn.Module):
            """Has feats_forward + linear."""

            def __init__(self, num_classes=3):
                super().__init__()
                self.backbone = nn.Linear(4, 8)
                self.linear = nn.Linear(8, num_classes)

            def feats_forward(self, x):
                b = x.size(0) if hasattr(x, "size") else 2
                return self.backbone(torch.zeros((b, 4)))

            def forward(self, x):
                return self.linear(self.feats_forward(x))

        class FeatsNoHeadNoLinear(nn.Module):
            """Has feats_forward but neither classification_head nor linear."""

            def feats_forward(self, x):
                b = x.size(0) if hasattr(x, "size") else 2
                return torch.zeros((b, 8))

            def forward(self, x):
                b = x.size(0) if hasattr(x, "size") else 2
                return torch.zeros((b, 3))

        for crit in ("mutual_information", "variation_ratio"):
            with pytest.raises(ValueError, match="mutual information|variation ratio"):
                ClassificationRoutine(
                    model=BareModel(),
                    num_classes=3,
                    is_ensemble=False,
                    ood_criterion=crit,
                )

        with pytest.raises(NotImplementedError, match="Logit-based criteria"):
            ClassificationRoutine(
                model=BareModel(),
                num_classes=3,
                is_ensemble=True,
                ood_criterion="logit",
            )

        with pytest.raises(NotImplementedError, match="Grouping loss for ensembles"):
            ClassificationRoutine(
                model=FeatsWithHead(),
                num_classes=3,
                is_ensemble=True,
                eval_grouping_loss=True,
            )

        with pytest.raises(ValueError, match="positive integer"):
            ClassificationRoutine(
                model=BareModel(),
                num_classes=0,
            )

        with pytest.raises(ValueError, match="feats_forward"):
            ClassificationRoutine(
                model=BareModel(),
                num_classes=3,
                eval_grouping_loss=True,
            )

        with pytest.raises(ValueError, match="classification_head|linear"):
            ClassificationRoutine(
                model=FeatsNoHeadNoLinear(),
                num_classes=3,
                eval_grouping_loss=True,
            )

        with pytest.raises(ValueError, match="at least 2"):
            ClassificationRoutine(
                model=BareModel(),
                num_classes=3,
                num_bins_cal_err=1,
            )

        with pytest.raises(ValueError, match="Mixup is not supported for ensembles"):
            ClassificationRoutine(
                model=BareModel(),
                num_classes=3,
                is_ensemble=True,
                mixup_params={"mixup_alpha": 1.0},
                format_batch_fn=RepeatTarget(num_repeats=2),
            )

        with pytest.raises(ValueError, match="Ensembles and post-processing"):
            ClassificationRoutine(
                model=BareModel(),
                num_classes=3,
                is_ensemble=True,
                post_processing=ConformalClsTHR(alpha=0.1),
            )

        ClassificationRoutine(
            model=BareModel(),
            num_classes=3,
            is_ensemble=False,
            ood_criterion="msp",
            eval_grouping_loss=False,
            num_bins_cal_err=15,
            mixup_params=None,
            post_processing=None,
            format_batch_fn=None,
        )

        ClassificationRoutine(
            model=FeatsWithHead(),
            num_classes=3,
            eval_grouping_loss=True,
        )

        ClassificationRoutine(
            model=FeatsWithLinear(),
            num_classes=3,
            eval_grouping_loss=True,
        )
