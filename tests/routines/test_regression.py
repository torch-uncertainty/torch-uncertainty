from pathlib import Path

import pytest
import torch
from torch import nn

from tests._dummies import DummyRegressionBaseline, DummyRegressionDataModule
from torch_uncertainty import TUTrainer
from torch_uncertainty.losses import DistributionNLLLoss, ELBOLoss, PinballLoss
from torch_uncertainty.optim_recipes import optim_cifar10_resnet18
from torch_uncertainty.routines import RegressionRoutine


class TestRegression:
    """Testing the Regression routine."""

    def test_one_estimator_one_output(self) -> None:
        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)

        root = Path(__file__).parent.absolute().parents[0] / "data"
        dm = DummyRegressionDataModule(out_features=1, root=root, batch_size=4)

        model = DummyRegressionBaseline(
            in_features=dm.in_features,
            output_dim=1,
            loss=DistributionNLLLoss(),
            optim_recipe=optim_cifar10_resnet18,
            baseline_type="single",
            ema=True,
            dist_family="normal",
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)
        model = DummyRegressionBaseline(
            in_features=dm.in_features,
            output_dim=1,
            loss=nn.MSELoss(),
            optim_recipe=optim_cifar10_resnet18,
            baseline_type="single",
            swa=True,
            dist_family=None,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_one_estimator_two_outputs(self) -> None:
        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)

        root = Path(__file__).parent.absolute().parents[0] / "data"
        dm = DummyRegressionDataModule(out_features=2, root=root, batch_size=4)

        model = DummyRegressionBaseline(
            in_features=dm.in_features,
            output_dim=2,
            loss=DistributionNLLLoss(),
            optim_recipe=optim_cifar10_resnet18,
            baseline_type="single",
            dist_family="laplace",
        )
        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)
        model = DummyRegressionBaseline(
            in_features=dm.in_features,
            output_dim=2,
            loss=nn.MSELoss(),
            optim_recipe=optim_cifar10_resnet18,
            baseline_type="single",
            dist_family=None,
        )
        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_two_estimators_one_output(self) -> None:
        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)

        root = Path(__file__).parent.absolute().parents[0] / "data"
        dm = DummyRegressionDataModule(out_features=1, root=root, batch_size=4)

        model = DummyRegressionBaseline(
            in_features=dm.in_features,
            output_dim=1,
            loss=DistributionNLLLoss(),
            optim_recipe=optim_cifar10_resnet18,
            baseline_type="ensemble",
            dist_family="nig",
        )
        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)
        model = DummyRegressionBaseline(
            in_features=dm.in_features,
            output_dim=1,
            loss=nn.MSELoss(),
            optim_recipe=optim_cifar10_resnet18,
            baseline_type="ensemble",
            dist_family=None,
        )
        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_two_estimators_two_outputs(self) -> None:
        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)

        root = Path(__file__).parent.absolute().parents[0] / "data"
        dm = DummyRegressionDataModule(out_features=2, root=root, batch_size=4)

        model = DummyRegressionBaseline(
            in_features=dm.in_features,
            output_dim=2,
            loss=DistributionNLLLoss(),
            optim_recipe=optim_cifar10_resnet18,
            baseline_type="ensemble",
            dist_family="normal",
        )
        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)
        model = DummyRegressionBaseline(
            in_features=dm.in_features,
            output_dim=2,
            loss=nn.MSELoss(),
            optim_recipe=optim_cifar10_resnet18,
            baseline_type="ensemble",
            dist_family=None,
        )
        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_one_estimator_elbo_logs(self) -> None:
        """Cover the ELBO loss, calibration plotting and CSV-saving paths."""
        trainer = TUTrainer(
            accelerator="cpu",
            max_epochs=1,
            limit_train_batches=1,
            limit_val_batches=1,
            limit_test_batches=1,
            enable_checkpointing=False,
        )

        root = Path(__file__).parent.absolute().parents[0] / "data"
        dm = DummyRegressionDataModule(out_features=1, root=root, batch_size=4)

        model = DummyRegressionBaseline(
            in_features=dm.in_features,
            output_dim=1,
            loss=ELBOLoss(
                model=None,
                inner_loss=DistributionNLLLoss(),
                kl_weight=1.0,
                num_samples=2,
                dist_family="normal",
            ),
            optim_recipe=optim_cifar10_resnet18,
            baseline_type="single",
            dist_family="normal",
            save_to_csv=True,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)

    def test_quantile_mode(self) -> None:
        """The quantile output mode stores the levels and routes the median head."""
        routine = RegressionRoutine(
            output_dim=1,
            model=nn.Linear(4, 3),
            loss=PinballLoss(quantile=0.5),
            quantiles=[0.1, 0.5, 0.9],
        )
        assert routine.quantile_mode
        assert not routine.probabilistic
        assert routine.median_index == 1

        inputs = torch.randn(8, 4)
        # the quantile axis must be preserved by the forward pass
        assert routine(inputs).shape == (8, 3)

        preds, dist = routine.evaluation_forward(inputs)
        assert dist is None
        assert preds.shape == (8, 1)
        assert routine.last_quantile_preds.shape == (8, 3)
        # the point estimate is the median head
        assert torch.allclose(preds.squeeze(-1), routine.last_quantile_preds[:, 1])

    def test_quantile_mode_single_level(self) -> None:
        """A single quantile level must not collapse the quantile axis."""
        routine = RegressionRoutine(
            output_dim=1,
            model=nn.Linear(4, 1),
            loss=PinballLoss(quantile=0.5),
            quantiles=[0.5],
        )
        assert routine.median_index == 0
        assert routine(torch.randn(8, 4)).shape == (8, 1)

    def test_quantile_mode_failures(self) -> None:
        common = {"output_dim": 1, "model": nn.Identity(), "loss": nn.MSELoss()}
        with pytest.raises(ValueError, match=r"mutually exclusive"):
            RegressionRoutine(dist_family="normal", quantiles=[0.5], **common)
        with pytest.raises(ValueError, match=r"must not be empty"):
            RegressionRoutine(quantiles=[], **common)
        with pytest.raises(ValueError, match=r"must lie in \(0, 1\)"):
            RegressionRoutine(quantiles=[0.0, 0.5], **common)
        with pytest.raises(ValueError, match=r"must be sorted"):
            RegressionRoutine(quantiles=[0.9, 0.1], **common)
        with pytest.raises(ValueError, match=r"must not contain duplicates"):
            RegressionRoutine(quantiles=[0.5, 0.5], **common)

    def test_regression_failures(self) -> None:
        with pytest.raises(ValueError, match=r"output_dim must be positive"):
            RegressionRoutine(
                dist_family="normal",
                output_dim=0,
                model=nn.Identity(),
                loss=nn.MSELoss(),
            )

        routine = RegressionRoutine(
            dist_family="normal",
            output_dim=1,
            model=nn.Identity(),
            loss=nn.MSELoss(),
        )
        with pytest.raises(TypeError):
            routine(torch.randn(1, 1))

        with pytest.raises(
            ValueError,
            match=r"To train a model, you must specify the `loss` argument in the routine. Got None.",
        ):
            RegressionRoutine(
                dist_family="normal",
                output_dim=1,
                model=nn.Identity(),
                loss=None,
            ).training_step((torch.tensor(float("nan")), torch.tensor(float("nan"))))
