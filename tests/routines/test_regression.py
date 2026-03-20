from pathlib import Path

import pytest
import torch
from torch import nn

from tests._dummies import DummyRegressionBaseline, DummyRegressionDataModule
from torch_uncertainty import TUTrainer
from torch_uncertainty.losses import DistributionNLLLoss
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
