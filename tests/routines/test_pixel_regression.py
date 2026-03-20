from pathlib import Path

import pytest
import torch
from torch import nn

from tests._dummies import (
    DummyPixelRegressionBaseline,
    DummyPixelRegressionDataModule,
)
from torch_uncertainty import TUTrainer
from torch_uncertainty.losses import DistributionNLLLoss
from torch_uncertainty.optim_recipes import optim_cifar10_resnet18
from torch_uncertainty.routines.pixel_regression import (
    PixelRegressionRoutine,
    colorize,
)


class TestPixelRegression:
    def test_one_estimator_two_classes(self) -> None:
        trainer = TUTrainer(
            accelerator="cpu",
            max_epochs=1,
            logger=None,
            enable_checkpointing=False,
        )

        root = Path(__file__).parent.absolute().parents[0] / "data"
        dm = DummyPixelRegressionDataModule(root=root, batch_size=5, output_dim=3)

        model = DummyPixelRegressionBaseline(
            dist_family=None,
            in_channels=dm.num_channels,
            output_dim=dm.output_dim,
            image_size=dm.image_size,
            loss=nn.MSELoss(),
            baseline_type="single",
            optim_recipe=optim_cifar10_resnet18,
            ema=True,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

        trainer = TUTrainer(
            accelerator="cpu",
            max_epochs=1,
            logger=None,
            enable_checkpointing=False,
        )
        model = DummyPixelRegressionBaseline(
            dist_family="normal",
            in_channels=dm.num_channels,
            output_dim=dm.output_dim,
            image_size=dm.image_size,
            loss=DistributionNLLLoss(),
            baseline_type="single",
            optim_recipe=optim_cifar10_resnet18,
            swa=True,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_two_estimators_one_class(self) -> None:
        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)

        root = Path(__file__).parent.absolute().parents[0] / "data"
        dm = DummyPixelRegressionDataModule(root=root, batch_size=4, output_dim=1)

        model = DummyPixelRegressionBaseline(
            dist_family=None,
            in_channels=dm.num_channels,
            output_dim=dm.output_dim,
            image_size=dm.image_size,
            loss=nn.MSELoss(),
            baseline_type="ensemble",
            optim_recipe=optim_cifar10_resnet18,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True, logger=None)
        model = DummyPixelRegressionBaseline(
            dist_family="gamma",
            in_channels=dm.num_channels,
            output_dim=dm.output_dim,
            image_size=dm.image_size,
            loss=DistributionNLLLoss(),
            baseline_type="ensemble",
            optim_recipe=optim_cifar10_resnet18,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

        colorize(torch.ones((10, 10)), 0, 1)
        colorize(torch.ones((10, 10)), 0, 0)

    def test_depth_errors(self) -> None:
        with pytest.raises(ValueError, match=r"output_dim must be positive"):
            PixelRegressionRoutine(
                model=nn.Identity(),
                output_dim=0,
                loss=nn.MSELoss(),
                dist_family=None,
            )

        with pytest.raises(ValueError, match=r"num_image_plot must be positive"):
            PixelRegressionRoutine(
                model=nn.Identity(),
                output_dim=1,
                loss=nn.MSELoss(),
                dist_family=None,
                num_image_plot=0,
                log_plots=True,
            )

        with pytest.raises(
            ValueError,
            match=r"To train a model, you must specify the `loss` argument in the routine. Got None.",
        ):
            PixelRegressionRoutine(
                model=nn.Identity(),
                output_dim=1,
                loss=None,
            ).training_step((torch.tensor(float("nan")), torch.tensor(float("nan"))))
