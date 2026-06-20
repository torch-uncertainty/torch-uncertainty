import sys

from torch_uncertainty.datamodules import CIFAR10DataModule
from torch_uncertainty.routines import ClassificationRoutine
from torch_uncertainty.utils.cli import TULightningCLI, TUSaveConfigCallback


class TestCLI:
    """Testing torch-uncertainty CLI."""

    _BASE_ARGV = [
        "file.py",
        "--routine.model.class_path",
        "torch_uncertainty.models.resnet",
        "--routine.model.init_args.in_channels",
        "3",
        "--routine.model.init_args.num_classes",
        "10",
        "--routine.model.init_args.arch",
        "18",
        "--routine.num_classes",
        "10",
        "--routine.loss.class_path",
        "torch.nn.CrossEntropyLoss",
        "--data.root",
        "./data",
        "--data.batch_size",
        "4",
        "--trainer.callbacks+=ModelCheckpoint",
        "--trainer.callbacks.monitor=val/cls/Acc",
        "--trainer.callbacks.mode=max",
    ]

    def test_cli_init(self) -> None:
        """Test CLI initialization with default (no) save-config callback."""
        sys.argv = self._BASE_ARGV
        cli = TULightningCLI(ClassificationRoutine, CIFAR10DataModule, run=False)
        assert cli.eval_after_fit_default is False
        assert cli.save_config_callback is None

    def test_cli_init_save_config_callback(self) -> None:
        """Test CLI with TUSaveConfigCallback passed explicitly."""
        sys.argv = self._BASE_ARGV
        cli = TULightningCLI(
            ClassificationRoutine,
            CIFAR10DataModule,
            run=False,
            save_config_callback=TUSaveConfigCallback,
        )
        assert isinstance(cli.trainer.callbacks[0], TUSaveConfigCallback)
        cli.trainer.callbacks[0].setup(cli.trainer, cli.model, stage="fit")
        cli.trainer.callbacks[0].already_saved = True
        cli.trainer.callbacks[0].setup(cli.trainer, cli.model, stage="fit")
