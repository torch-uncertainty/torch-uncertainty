import contextlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.cli import (
    ArgsType,
    LightningArgumentParser,
    LightningCLI,
    SaveConfigCallback,
)
from typing_extensions import override

from torch_uncertainty.utils.trainer import TUTrainer


class TUSaveConfigCallback(SaveConfigCallback):
    @override
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if self.already_saved:
            return

        if self.save_to_log_dir and stage == "fit":
            log_dir = trainer.log_dir  # this broadcasts the directory
            assert log_dir is not None
            config_path = Path(log_dir) / self.config_filename
            fs = get_filesystem(log_dir)

            if not self.overwrite:
                # check if the file exists on rank 0
                file_exists = fs.isfile(config_path) if trainer.is_global_zero else False
                # broadcast whether to fail to all ranks
                file_exists = trainer.strategy.broadcast(file_exists)
                if file_exists:  # coverage: ignore
                    raise RuntimeError(
                        f"{self.__class__.__name__} expected {config_path} to NOT exist. Aborting to avoid overwriting"
                        " results of a previous run. You can delete the previous config file,"
                        " set `LightningCLI(save_config_callback=None)` to disable config saving,"
                        ' or set `LightningCLI(save_config_kwargs={"overwrite": True})` to overwrite the config file.'
                    )

            if trainer.is_global_zero:
                fs.makedirs(log_dir, exist_ok=True)
                self.parser.save(
                    self.config,
                    config_path,
                    skip_none=False,
                    overwrite=self.overwrite,
                    multifile=self.multifile,
                )
        if trainer.is_global_zero:
            self.save_config(trainer, pl_module, stage)
            self.already_saved = True

        self.already_saved = trainer.strategy.broadcast(self.already_saved)


class TULightningCLI(LightningCLI):
    def __init__(
        self,
        model_class: (type[LightningModule] | Callable[..., LightningModule] | None) = None,
        datamodule_class: (
            type[LightningDataModule] | Callable[..., LightningDataModule] | None
        ) = None,
        save_config_callback: type[SaveConfigCallback] | None = TUSaveConfigCallback,
        save_config_kwargs: dict[str, Any] | None = None,
        trainer_class: type[Trainer] | Callable[..., Trainer] = TUTrainer,
        trainer_defaults: dict[str, Any] | None = None,
        seed_everything_default: bool | int = True,
        parser_kwargs: dict[str, Any] | dict[str, dict[str, Any]] | None = None,
        subclass_mode_model: bool = False,
        subclass_mode_data: bool = False,
        args: ArgsType = None,
        run: bool = True,
        auto_configure_optimizers: bool = True,
        eval_after_fit_default: bool = False,
        **kwargs: dict,
    ) -> None:
        """Custom LightningCLI for torch-uncertainty.

        Args:
            model_class (type[LightningModule] | Callable[..., LightningModule] | None, optional):
                An optional `LightningModule` class to train or a callable which returns a
                ``LightningModule`` instance when called. If ``None``, you can pass a registered model
                with ``--model=MyModel``. Defaults to ``None``.
            datamodule_class (type[LightningDataModule] | Callable[..., LightningDataModule] | None, optional):
                An optional ``LightningDataModule`` class or a callable which returns a ``LightningDataModule``
                instance when called. If ``None``, you can pass a registered datamodule with ``--data=MyDataModule``.
                Defaults to ``None``.
            save_config_callback (type[SaveConfigCallback] | None, optional): A callback class to
                save the config. Defaults to ``TUSaveConfigCallback``.
            save_config_kwargs (dict[str, Any] | None, optional): Parameters that will be used to
                instantiate the save_config_callback. Defaults to ``None``.
            trainer_class (type[Trainer] | Callable[..., Trainer], optional): An optional subclass
                of the Trainer class or a callable which returns a ``Trainer`` instance when called.
                Defaults to ``TUTrainer``.
            trainer_defaults (dict[str, Any] | None, optional): Set to override Trainer defaults
                or add persistent callbacks. The callbacks added through this argument will not
                be configurable from a configuration file and will always be present for this
                particular CLI. Alternatively, configurable callbacks can be added as explained
                in the CLI docs. Defaults to ``None``.
            seed_everything_default (bool | int, optional): Number for the ``seed_everything()``
                seed value. Set to ``True`` to automatically choose a seed value. Setting it to ``False``
                will avoid calling seed_everything. Defaults to ``True``.
            parser_kwargs (dict[str, Any] | dict[str, dict[str, Any]] | None, optional): Additional
                arguments to instantiate each ``LightningArgumentParser``. Defaults to
                ``LightningArgumentParser``. Defaults to ``None``.
            subclass_mode_model (bool, optional): Whether model can be any subclass of the given
                class. Defaults to ``False``.
            subclass_mode_data (bool, optional): Whether datamodule can be any subclass of the
                given class. Defaults to ``False``.
            args (ArgsType, optional): Arguments to parse. If `None` the arguments are taken from
                ``sys.argv``. Command line style arguments can be given in a ``list``. Alternatively,
                structured config options can be given in a ``dict`` or ``jsonargparse.Namespace``.
                Defaults to `None`.
            run (bool, optional): Whether subcommands should be added to run a ``Trainer`` method. If
                set to `False`, the trainer and model classes will be instantiated only. Defaults
                to ``True``.
            auto_configure_optimizers (bool, optional): Defaults to ``True``.
            eval_after_fit_default (bool, optional): Store whether an evaluation should be performed
                after the training. Defaults to ``False``.
            **kwargs: Additional keyword arguments to pass to the parent class added for
                ``lightning>2.5.0``:

                    - parser_class: The parser class to use. Defaults to `LightningArgumentParser`.
                      Available in ``lightning>=2.5.1``
        """
        self.eval_after_fit_default = eval_after_fit_default
        super().__init__(
            model_class=model_class,
            datamodule_class=datamodule_class,
            save_config_callback=save_config_callback,
            save_config_kwargs=save_config_kwargs,
            trainer_class=trainer_class,
            trainer_defaults=trainer_defaults,
            seed_everything_default=seed_everything_default,
            parser_kwargs=parser_kwargs,
            subclass_mode_model=subclass_mode_model,
            subclass_mode_data=subclass_mode_data,
            args=args,
            run=run,
            auto_configure_optimizers=auto_configure_optimizers,
            **kwargs,
        )

    def add_core_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Adds arguments from the core classes to the parser."""
        parser.add_lightning_class_args(self.trainer_class, "trainer")
        trainer_defaults = {
            "trainer." + k: v for k, v in self.trainer_defaults.items() if k != "callbacks"
        }
        parser.set_defaults(trainer_defaults)

        parser.add_lightning_class_args(
            self._model_class, "routine", subclass_mode=self.subclass_mode_model
        )

        if self.datamodule_class is not None:
            parser.add_lightning_class_args(
                self._datamodule_class, "data", subclass_mode=self.subclass_mode_data
            )
        else:
            # this should not be required because the user might want to use the `LightningModule` dataloaders
            parser.add_lightning_class_args(
                self._datamodule_class,
                "data",
                subclass_mode=self.subclass_mode_data,
                required=False,
            )

    def instantiate_classes(self) -> None:
        """Instantiates the classes and sets their attributes."""
        self.config_init = self.parser.instantiate_classes(self.config)
        self.datamodule = self._get(self.config_init, "data")
        self.routine = self._get(self.config_init, "routine")
        self._add_configure_optimizers_method_to_model(self.subcommand)
        self.trainer = self.instantiate_trainer()

    @property
    def model(self):
        return self.routine

    def add_default_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Adds default arguments to the parser."""
        parser.add_argument(
            "--eval_after_fit",
            action="store_true",
            default=self.eval_after_fit_default,
        )
        super().add_default_arguments_to_parser(parser)

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)
        with contextlib.suppress(ValueError):
            parser.link_arguments("data.eval_ood", "routine.eval_ood")
        with contextlib.suppress(ValueError):
            parser.link_arguments("data.eval_shift", "routine.eval_shift")
        with contextlib.suppress(ValueError):
            parser.link_arguments("data.num_tta", "routine.num_tta")
