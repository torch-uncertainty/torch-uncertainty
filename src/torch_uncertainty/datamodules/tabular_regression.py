from pathlib import Path

from torch_uncertainty.datasets.regression import (
    BostonHousing,
    Concrete,
    EnergyEfficiency,
    EnergyPrediction,
    Kin8NM,
    NavalPropulsionPlant,
    PowerPlant,
    Protein,
    TabularRegressionDataset,
    WineQuality,
    Yacht,
)
from torch_uncertainty.datasets.utils import create_train_val_split

from .abstract import TUDataModule


class TabularRegressionDataModule(TUDataModule):
    """Base datamodule for UCI regression datasets.

    Subclasses must set :attr:`dataset_class` to the corresponding
    :class:`TabularRegressionDataset` subclass.
    """

    training_task = "regression"
    dataset_class: type[TabularRegressionDataset] | None = None

    def __init__(
        self,
        root: str | Path,
        batch_size: int,
        eval_batch_size: int | None = None,
        val_split: float = 0.0,
        test_split: float = 0.2,
        num_workers: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ) -> None:
        """UCI regression datamodule.

        Args:
            root: Root directory of the datasets.
            batch_size: Number of samples per batch during training.
            eval_batch_size: Number of samples per batch during evaluation.
                Defaults to :attr:`batch_size`.
            val_split: Share of training samples used for validation.
                Defaults to ``0``.
            test_split: Share of the full dataset held out as test set.
                Defaults to ``0.2``.
            num_workers: Number of data-loading subprocesses. Defaults to ``1``.
            pin_memory: Whether to pin memory. Defaults to ``True``.
            persistent_workers: Whether to keep workers alive between epochs.
                Defaults to ``True``.
        """
        super().__init__(
            root=root,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            val_split=val_split,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        if self.__class__.dataset_class is None:
            raise TypeError(
                f"{self.__class__.__name__} must set `dataset_class` as a class attribute."
            )
        self.test_split = test_split

    def prepare_data(self) -> None:
        """Download the dataset if not already present."""
        self.dataset_class(root=self.root, download=True)

    def setup(self, stage: str | None = None) -> None:
        """Create train, val, and test splits.

        Args:
            stage: ``"fit"``, ``"test"``, or ``None`` (both). Defaults to ``None``.
        """
        if stage == "fit" or stage is None:
            full = self.dataset_class(
                self.root,
                train=True,
                download=False,
                test_split=self.test_split,
            )
            if self.val_split:
                self.train, self.val = create_train_val_split(full, self.val_split)
            else:
                self.train = full
                self.val = self.dataset_class(
                    self.root,
                    train=False,
                    download=False,
                    test_split=self.test_split,
                )
        if stage == "test" or stage is None:
            self.test = self.dataset_class(
                self.root,
                train=False,
                download=False,
                test_split=self.test_split,
            )
        if stage not in ("fit", "test", None):
            raise ValueError(f"Stage {stage!r} is not supported.")

    def _extra_repr(self) -> str:
        return ""


class BostonHousingDataModule(TabularRegressionDataModule):
    """Datamodule for the UCI Boston Housing dataset."""

    dataset_class = BostonHousing


class ConcreteDataModule(TabularRegressionDataModule):
    """Datamodule for the UCI Concrete Compressive Strength dataset."""

    dataset_class = Concrete


class EnergyEfficiencyDataModule(TabularRegressionDataModule):
    """Datamodule for the UCI Energy Efficiency dataset."""

    dataset_class = EnergyEfficiency


class EnergyPredictionDataModule(TabularRegressionDataModule):
    """Datamodule for the UCI Appliances Energy Prediction dataset."""

    dataset_class = EnergyPrediction


class Kin8NMDataModule(TabularRegressionDataModule):
    """Datamodule for the Kin8NM robot arm kinematics dataset."""

    dataset_class = Kin8NM


class NavalPropulsionPlantDataModule(TabularRegressionDataModule):
    """Datamodule for the UCI Naval Propulsion Plants dataset."""

    dataset_class = NavalPropulsionPlant


class PowerPlantDataModule(TabularRegressionDataModule):
    """Datamodule for the UCI Combined Cycle Power Plant dataset."""

    dataset_class = PowerPlant


class ProteinDataModule(TabularRegressionDataModule):
    """Datamodule for the UCI Protein Tertiary Structure dataset."""

    dataset_class = Protein


class WineQualityRegressionDataModule(TabularRegressionDataModule):
    """Datamodule for the UCI Wine Quality dataset (regression)."""

    dataset_class = WineQuality

    def __init__(
        self,
        root: str | Path,
        batch_size: int,
        eval_batch_size: int | None = None,
        val_split: float = 0.0,
        test_split: float = 0.2,
        num_workers: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        variant: str = "red",
    ) -> None:
        """Wine Quality regression datamodule.

        Args:
            root: Root directory of the datasets.
            batch_size: Number of samples per training batch.
            eval_batch_size: Samples per evaluation batch. Defaults to
                :attr:`batch_size`.
            val_split: Share of training samples used for validation.
                Defaults to ``0``.
            test_split: Share of the full dataset held out as test set.
                Defaults to ``0.2``.
            num_workers: Data-loading subprocesses. Defaults to ``1``.
            pin_memory: Whether to pin memory. Defaults to ``True``.
            persistent_workers: Whether to keep workers alive between epochs.
                Defaults to ``True``.
            variant: ``"red"`` or ``"white"``. Defaults to ``"red"``.
        """
        super().__init__(
            root=root,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            val_split=val_split,
            test_split=test_split,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        self.variant = variant

    def _extra_repr(self) -> str:
        return f"variant={self.variant!r}"

    def prepare_data(self) -> None:
        WineQuality(root=self.root, download=True, variant=self.variant)

    def setup(self, stage: str | None = None) -> None:
        kwargs = {
            "download": False,
            "test_split": self.test_split,
            "variant": self.variant,
        }
        if stage == "fit" or stage is None:
            full = WineQuality(self.root, train=True, **kwargs)
            if self.val_split:
                self.train, self.val = create_train_val_split(full, self.val_split)
            else:
                self.train = full
                self.val = WineQuality(self.root, train=False, **kwargs)
        if stage == "test" or stage is None:
            self.test = WineQuality(self.root, train=False, **kwargs)
        if stage not in ("fit", "test", None):
            raise ValueError(f"Stage {stage!r} is not supported.")


class YachtDataModule(TabularRegressionDataModule):
    """Datamodule for the UCI Yacht Hydrodynamics dataset."""

    dataset_class = Yacht
