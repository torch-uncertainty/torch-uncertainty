from pathlib import Path

from torch_uncertainty.datamodules.abstract import TUDataModule
from torch_uncertainty.datasets.classification.tabular import (
    HTRU2,
    AdultCensusIncome,
    AmazonAccess,
    APSFailure,
    BankMarketing,
    CreditApproval,
    DOTA2Games,
    GermanCredit,
    HiggsBoson,
    KDDChurn,
    OnlineShoppers,
    PimaDiabetes,
    SpamBase,
    TabularClassificationDataset,
    TelcoChurn,
    WineQuality,
)
from torch_uncertainty.datasets.utils import create_train_val_split


class TabularClassificationDataModule(TUDataModule):
    """Base datamodule for tabular binary classification datasets.

    Subclasses must set :attr:`dataset_class` to the corresponding
    :class:`TabularClassificationDataset` subclass. No ``__init__`` override
    is needed in the subclass.

    Example::

        class HTRU2DataModule(TabularClassificationDataModule):
            dataset_class = HTRU2
    """

    training_task = "classification"
    dataset_class: type[TabularClassificationDataset] | None = None

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
        binary: bool = True,
    ) -> None:
        """Tabular binary classification datamodule.

        Args:
            root: Root directory of the datasets.
            batch_size: Number of samples per batch during training.
            eval_batch_size: Number of samples per batch during evaluation. Defaults to
                :attr:`batch_size`.
            val_split: Share of the training samples to use as validation set. Defaults to ``0``.
            test_split: Share of the full dataset to hold out as test set (used when the dataset
                has no predefined split). Defaults to ``0.2``.
            num_workers: Number of data-loading subprocesses. Defaults to ``1``.
            pin_memory: Whether to pin memory. Defaults to ``True``.
            persistent_workers: Whether to keep workers alive between epochs. Defaults to ``True``.
            binary: If ``True``, returns scalar targets. Defaults to ``True``.
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
        self.binary = binary

    def prepare_data(self) -> None:
        """Download the dataset if not already present."""
        self.dataset_class(root=self.root, download=True, download_only=True)

    def setup(self, stage: str | None = None) -> None:
        """Create train, val, and test splits.

        Args:
            stage: ``"fit"``, ``"test"``, or ``None``. Defaults to ``None``.
        """
        if stage == "fit" or stage is None:
            full = self.dataset_class(
                self.root,
                train=True,
                download=False,
                binary=self.binary,
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
                    binary=self.binary,
                    test_split=self.test_split,
                )
        if stage == "test" or stage is None:
            self.test = self.dataset_class(
                self.root,
                train=False,
                download=False,
                binary=self.binary,
                test_split=self.test_split,
            )
        if stage not in ("fit", "test", None):
            raise ValueError(f"Stage {stage!r} is not supported.")


class AdultCensusIncomeDataModule(TabularClassificationDataModule):
    """Datamodule for the UCI Adult Census Income dataset."""

    dataset_class = AdultCensusIncome


class AmazonAccessDataModule(TabularClassificationDataModule):
    """Datamodule for the Amazon Employee Access dataset."""

    dataset_class = AmazonAccess


class APSFailureDataModule(TabularClassificationDataModule):
    """Datamodule for the UCI APS Failure at Scania Trucks dataset."""

    dataset_class = APSFailure


class BankMarketingDataModule(TabularClassificationDataModule):
    """Datamodule for the UCI Bank Marketing dataset."""

    dataset_class = BankMarketing


class CreditApprovalDataModule(TabularClassificationDataModule):
    """Datamodule for the UCI Credit Approval dataset."""

    dataset_class = CreditApproval


class DOTA2GamesDataModule(TabularClassificationDataModule):
    """Datamodule for the UCI DOTA 2 Games Results dataset."""

    dataset_class = DOTA2Games


class GermanCreditDataModule(TabularClassificationDataModule):
    """Datamodule for the UCI Statlog German Credit dataset."""

    dataset_class = GermanCredit


class HiggsBosonDataModule(TabularClassificationDataModule):
    """Datamodule for the Higgs Boson dataset (OpenML 23512)."""

    dataset_class = HiggsBoson


class HTRU2DataModule(TabularClassificationDataModule):
    """Datamodule for the UCI HTRU2 pulsar dataset."""

    dataset_class = HTRU2


class KDDChurnDataModule(TabularClassificationDataModule):
    """Datamodule for the KDD Cup 2009 Customer Churn dataset (OpenML 1112)."""

    dataset_class = KDDChurn


class OnlineShoppersDataModule(TabularClassificationDataModule):
    """Datamodule for the UCI Online Shoppers Purchasing Intention dataset."""

    dataset_class = OnlineShoppers


class PimaDiabetesDataModule(TabularClassificationDataModule):
    """Datamodule for the UCI Pima Indians Diabetes dataset."""

    dataset_class = PimaDiabetes


class SpamBaseDataModule(TabularClassificationDataModule):
    """Datamodule for the UCI SpamBase e-mail spam dataset."""

    dataset_class = SpamBase


class TelcoChurnDataModule(TabularClassificationDataModule):
    """Datamodule for the Telecom Customer Churn dataset (OpenML 40701)."""

    dataset_class = TelcoChurn


class WineQualityDataModule(TabularClassificationDataModule):
    """Datamodule for the UCI Wine Quality dataset."""

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
        binary: bool = True,
        variant: str = "red",
        threshold: int = 6,
    ) -> None:
        """Wine Quality datamodule.

        Args:
            root: Root directory of the datasets.
            batch_size: Number of samples per training batch.
            eval_batch_size: Samples per evaluation batch. Defaults to :attr:`batch_size`.
            val_split: Share of training samples used for validation. Defaults to ``0``.
            test_split: Share of the full dataset held out as test set. Defaults to ``0.2``.
            num_workers: Data-loading subprocesses. Defaults to ``1``.
            pin_memory: Whether to pin memory. Defaults to ``True``.
            persistent_workers: Whether to keep workers alive between epochs. Defaults to ``True``.
            binary: If ``True``, binarises quality scores. Defaults to ``True``.
            variant: ``"red"`` or ``"white"``. Defaults to ``"red"``.
            threshold: Quality threshold for binary mode. Defaults to ``6``.
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
            binary=binary,
        )
        self.variant = variant
        self.threshold = threshold

    def prepare_data(self) -> None:
        self.dataset_class(root=self.root, download=True, download_only=True, variant=self.variant)

    def setup(self, stage: str | None = None) -> None:
        kwargs = {
            "download": False,
            "binary": self.binary,
            "test_split": self.test_split,
            "variant": self.variant,
            "threshold": self.threshold,
        }
        if stage == "fit" or stage is None:
            full = self.dataset_class(self.root, train=True, **kwargs)
            if self.val_split:
                self.train, self.val = create_train_val_split(full, self.val_split)
            else:
                self.train = full
                self.val = self.dataset_class(self.root, train=False, **kwargs)
        if stage == "test" or stage is None:
            self.test = self.dataset_class(self.root, train=False, **kwargs)
        if stage not in ("fit", "test", None):
            raise ValueError(f"Stage {stage!r} is not supported.")
