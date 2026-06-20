import warnings
from urllib.error import URLError

import pytest

from tests._dummies.dataset import DummyRegressionDataset
from torch_uncertainty.datamodules.classification import (
    AdultCensusIncomeDataModule,
    AmazonAccessDataModule,
    APSFailureDataModule,
    BankMarketingDataModule,
    CreditApprovalDataModule,
    DOTA2GamesDataModule,
    GermanCreditDataModule,
    HiggsBosonDataModule,
    HTRU2DataModule,
    KDDChurnDataModule,
    OnlineShoppersDataModule,
    PimaDiabetesDataModule,
    SpamBaseDataModule,
    TabularClassificationDataModule,
    TelcoChurnDataModule,
    WineQualityDataModule,
)


def _exercise_lifecycle(dm) -> None:
    dm.prepare_data()
    dm.setup()
    dm.train_dataloader()
    dm.val_dataloader()
    dm.test_dataloader()
    dm.setup("fit")
    dm.setup("test")
    with pytest.raises(ValueError):
        dm.setup("other")


class TestTabularClassificationDataModule:
    """TabularClassificationDataModule base-class and HTRU2 representative behavior."""

    def test_missing_dataset_class_raises(self) -> None:
        with pytest.raises(TypeError):
            TabularClassificationDataModule(root="./data/", batch_size=128)

    @pytest.mark.parametrize("val_split", [0.0, 0.5])
    def test_htru2_lifecycle(self, val_split: float) -> None:
        dm = HTRU2DataModule(root="./data/", batch_size=128, val_split=val_split)
        dm.dataset_class = DummyRegressionDataset
        _exercise_lifecycle(dm)


class TestWineQualityDataModule:
    """WineQualityDataModule has its own setup/prepare_data override."""

    @pytest.mark.parametrize("val_split", [0.0, 0.5])
    def test_wine_quality_lifecycle_with_dummy(self, val_split: float) -> None:
        dm = WineQualityDataModule(
            root="./data/", batch_size=128, variant="red", val_split=val_split
        )
        dm.dataset_class = DummyRegressionDataset
        _exercise_lifecycle(dm)

    def test_wine_quality_real_download(self) -> None:
        try:
            dm = WineQualityDataModule(
                root="./data/", batch_size=128, variant="white", val_split=0.1
            )
            dm.prepare_data()
            dm.setup()
        except URLError as e:
            warnings.warn(f"Data download failed due to network error: {e}", stacklevel=2)


class TestOtherClassificationDataModules:
    """Smoke-test instantiation of all remaining classification datamodules."""

    @pytest.mark.parametrize(
        "cls",
        [
            AdultCensusIncomeDataModule,
            AmazonAccessDataModule,
            APSFailureDataModule,
            BankMarketingDataModule,
            CreditApprovalDataModule,
            DOTA2GamesDataModule,
            GermanCreditDataModule,
            HiggsBosonDataModule,
            KDDChurnDataModule,
            OnlineShoppersDataModule,
            PimaDiabetesDataModule,
            SpamBaseDataModule,
            TelcoChurnDataModule,
        ],
    )
    def test_module_instantiates(self, cls) -> None:
        cls(root="./data/", batch_size=128)
