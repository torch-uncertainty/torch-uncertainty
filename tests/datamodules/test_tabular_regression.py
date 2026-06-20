import warnings
from urllib.error import URLError

import pytest

from tests._dummies.dataset import DummyRegressionDataset
from torch_uncertainty.datamodules import (
    BostonHousingDataModule,
    ConcreteDataModule,
    EnergyEfficiencyDataModule,
    EnergyPredictionDataModule,
    Kin8NMDataModule,
    NavalPropulsionPlantDataModule,
    PowerPlantDataModule,
    ProteinDataModule,
    TabularRegressionDataModule,
    WineQualityRegressionDataModule,
    YachtDataModule,
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


class TestTabularRegressionDataModule:
    """TabularRegressionDataModule base-class and Kin8NM representative behavior."""

    def test_missing_dataset_class_raises(self) -> None:
        with pytest.raises(TypeError):
            TabularRegressionDataModule(root="./data/", batch_size=128)

    @pytest.mark.parametrize("val_split", [0.0, 0.5])
    def test_kin8nm_lifecycle(self, val_split: float) -> None:
        dm = Kin8NMDataModule(root="./data/", batch_size=128, val_split=val_split)
        dm.dataset_class = DummyRegressionDataset
        _exercise_lifecycle(dm)
        assert dm._extra_repr() == ""


class TestWineQualityRegressionDataModule:
    """WineQualityRegressionDataModule has its own setup/prepare_data override."""

    @pytest.mark.parametrize("val_split", [0.0, 0.5])
    def test_wine_quality_lifecycle_with_dummy(
        self, val_split: float, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The override calls ``WineQuality(...)`` directly, so patch the import site.
        monkeypatch.setattr(
            "torch_uncertainty.datamodules.tabular_regression.WineQuality",
            DummyRegressionDataset,
        )
        dm = WineQualityRegressionDataModule(
            root="./data/", batch_size=128, variant="red", val_split=val_split
        )
        _exercise_lifecycle(dm)
        assert dm._extra_repr() == "variant='red'"

    def test_wine_quality_real_download(self) -> None:
        try:
            dm = WineQualityRegressionDataModule(
                root="./data/", batch_size=128, variant="white", val_split=0.1
            )
            dm.prepare_data()
            dm.setup()
        except URLError as e:
            warnings.warn(f"Data download failed due to network error: {e}", stacklevel=2)


class TestOtherRegressionDataModules:
    """Smoke-test instantiation of all remaining regression datamodules."""

    @pytest.mark.parametrize(
        "cls",
        [
            BostonHousingDataModule,
            ConcreteDataModule,
            EnergyEfficiencyDataModule,
            EnergyPredictionDataModule,
            NavalPropulsionPlantDataModule,
            PowerPlantDataModule,
            ProteinDataModule,
            YachtDataModule,
        ],
    )
    def test_module_instantiates(self, cls) -> None:
        cls(root="./data/", batch_size=128)
