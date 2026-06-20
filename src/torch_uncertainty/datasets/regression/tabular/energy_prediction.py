import pandas as pd
import torch

from .base import TabularRegressionDataset

_COLUMNS = [
    "Appliances",
    "lights",
    "T1",
    "RH_1",
    "T2",
    "RH_2",
    "T3",
    "RH_3",
    "T4",
    "RH_4",
    "T5",
    "RH_5",
    "T6",
    "RH_6",
    "T7",
    "RH_7",
    "T8",
    "RH_8",
    "T9",
    "RH_9",
]


class EnergyPrediction(TabularRegressionDataset):
    """The UCI Appliances Energy Prediction dataset.

    Predicts appliance energy consumption from indoor temperature/humidity
    sensors and lights energy. The outdoor temperature column (``T_out``) is
    excluded from the features.

    Note:
        The licenses of the datasets may differ from TorchUncertainty's
        license. Check before use.
    """

    url = "https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip"
    filename = "energydata_complete.csv"
    dataset_name = "energy-prediction"
    md5 = "d0f0f8ceaaf45df2233ce0600097bd84"

    def _make_dataset(self) -> None:
        array = pd.read_csv(self._data_path / self.filename)[_COLUMNS].to_numpy()
        self.targets = torch.tensor(array[:, 0], dtype=torch.float32)
        self.data = torch.tensor(array[:, 1:], dtype=torch.float32)
