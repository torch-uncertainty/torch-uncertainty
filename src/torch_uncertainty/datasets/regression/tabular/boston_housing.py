import pandas as pd
import torch

from .base import TabularRegressionDataset

_COLUMNS = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT",
    "MEDV",
]


class BostonHousing(TabularRegressionDataset):
    """The Boston Housing dataset.

    Note:
        You may want to avoid using this dataset because of ethical concerns.
        The licenses of the datasets may differ from TorchUncertainty's
        license. Check before use.
    """

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
    filename = "housing.data"
    dataset_name = "boston"
    is_archive = False
    md5 = "d4accdce7a25600298819f8e28e8d593"

    def _make_dataset(self) -> None:
        array = pd.read_table(
            self._data_path / self.filename,
            names=_COLUMNS,
            header=None,
            sep=r"\s+",
        ).to_numpy()
        self.data = torch.tensor(array[:, :-1], dtype=torch.float32)
        self.targets = torch.tensor(array[:, -1], dtype=torch.float32)
