import pandas as pd
import torch

from .base import TabularRegressionDataset


class Yacht(TabularRegressionDataset):
    """The UCI Yacht Hydrodynamics dataset.

    Note:
        The licenses of the datasets may differ from TorchUncertainty's
        license. Check before use.
    """

    url = "https://archive.ics.uci.edu/static/public/243/yacht+hydrodynamics.zip"
    filename = "yacht_hydrodynamics.data"
    dataset_name = "yacht"
    md5 = "4e6727f462779e2d396e8f7d2ddb79a3"

    def _make_dataset(self) -> None:
        array = pd.read_csv(
            self._data_path / self.filename,
            sep=r"\s+",
            header=None,
        ).to_numpy()
        self.data = torch.tensor(array[:, :-1], dtype=torch.float32)
        self.targets = torch.tensor(array[:, -1], dtype=torch.float32)
