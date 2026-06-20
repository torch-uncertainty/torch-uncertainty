import pandas as pd
import torch

from .base import TabularRegressionDataset


class PowerPlant(TabularRegressionDataset):
    """The UCI Combined Cycle Power Plant dataset.

    Note:
        The licenses of the datasets may differ from TorchUncertainty's
        license. Check before use.
    """

    url = "https://archive.ics.uci.edu/static/public/294/combined+cycle+power+plant.zip"
    filename = "CCPP/Folds5x2_pp.xlsx"
    dataset_name = "power-plant"
    md5 = "f5065a616eae05eb4ecae445ecf6e720"

    def _make_dataset(self) -> None:
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            raise ImportError(
                "openpyxl is required to read the Power Plant dataset (.xlsx format). "
                "Install it with: pip install openpyxl"
            ) from None
        array = pd.read_excel(self._data_path / self.filename, engine="openpyxl").to_numpy()
        self.data = torch.tensor(array[:, :-1], dtype=torch.float32)
        self.targets = torch.tensor(array[:, -1], dtype=torch.float32)
