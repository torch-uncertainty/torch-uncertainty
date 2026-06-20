import pandas as pd
import torch

from .base import TabularRegressionDataset


class WineQuality(TabularRegressionDataset):
    """The UCI Wine Quality dataset (regression).

    Predicts the wine quality score as a continuous value.

    Args:
        variant: ``"red"`` or ``"white"``. Defaults to ``"red"``.

    Note:
        The licenses of the datasets may differ from TorchUncertainty's
        license. Check before use.
    """

    url = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"
    dataset_name = "wine-quality"
    md5 = "0ddfa7a9379510fe7ff88b9930e3c332"

    def __init__(self, *args, variant: str = "red", **kwargs) -> None:
        if variant not in ("red", "white"):
            raise ValueError(f"variant must be 'red' or 'white', got {variant!r}.")
        self.variant = variant
        self.filename = f"winequality-{variant}.csv"
        super().__init__(*args, **kwargs)

    def _make_dataset(self) -> None:
        array = pd.read_csv(self._data_path / self.filename, sep=";").to_numpy()
        self.data = torch.tensor(array[:, :-1], dtype=torch.float32)
        self.targets = torch.tensor(array[:, -1], dtype=torch.float32)
