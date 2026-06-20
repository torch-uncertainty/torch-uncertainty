import torch

from .base import TabularRegressionDataset, load_arff


class Concrete(TabularRegressionDataset):
    """The UCI Concrete Compressive Strength dataset.

    Note:
        The licenses of the datasets may differ from TorchUncertainty's
        license. Check before use.
    """

    url = "https://api.openml.org/data/v1/download/22111823"
    filename = "concrete.arff"
    dataset_name = "concrete"
    is_archive = False

    def _make_dataset(self) -> None:
        array = load_arff(self._data_path / self.filename).to_numpy()
        self.data = torch.tensor(array[:, :-1], dtype=torch.float32)
        self.targets = torch.tensor(array[:, -1], dtype=torch.float32)
