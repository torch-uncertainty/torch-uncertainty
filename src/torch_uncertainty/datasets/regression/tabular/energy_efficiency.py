import torch

from .base import TabularRegressionDataset, load_arff


class EnergyEfficiency(TabularRegressionDataset):
    """The UCI Energy Efficiency dataset.

    Predicts the heating load of buildings from eight building features.
    The cooling load (second target) is dropped.

    Note:
        The licenses of the datasets may differ from TorchUncertainty's
        license. Check before use.
    """

    url = "https://api.openml.org/data/v1/download/22111824"
    filename = "energy_efficiency.arff"
    dataset_name = "energy-efficiency"
    is_archive = False

    def _make_dataset(self) -> None:
        array = load_arff(self._data_path / self.filename).to_numpy()
        self.data = torch.tensor(array[:, :-2], dtype=torch.float32)
        self.targets = torch.tensor(array[:, -2], dtype=torch.float32)
