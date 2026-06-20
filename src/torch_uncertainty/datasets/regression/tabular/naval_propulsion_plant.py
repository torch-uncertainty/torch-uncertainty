import torch

from .base import TabularRegressionDataset, load_arff


class NavalPropulsionPlant(TabularRegressionDataset):
    """The UCI Condition Based Maintenance of Naval Propulsion Plants dataset.

    Predicts the gas turbine compressor decay state coefficient (second-to-last
    column). The turbine decay state coefficient (last column) is dropped.

    Note:
        The licenses of the datasets may differ from TorchUncertainty's
        license. Check before use.
    """

    url = "https://api.openml.org/data/v1/download/22111833"
    filename = "naval_propulsion_plant.arff"
    dataset_name = "naval-propulsion-plant"
    is_archive = False

    def _make_dataset(self) -> None:
        array = load_arff(self._data_path / self.filename).to_numpy()
        self.data = torch.tensor(array[:, :-2], dtype=torch.float32)
        self.targets = torch.tensor(array[:, -2], dtype=torch.float32)
