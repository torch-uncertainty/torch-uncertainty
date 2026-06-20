import torch

from .base import TabularRegressionDataset, load_arff


class Protein(TabularRegressionDataset):
    """The UCI Physicochemical Properties of Protein Tertiary Structure dataset.

    Predicts the RMSD (Root Mean Square Deviation) from nine structural
    features (F1-F9). RMSD is the first column in the source file.

    Note:
        The licenses of the datasets may differ from TorchUncertainty's
        license. Check before use.
    """

    url = "https://api.openml.org/data/v1/download/22111827"
    filename = "protein.arff"
    dataset_name = "protein"
    is_archive = False

    def _make_dataset(self) -> None:
        array = load_arff(self._data_path / self.filename).to_numpy()
        self.data = torch.tensor(array[:, 1:], dtype=torch.float32)
        self.targets = torch.tensor(array[:, 0], dtype=torch.float32)
