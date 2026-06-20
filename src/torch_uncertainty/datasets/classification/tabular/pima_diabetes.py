import torch

from .base import TabularClassificationDataset, load_arff


class PimaDiabetes(TabularClassificationDataset):
    """The UCI Pima Indians Diabetes dataset.

    Predicts diabetes onset from clinical measurements. All features are
    numeric. The dataset is downloaded from the OpenML static data server
    (dataset 37).

    Reference:
        J.W. Smith et al., *Using the ADAP Learning Algorithm to Forecast the
        Onset of Diabetes Mellitus*, Proc. SCAMC, 1988.

    Note:
        The licenses of the datasets may differ from TorchUncertainty's
        license. Check before use.
    """

    # OpenML dataset 37 — static CDN, no database dependency
    url = "https://data.openml.org/datasets/0000/0037/dataset_37.arff"
    dataset_name = "pima_diabetes"
    filename = "pima_diabetes.arff"
    is_archive = False

    def _make_dataset(self) -> None:
        df = load_arff(self.root / self.dataset_name / self.filename)
        target_col = "class"
        target_vals = df[target_col].str.strip()
        # OpenML encodes: tested_negative → 0, tested_positive → 1
        self.targets = torch.as_tensor(
            (target_vals == "tested_positive").astype(int).values.copy(), dtype=torch.long
        )
        df = df.drop(columns=[target_col])
        self.data = torch.as_tensor(df.values.astype(float).copy(), dtype=torch.float32)
        self.num_features = self.data.shape[1]
