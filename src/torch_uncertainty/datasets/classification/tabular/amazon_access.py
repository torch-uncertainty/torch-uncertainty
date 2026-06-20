import torch

from .base import TabularClassificationDataset, load_arff


class AmazonAccess(TabularClassificationDataset):
    """The Amazon Employee Access dataset (OpenML 41135).

    Predicts whether an employee's request for access to a resource should be
    granted, based on role and resource identifiers. All nine predictive features
    are categorical (encoded as integers). Downloaded from OpenML as an ARFF.

    Note:
        The exact OpenML dataset identifier for this dataset should be verified
        before use. The URL below targets OpenML dataset 41135; if the download
        fails, check ``https://www.openml.org`` for the correct file identifier.

        The licenses of the datasets may differ from TorchUncertainty's
        license. Check before use.
    """

    # OpenML dataset 4135 — file_id from https://www.openml.org/api/v1/json/data/4135
    url = "https://api.openml.org/data/v1/download/1681098"
    dataset_name = "amazon_access"
    filename = "amazon_employee_access.arff"
    is_archive = False

    def _make_dataset(self) -> None:
        df = load_arff(self.root / self.dataset_name / self.filename)
        target_col = "target" if "target" in df.columns else df.columns[-1]
        self.targets = torch.tensor(df[target_col].astype(int).values.copy(), dtype=torch.long)
        df = df.drop(columns=[target_col])
        # All 9 features are high-cardinality nominal integer IDs; label-encode
        # each column to avoid the OOM that one-hot encoding would cause.
        for col in df.columns:
            df[col] = df[col].astype("category").cat.codes.astype(float)
        self.data = torch.as_tensor(df.values.copy(), dtype=torch.float32)
        self.num_features = self.data.shape[1]
