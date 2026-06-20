import pandas as pd
import torch

from .base import TabularClassificationDataset


class CreditApproval(TabularClassificationDataset):
    """The UCI Credit Approval dataset.

    Predicts credit card application approval. Features are anonymised;
    missing values (``?``) are imputed with the column mean for numeric
    attributes and the mode for categorical ones.

    Reference:
        J.R. Quinlan, *Simplifying Decision Trees*, IJMMS, 1987.

    Note:
        The licenses of the datasets may differ from TorchUncertainty's
        license. Check before use.
    """

    url = "https://archive.ics.uci.edu/static/public/27/credit+approval.zip"
    dataset_name = "credit_approval"
    filename = "crx.data"

    def _make_dataset(self) -> None:
        data = pd.read_csv(
            self.root / self.dataset_name / self.filename,
            header=None,
            na_values=["?"],
        )
        # Target is the last column: '+' → 1, '-' → 0
        self.targets = torch.as_tensor(
            (data.iloc[:, -1] == "+").astype(int).values.copy(), dtype=torch.long
        )
        data = data.iloc[:, :-1]
        # Impute missing values
        for col in data.columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                data[col] = data[col].fillna(data[col].mode()[0])
            else:
                data[col] = data[col].fillna(data[col].mean())
        self.data = torch.as_tensor(
            pd.get_dummies(data).astype(float).values.copy(), dtype=torch.float32
        )
        self.num_features = self.data.shape[1]
