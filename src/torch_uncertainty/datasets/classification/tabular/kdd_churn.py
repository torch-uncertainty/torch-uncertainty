import pandas as pd
import torch

from .base import TabularClassificationDataset, load_arff


class KDDChurn(TabularClassificationDataset):
    """The KDD Cup 2009 Customer Churn dataset (OpenML 1112).

    Predicts telecommunications customer churn from 190 numeric and 40
    categorical features. Missing values are imputed with the column mean
    (numeric); categorical columns use label encoding because several have
    thousands of unique values, making one-hot encoding impractical.
    Downloaded from OpenML as an ARFF.

    Reference:
        G. Lemaitre et al., *Challenges in Representation Learning: A Report
        on Three Machine Learning Contests*, KDD Cup, 2009.

    Note:
        The licenses of the datasets may differ from TorchUncertainty's
        license. Check before use.
    """

    # OpenML dataset 1112, file_id 53995
    url = "https://api.openml.org/data/v1/download/53995"
    dataset_name = "kdd_churn"
    filename = "KDDCup09_churn.arff"
    is_archive = False

    def _make_dataset(self) -> None:
        df = load_arff(self.root / self.dataset_name / self.filename)
        target_col = "CHURN"
        self.targets = torch.as_tensor(
            (df[target_col].astype(float) > 0).astype(int).values.copy(), dtype=torch.long
        )
        df = df.drop(columns=[target_col])
        # Impute and label-encode: one-hot encoding is impractical here because
        # several categorical columns have thousands of unique values (>70k total).
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna("__missing__").astype("category").cat.codes.astype(float)
            else:
                df[col] = df[col].fillna(df[col].mean())
        self.data = torch.as_tensor(df.values.astype(float).copy(), dtype=torch.float32)
        self.num_features = self.data.shape[1]
