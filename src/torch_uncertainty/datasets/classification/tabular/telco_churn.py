import pandas as pd
import torch

from .base import TabularClassificationDataset, load_arff


class TelcoChurn(TabularClassificationDataset):
    """The Telecom Customer Churn dataset (OpenML 40701).

    Predicts whether a customer churns. The ``phone_number`` column is dropped
    as it is a non-predictive identifier. Downloaded from the OpenML repository
    as an ARFF file.

    Note:
        The licenses of the datasets may differ from TorchUncertainty's
        license. Check before use.
    """

    # OpenML dataset 40701, file_id 4965302
    url = "https://api.openml.org/data/v1/download/4965302"
    dataset_name = "telco_churn"
    filename = "churn.arff"
    is_archive = False

    def _make_dataset(self) -> None:
        df = load_arff(self.root / self.dataset_name / self.filename)
        # Drop non-predictive identifier
        df = df.drop(columns=["phone_number"], errors="ignore")
        target_col = "class"
        normalised = df[target_col].astype(str).str.strip().str.rstrip(".").str.lower()
        unique = set(normalised.unique())
        positive_aliases = {"true", "1", "yes"}
        negative_aliases = {"false", "0", "no"}
        if not unique <= positive_aliases | negative_aliases:
            raise ValueError(
                f"TelcoChurn: unexpected values in '{target_col}': {sorted(unique)}. "
                "Expected a binary nominal attribute (True/False, 1/0, or yes/no)."
            )
        self.targets = torch.as_tensor(
            normalised.isin(positive_aliases).astype(int).values.copy(), dtype=torch.long
        )
        df = df.drop(columns=[target_col])
        cat_cols = df.select_dtypes(include="object").columns
        df = pd.get_dummies(df, columns=cat_cols).astype(float)
        self.data = torch.as_tensor(df.values.copy(), dtype=torch.float32)
        self.num_features = self.data.shape[1]
