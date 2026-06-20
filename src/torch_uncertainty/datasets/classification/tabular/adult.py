import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .base import TabularClassificationDataset

_ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]


class AdultCensusIncome(TabularClassificationDataset):
    """The UCI Adult Census Income dataset.

    Predicts whether income exceeds $50K/year. The ``fnlwgt`` sampling-weight
    column is dropped as it is not a predictive feature. Train and test files
    are pre-split by the data provider; one-hot columns and imputation
    statistics are derived from the training file and applied to the test file
    to avoid distribution drift.

    Reference:
        R. Kohavi, *Scaling Up the Accuracy of Naive-Bayes Classifiers: a
        Decision-Tree Hybrid*, KDD, 1996.

    Note:
        The licenses of the datasets may differ from TorchUncertainty's
        license. Check before use.
    """

    url = "https://archive.ics.uci.edu/static/public/2/adult.zip"
    dataset_name = "adult"
    filename = "adult.data"
    need_split = False
    pre_split = True

    def _read(self, fname: str, skiprows: int) -> pd.DataFrame:
        return pd.read_csv(
            self.root / self.dataset_name / fname,
            names=_ADULT_COLUMNS,
            skipinitialspace=True,
            na_values=["?"],
            skiprows=skiprows,
        )

    def _make_pre_split_dataset(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        train_df = self._read("adult.data", skiprows=0)
        # Test file begins with a header comment line.
        test_df = self._read("adult.test", skiprows=1)

        # Strip trailing period from the test file target column
        train_df["income"] = train_df["income"].str.strip().str.rstrip(".")
        test_df["income"] = test_df["income"].str.strip().str.rstrip(".")

        train_targets = torch.as_tensor(
            np.where(train_df["income"] == ">50K", 1, 0).copy(), dtype=torch.long
        )
        test_targets = torch.as_tensor(
            np.where(test_df["income"] == ">50K", 1, 0).copy(), dtype=torch.long
        )
        train_df = train_df.drop(columns=["income", "fnlwgt"])
        test_df = test_df.drop(columns=["income", "fnlwgt"])

        # Fill missing categorical values with the training-set mode so test
        # imputation does not depend on test statistics.
        for col in train_df.select_dtypes(include="object").columns:
            mode = train_df[col].mode()[0]
            train_df[col] = train_df[col].fillna(mode)
            test_df[col] = test_df[col].fillna(mode)

        n_train = len(train_df)
        combined = pd.get_dummies(pd.concat([train_df, test_df], axis=0)).astype(float)
        train_data = torch.as_tensor(combined.iloc[:n_train].values.copy(), dtype=torch.float32)
        test_data = torch.as_tensor(combined.iloc[n_train:].values.copy(), dtype=torch.float32)
        self.num_features = train_data.shape[1]
        return train_data, train_targets, test_data, test_targets
