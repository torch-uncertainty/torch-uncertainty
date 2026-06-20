import pandas as pd
import torch
from torch import Tensor

from .base import TabularClassificationDataset


class APSFailure(TabularClassificationDataset):
    """The UCI APS Failure at Scania Trucks dataset.

    Predicts whether an air pressure system (APS) component caused a truck
    failure. The dataset is provided pre-split; ``train=False`` loads the
    held-out test set. Missing values (``na``) are imputed with the
    training-set column mean for both splits.

    Reference:
        M. Cerqueira et al., *Predicting Failures in Industrial Plants*,
        UCI ML Repository, 2016.

    Note:
        The licenses of the datasets may differ from TorchUncertainty's
        license. Check before use.
    """

    url = "https://archive.ics.uci.edu/static/public/421/aps+failure+at+scania+trucks.zip"
    dataset_name = "aps_failure"
    filename = "aps_failure_training_set.csv"
    need_split = False
    pre_split = True

    def _read(self, fname: str) -> pd.DataFrame:
        # The CSV files begin with ~20 comment lines followed by the header row.
        return pd.read_csv(
            self.root / self.dataset_name / fname,
            na_values=["na"],
            comment=None,
            header=0,
            skiprows=20,
        )

    def _make_pre_split_dataset(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        train_df = self._read("aps_failure_training_set.csv")
        test_df = self._read("aps_failure_test_set.csv")

        train_targets = torch.tensor(
            (train_df["class"] == "pos").astype(int).to_numpy().copy(), dtype=torch.long
        )
        test_targets = torch.tensor(
            (test_df["class"] == "pos").astype(int).to_numpy().copy(), dtype=torch.long
        )
        train_df = train_df.drop(columns=["class"])
        test_df = test_df.drop(columns=["class"])

        train_df = train_df.apply(pd.to_numeric, errors="coerce")
        test_df = test_df.apply(pd.to_numeric, errors="coerce")
        # Impute with training-set column mean
        train_means = train_df.mean()
        train_df = train_df.fillna(train_means)
        test_df = test_df.fillna(train_means)

        train_data = torch.tensor(train_df.to_numpy(dtype=float).copy(), dtype=torch.float32)
        test_data = torch.tensor(test_df.to_numpy(dtype=float).copy(), dtype=torch.float32)
        self.num_features = train_data.shape[1]
        return train_data, train_targets, test_data, test_targets
