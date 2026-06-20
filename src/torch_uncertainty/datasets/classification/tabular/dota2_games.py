import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .base import TabularClassificationDataset


class DOTA2Games(TabularClassificationDataset):
    """The UCI DOTA 2 Games Results dataset.

    Predicts the winning team from hero selection in DOTA 2 matches. The
    dataset is provided pre-split; standardization statistics are computed
    from the training file.

    Note:
        The licenses of the datasets may differ from TorchUncertainty's
        license. Check before use.
    """

    md5_zip = "896623c082b062f56b9c49c6c1fc0bf7"
    url = "https://archive.ics.uci.edu/static/public/367/dota2+games+results.zip"
    dataset_name = "dota2_games"
    filename = "dota2Train.csv"
    num_features = 116
    need_split = False
    pre_split = True

    def _read(self, fname: str) -> pd.DataFrame:
        return pd.read_csv(self.root / self.dataset_name / fname, header=None)

    def _split(self, df: pd.DataFrame) -> tuple[Tensor, Tensor]:
        targets = torch.as_tensor(np.where(df.iloc[:, 0] == 1, 1, 0).copy(), dtype=torch.long)
        features = df.drop(columns=[0])
        data = torch.as_tensor(features.values.astype(float).copy(), dtype=torch.float32)
        return data, targets

    def _make_pre_split_dataset(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        train_data, train_targets = self._split(self._read("dota2Train.csv"))
        test_data, test_targets = self._split(self._read("dota2Test.csv"))
        self.num_features = train_data.shape[1]
        return train_data, train_targets, test_data, test_targets
