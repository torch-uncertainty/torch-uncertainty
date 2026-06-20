import pandas as pd
import torch

from .base import TabularClassificationDataset


class HTRU2(TabularClassificationDataset):
    """The HTRU2 pulsar candidate dataset.

    Predicts whether a radio pulsar candidate is a genuine pulsar from
    integrated pulse profile and DM-SNR curve statistics.

    Reference:
        R.J. Lyon et al., *Fifty Years of Pulsar Candidate Selection*, MNRAS, 2016.

    Note:
        The licenses of the datasets may differ from TorchUncertainty's
        license. Check before use.
    """

    md5_zip = "1cfbf71c604debc06dedcbb6c1ccb43f"
    url = "https://archive.ics.uci.edu/static/public/372/htru2.zip"
    dataset_name = "htru2"
    filename = "HTRU_2.csv"
    num_features = 8

    def _make_dataset(self) -> None:
        data = pd.read_csv(self.root / self.dataset_name / self.filename, header=None)
        self.targets = torch.tensor(data.iloc[:, -1].to_numpy().copy(), dtype=torch.long)
        self.data = torch.tensor(data.iloc[:, :-1].to_numpy().copy(), dtype=torch.float32)
        self.num_features = self.data.shape[1]
