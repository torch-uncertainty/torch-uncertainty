import pandas as pd
import torch

from .base import TabularClassificationDataset


class SpamBase(TabularClassificationDataset):
    """The UCI SpamBase e-mail spam dataset.

    Classifies email messages as spam or non-spam from word and character
    frequency features.

    Note:
        The licenses of the datasets may differ from TorchUncertainty's
        license. Check before use.
    """

    md5_zip = "6159c57c5571b3c20218e32fc94e8e91"
    url = "https://archive.ics.uci.edu/static/public/94/spambase.zip"
    dataset_name = "spambase"
    filename = "spambase.data"
    num_features = 57

    def _make_dataset(self) -> None:
        data = pd.read_csv(self.root / self.dataset_name / self.filename, header=None)
        self.targets = torch.tensor(data.iloc[:, -1].to_numpy().copy(), dtype=torch.long)
        self.data = torch.tensor(data.iloc[:, :-1].to_numpy().copy(), dtype=torch.float32)
        self.num_features = self.data.shape[1]
