import pandas as pd
import torch

from .base import TabularClassificationDataset


class OnlineShoppers(TabularClassificationDataset):
    """The UCI Online Shoppers Purchasing Intention dataset.

    Predicts whether a browsing session results in a purchase from user
    behaviour and page metrics.

    Note:
        The licenses of the datasets may differ from TorchUncertainty's
        license. Check before use.
    """

    md5_zip = "d835049e5f428f3b8cb8a6e6937f5537"
    url = "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip"
    dataset_name = "online_shoppers"
    filename = "online_shoppers_intention.csv"
    num_features = 28

    def _make_dataset(self) -> None:
        data = pd.read_csv(
            self.root / self.dataset_name / self.filename,
            true_values=["TRUE"],
            false_values=["FALSE"],
        )
        self.targets = torch.as_tensor(data["Revenue"].values.copy(), dtype=torch.long)
        data = pd.get_dummies(data).astype(float)
        data = data.drop(columns=["Revenue"])
        self.data = torch.as_tensor(data.values.copy(), dtype=torch.float32)
        self.num_features = self.data.shape[1]
