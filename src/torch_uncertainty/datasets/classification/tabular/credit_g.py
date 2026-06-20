import pandas as pd
import torch

from .base import TabularClassificationDataset


class GermanCredit(TabularClassificationDataset):
    """The UCI Statlog German Credit dataset.

    Predicts credit risk (good/bad). Reads the ``german.data`` file, whose
    categorical attributes are encoded as string codes (``A11``, ``A34`` ...)
    and are one-hot expanded via :func:`pandas.get_dummies`; numeric attributes
    are kept as-is.

    Reference:
        H. Hofmann, *Statlog (German Credit Data)*, UCI ML Repository, 1994.

    Note:
        The licenses of the datasets may differ from TorchUncertainty's
        license. Check before use.
    """

    url = "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"
    dataset_name = "german_credit"
    filename = "german.data"

    def _make_dataset(self) -> None:
        data = pd.read_csv(
            self.root / self.dataset_name / self.filename,
            sep=r"\s+",
            header=None,
        )
        # Last column: 1 = good credit → 0, 2 = bad credit → 1
        self.targets = torch.as_tensor((data.iloc[:, -1].values - 1).copy(), dtype=torch.long)
        data = data.iloc[:, :-1]
        self.data = torch.as_tensor(
            pd.get_dummies(data).astype(float).values.copy(), dtype=torch.float32
        )
        self.num_features = self.data.shape[1]
