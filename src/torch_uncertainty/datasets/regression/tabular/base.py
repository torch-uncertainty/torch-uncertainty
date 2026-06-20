import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Generator, Tensor, tensor
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive, download_url

from torch_uncertainty.datasets.utils import load_arff  # noqa: F401


class TabularRegressionDataset(Dataset, ABC):
    """Abstract base class for UCI regression datasets.

    Mirrors :class:`~torch_uncertainty.datasets.classification.tabular.TabularClassificationDataset`:
    subclasses set :attr:`url`, :attr:`filename`, :attr:`dataset_name` and
    implement :meth:`_make_dataset`. Standardization is computed from the
    training split only and then applied to both splits.
    """

    root_appendix = "uci_regression"
    url: str = ""
    filename: str = ""
    dataset_name: str = ""
    need_split: bool = True
    apply_standardization: bool = True
    is_archive: bool = True
    md5: str | None = None

    def __init__(
        self,
        root: Path | str,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
        train: bool = True,
        test_split: float = 0.2,
        split_seed: int = 42,
    ) -> None:
        """UCI regression dataset.

        Args:
            root: Root directory of the datasets.
            transform: Transform applied to each input sample. Defaults to ``None``.
            target_transform: Transform applied to each target. Defaults to ``None``.
            download: If ``True``, downloads the dataset. Defaults to ``False``.
            train: If ``True``, returns the training split. Defaults to ``True``.
            test_split: Fraction of the dataset held out as test set. Defaults to ``0.2``.
            split_seed: Random seed for the train/test split. Defaults to ``42``.

        Note:
            The licenses of the datasets may differ from TorchUncertainty's
            license. Check before use.
        """
        super().__init__()
        self.root = Path(root)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        self._make_dataset()

        if self.need_split:
            gen = Generator().manual_seed(split_seed)
            train_idx = torch.ones(len(self)).multinomial(
                num_samples=int((1 - test_split) * len(self)),
                replacement=False,
                generator=gen,
            )
            test_idx = tensor([i for i in range(len(self)) if i not in train_idx])
            if self.apply_standardization:
                self._compute_statistics(self.data[train_idx], self.targets[train_idx])
                self._standardize()
            self.split_idx = train_idx if self.train else test_idx
            self.data = self.data[self.split_idx]
            self.targets = self.targets[self.split_idx]
        elif self.apply_standardization:
            self._compute_statistics()
            self._standardize()

    @property
    def _data_path(self) -> Path:
        return self.root / self.root_appendix / self.dataset_name

    def __len__(self) -> int:
        """Get the length of the tabular regression dataset."""
        return self.data.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Get an element of the tabular regression dataset."""
        data = self.data[index]
        if self.transform is not None:
            data = self.transform(data)
        target = self.targets[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target

    def _check_integrity(self) -> bool:
        return (self._data_path / self.filename).is_file()

    def _compute_statistics(
        self,
        data: Tensor | None = None,
        targets: Tensor | None = None,
    ) -> None:
        # Use float64 to avoid precision loss for large-valued features when
        # computing the mean (e.g. NavalPropulsionPlant columns at ~1e9).
        d = (self.data if data is None else data).double()
        t = (self.targets if targets is None else targets).double()
        self.data_mean = d.mean(dim=0).float()
        self.data_std = d.std(dim=0).float()
        self.data_std[self.data_std == 0] = 1
        self.target_mean = t.mean(dim=0).float()
        self.target_std = t.std(dim=0).float()
        self.target_std[self.target_std == 0] = 1

    def _standardize(self) -> None:
        self.data = (self.data - self.data_mean) / self.data_std
        self.targets = (self.targets - self.target_mean) / self.target_std

    def download(self) -> None:
        """Download and, if needed, extract the dataset."""
        if self._check_integrity():
            logging.info("Files already downloaded and verified")
            return
        self._data_path.mkdir(parents=True, exist_ok=True)
        if self.is_archive:
            download_and_extract_archive(
                self.url,
                download_root=self._data_path,
                filename=self.dataset_name + ".zip",
                md5=self.md5,
            )
        else:
            download_url(
                self.url,
                root=str(self._data_path),
                filename=self.filename,
                md5=self.md5,
            )

    @abstractmethod
    def _make_dataset(self) -> None:
        """Populate ``self.data`` and ``self.targets`` as float32 tensors."""
