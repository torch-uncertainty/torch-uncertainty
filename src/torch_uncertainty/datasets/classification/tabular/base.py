import logging
from abc import ABC
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Generator, Tensor
from torch.utils.data import Dataset
from torchvision.datasets.utils import (
    download_and_extract_archive,
    download_url,
)

from torch_uncertainty.datasets.utils import load_arff  # noqa: F401


class TabularClassificationDataset(Dataset, ABC):
    """Base class for tabular binary classification datasets.

    Subclasses must define the class attributes :attr:`url`, :attr:`filename`,
    :attr:`dataset_name` and implement :meth:`_make_dataset`.

    If the source is a plain file rather than a zip archive, set
    :attr:`is_archive` to ``False``; :meth:`download` will then call
    :func:`~torchvision.datasets.utils.download_url` instead of
    :func:`~torchvision.datasets.utils.download_and_extract_archive`.
    """

    md5_zip: str | None = None
    url: str = ""
    filename: str = ""
    dataset_name: str = ""
    num_features: int = 0
    need_split: bool = True
    pre_split: bool = False
    apply_standardization: bool = True
    is_archive: bool = True

    def __init__(
        self,
        root: Path | str,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        binary: bool = True,
        download: bool = False,
        train: bool = True,
        test_split: float = 0.2,
        split_seed: int = 21893027,
        download_only: bool = False,
    ) -> None:
        """Tabular binary classification dataset.

        Args:
            root: Root directory of the datasets.
            transform: A function/transform that takes in a
                tensor and returns a transformed version. Defaults to ``None``.
            target_transform: A function/transform that takes
                in the target and transforms it. Defaults to ``None``.
            binary: If ``True``, returns scalar targets; otherwise
                one-hot encodes them into two classes. Defaults to ``True``.
            download: If ``True``, downloads the dataset from the
                internet. If already present, it is not downloaded again. Defaults
                to ``False``.
            train: If ``True``, use the training split. Defaults to ``True``.
            test_split: Fraction of the dataset held out as test
                set when :attr:`need_split` is ``True``. Defaults to ``0.2``.
            split_seed: Random seed for the train/test split. Defaults to ``21893027``.
            download_only: If ``True``, only download the dataset and skip the
                feature processing. Useful when only triggering the download.
                Defaults to ``False``.

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

        if download_only:
            return

        if self.pre_split:
            train_data, train_targets, test_data, test_targets = self._make_pre_split_dataset()
            if self.apply_standardization:
                self._compute_statistics(train_data)
                train_data = self._standardize_tensor(train_data)
                test_data = self._standardize_tensor(test_data)
            if self.train:
                self.data, self.targets = train_data, train_targets
            else:
                self.data, self.targets = test_data, test_targets
        else:
            self._make_dataset()
            if self.need_split:
                gen = Generator().manual_seed(split_seed)
                num_train = int((1 - test_split) * len(self))
                train_idx = torch.ones(len(self)).multinomial(
                    num_samples=num_train,
                    replacement=False,
                    generator=gen,
                )
                mask = torch.zeros(len(self), dtype=torch.bool)
                mask[train_idx] = True
                test_idx = torch.nonzero(~mask, as_tuple=False).squeeze(1)
                if self.apply_standardization:
                    # Compute statistics from the training split only to avoid
                    # leaking test-set information into the normalization.
                    self._compute_statistics(self.data[train_idx])
                    self._standardize()
                self.split_idx = train_idx if self.train else test_idx
                self.data = self.data[self.split_idx]
                self.targets = self.targets[self.split_idx]
            elif self.apply_standardization:
                self._compute_statistics()
                self._standardize()

        self._postprocess_targets(binary)

    def __len__(self) -> int:
        """Get the number of rows of the tabular data."""
        return self.data.shape[0]

    def _check_integrity(self) -> bool:
        return (self.root / self.dataset_name / self.filename).is_file()

    def _standardize_tensor(self, data: Tensor) -> Tensor:
        out = (data - self.data_mean) / self.data_std
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _standardize(self) -> None:
        self.data = self._standardize_tensor(self.data)

    def _compute_statistics(self, data: Tensor | None = None) -> None:
        d = self.data if data is None else data
        # Ignore NaN values when computing statistics so a single residual NaN
        # doesn't poison the entire column.
        self.data_mean = torch.nanmean(d, dim=0)
        # nanstd: var via nanmean of squared deviations
        centered = d - self.data_mean
        n_valid = (~torch.isnan(d)).sum(dim=0).clamp(min=1)
        var = torch.nansum(centered**2, dim=0) / (n_valid - (n_valid > 1).long()).clamp(min=1)
        self.data_std = var.sqrt()
        self.data_std[~torch.isfinite(self.data_std)] = 1
        self.data_std[self.data_std == 0] = 1

    def download(self) -> None:
        """Download and, if needed, extract the dataset."""
        if self._check_integrity():
            logging.info("Files already downloaded and verified")
            return
        download_root = self.root / self.dataset_name
        if self.is_archive:
            download_and_extract_archive(
                self.url,
                download_root=download_root,
                filename=self.dataset_name + ".zip",
                md5=self.md5_zip,
            )
        else:
            download_url(
                self.url,
                root=str(download_root),
                filename=self.filename,
                md5=self.md5_zip,
            )

    def _postprocess_targets(self, binary: bool) -> None:
        """Post-process targets after splitting.

        The default behaviour one-hot encodes targets into two classes when
        ``binary`` is ``False``. Override this in subclasses that require
        different target handling (e.g. multi-class datasets).
        """
        if not binary:
            self.targets = torch.nn.functional.one_hot(self.targets, num_classes=2)

    def _make_dataset(self) -> None:
        """Populate ``self.data`` (float32 tensor) and ``self.targets`` (long tensor).

        Required for datasets without pre-existing train/test files. Datasets
        with separate train/test files should instead set :attr:`pre_split` and
        implement :meth:`_make_pre_split_dataset`.
        """
        raise NotImplementedError

    def _make_pre_split_dataset(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Load both train and test files and return ``(train_data, train_targets, test_data, test_targets)``.

        Implementations must align feature columns between train and test (e.g.
        by concatenating before one-hot encoding) and use training-set statistics
        for any imputation.
        """
        raise NotImplementedError

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Get the row of id index of the tabular data."""
        data = self.data[index, :]
        if self.transform is not None:
            data = self.transform(data)
        target = self.targets[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target
