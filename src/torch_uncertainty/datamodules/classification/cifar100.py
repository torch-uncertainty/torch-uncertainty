from pathlib import Path
from typing import Literal

import numpy as np
import torch
from numpy.typing import ArrayLike
from timm.data.auto_augment import rand_augment_transform
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, SVHN
from torchvision.transforms import v2

from torch_uncertainty.datamodules import TUDataModule
from torch_uncertainty.datasets import AggregatedDataset
from torch_uncertainty.datasets.classification import CIFAR100C
from torch_uncertainty.datasets.utils import create_train_val_split
from torch_uncertainty.transforms import Cutout


class CIFAR100DataModule(TUDataModule):
    num_classes = 100
    num_channels = 3
    input_shape = (3, 32, 32)
    training_task = "classification"
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    def __init__(
        self,
        root: str | Path,
        batch_size: int,
        eval_batch_size: int | None = None,
        eval_ood: bool = False,
        eval_shift: bool = False,
        num_tta: int = 1,
        shift_severity: int = 1,
        val_split: float | None = None,
        postprocess_set: Literal["val", "test"] = "val",
        train_transform: nn.Module | None = None,
        test_transform: nn.Module | None = None,
        basic_augment: bool = True,
        cutout: int | None = None,
        randaugment: bool = False,
        auto_augment: str | None = None,
        num_dataloaders: int = 1,
        num_workers: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ) -> None:
        """DataModule for CIFAR100.

        Args:
            root (str): Root directory of the datasets.
            eval_ood (bool): Whether to evaluate out-of-distribution performance.
            eval_shift (bool): Whether to evaluate on shifted data. Defaults to ``False``.
            batch_size (int): Number of samples per batch during training.
            eval_batch_size (int | None) : Number of samples per batch during evaluation (val
                and test). Set to batch_size if ``None``. Defaults to ``None``.
            val_split (float): Share of samples to use for validation. Defaults
                to ``0.0``.
            postprocess_set (str, optional): The post-hoc calibration dataset to
                use for the post-processing method. Defaults to ``val``.
            train_transform (nn.Module | None): Custom training transform. Defaults
                to ``None``. If not provided, a default transform is used.
            test_transform (nn.Module | None): Custom test transform. Defaults to
                ``None``. If not provided, a default transform is used.
            basic_augment (bool): Whether to apply base augmentations. Defaults to
                ``True``. Only used if train_transform is not provided.
            cutout (int): Size of cutout to apply to images. Defaults to ``None``.
                Only used if train_transform is not provided.
            randaugment (bool): Whether to apply RandAugment. Defaults to
                ``False``. Only used if train_transform is not provided.
            auto_augment (str): Which auto-augment to apply. Defaults to ``None``.
            num_tta (int): Number of test-time augmentations (TTA). Defaults to ``1`` (no TTA).
            shift_severity (int): Severity of corruption to apply to CIFAR100-C. Defaults to ``1``.
            num_dataloaders (int): Number of dataloaders to use. Defaults to ``1``.
            num_workers (int): Number of workers to use for data loading. Defaults to ``1``.
            pin_memory (bool): Whether to pin memory. Defaults to ``True``.
            persistent_workers (bool): Whether to use persistent workers. Defaults to ``True``.
        """
        super().__init__(
            root=root,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            val_split=val_split,
            num_tta=num_tta,
            postprocess_set=postprocess_set,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        self.eval_ood = eval_ood
        self.eval_shift = eval_shift
        self.num_dataloaders = num_dataloaders

        self.dataset = CIFAR100
        self.ood_dataset = SVHN
        self.shift_dataset = CIFAR100C

        self.shift_severity = shift_severity

        if train_transform is not None:
            self.train_transform = train_transform
        else:
            if (cutout is not None) + randaugment + int(auto_augment is not None) > 1:
                raise ValueError(
                    "Only one data augmentation can be chosen at a time. Raise a "
                    "GitHub issue if needed."
                )

            if basic_augment:
                basic_transform = v2.Compose(
                    [
                        v2.RandomCrop(32, padding=4),
                        v2.RandomHorizontalFlip(),
                    ]
                )
            else:
                basic_transform = nn.Identity()

            if cutout:
                main_transform = Cutout(cutout)
            elif randaugment:
                main_transform = v2.RandAugment(num_ops=2, magnitude=20)
            elif auto_augment:
                main_transform = v2.Compose(
                    [
                        v2.ToPILImage(),
                        rand_augment_transform(auto_augment, {}),
                        v2.ToImage(),
                    ]
                )
            else:
                main_transform = nn.Identity()

            self.train_transform = v2.Compose(
                [
                    v2.ToImage(),
                    basic_transform,
                    main_transform,
                    v2.ToDtype(dtype=torch.float32, scale=True),
                    v2.Normalize(mean=self.mean, std=self.std),
                ]
            )

        if num_tta != 1:
            self.test_transform = self.train_transform
        elif test_transform is not None:
            self.test_transform = test_transform
        else:
            self.test_transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(dtype=torch.float32, scale=True),
                    v2.Normalize(mean=self.mean, std=self.std),
                ]
            )

    def prepare_data(self) -> None:  # coverage: ignore
        self.dataset(self.root, train=True, download=True)
        self.dataset(self.root, train=False, download=True)

        if self.eval_ood:
            self.ood_dataset(
                self.root,
                split="test",
                download=True,
                transform=self.test_transform,
            )
        if self.eval_shift:
            self.shift_dataset(
                self.root,
                download=True,
                transform=self.test_transform,
                shift_severity=self.shift_severity,
            )

    def setup(self, stage: Literal["fit", "test"] | None = None) -> None:
        if stage == "fit" or stage is None:
            full = self.dataset(
                self.root,
                train=True,
                download=False,
                transform=self.train_transform,
            )
            if self.val_split:
                self.train, self.val = create_train_val_split(
                    full,
                    self.val_split,
                    self.test_transform,
                )
            else:
                self.train = full
                self.val = self.dataset(
                    self.root,
                    train=False,
                    download=False,
                    transform=self.test_transform,
                )
        if stage == "test" or stage is None:
            self.test = self.dataset(
                self.root,
                train=False,
                download=False,
                transform=self.test_transform,
            )
            if self.eval_ood:
                self.ood = self.ood_dataset(
                    self.root,
                    split="test",
                    download=False,
                    transform=self.test_transform,
                )
            if self.eval_shift:
                self.shift = self.shift_dataset(
                    self.root,
                    download=False,
                    shift_severity=self.shift_severity,
                    transform=self.test_transform,
                )
        if stage not in ["fit", "test", None]:
            raise ValueError(f"Stage {stage} is not supported.")

    def train_dataloader(self) -> DataLoader:
        """Get the training dataloader for CIFAR100.

        Return:
            DataLoader: CIFAR100 training dataloader.
        """
        if self.num_dataloaders > 1:
            return self._data_loader(
                AggregatedDataset(self.train, self.num_dataloaders), shuffle=True, training=True
            )
        return self._data_loader(self.train, training=True, shuffle=True)

    def test_dataloader(self) -> list[DataLoader]:
        r"""Get test dataloaders.

        Return:
            list[DataLoader]: test set for in distribution data, SVHN data, and/or
            CIFAR-100C data.
        """
        dataloader = [self._data_loader(self.get_test_set(), training=False, shuffle=False)]
        if self.eval_ood:
            dataloader.append(self._data_loader(self.get_ood_set(), training=False, shuffle=False))
        if self.eval_shift:
            dataloader.append(
                self._data_loader(self.get_shift_set(), training=False, shuffle=False)
            )
        return dataloader

    def _get_train_data(self) -> ArrayLike:
        if self.val_split:
            return self.train.dataset.data[self.train.indices]
        return self.train.data

    def _get_train_targets(self) -> ArrayLike:
        if self.val_split:
            return np.array(self.train.dataset.targets)[self.train.indices]
        return np.array(self.train.targets)
