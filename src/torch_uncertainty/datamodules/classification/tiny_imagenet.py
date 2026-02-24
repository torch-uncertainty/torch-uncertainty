from pathlib import Path
from typing import Literal

import numpy as np
import torch
from numpy.typing import ArrayLike
from timm.data.auto_augment import rand_augment_transform
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import DTD, SVHN
from torchvision.transforms import v2

from torch_uncertainty.datamodules import TUDataModule
from torch_uncertainty.datasets.classification import (
    ImageNetO,
    OpenImageO,
    TinyImageNet,
    TinyImageNetC,
)
from torch_uncertainty.datasets.utils import create_train_val_split
from torch_uncertainty.utils import (
    interpolation_modes_from_str,
)


class TinyImageNetDataModule(TUDataModule):
    num_classes = 200
    num_channels = 3
    training_task = "classification"
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def __init__(
        self,
        root: str | Path,
        batch_size: int,
        eval_batch_size: int | None = None,
        eval_ood: bool = False,
        eval_shift: bool = False,
        shift_severity: int = 1,
        val_split: float | None = None,
        num_tta: int = 1,
        postprocess_set: Literal["val", "test"] = "val",
        train_transform: nn.Module | None = None,
        test_transform: nn.Module | None = None,
        ood_ds: str = "svhn",
        interpolation: str = "bilinear",
        basic_augment: bool = True,
        rand_augment_opt: str | None = None,
        num_workers: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ) -> None:
        """DataModule for the Tiny-ImageNet dataset.

        This datamodule uses Tiny-ImageNet as In-distribution dataset, OpenImage-O, ImageNet-0,
        SVHN or DTD as Out-of-distribution dataset and Tiny-ImageNet-C as shifted dataset.

        Args:
            root (str): Root directory of the datasets.
            batch_size (int): Number of samples per batch during training.
            eval_batch_size (int | None) : Number of samples per batch during evaluation (val
                and test). Set to :attr:`batch_size` if ``None``. Defaults to ``None``.
            eval_ood (bool): Whether to evaluate out-of-distribution performance. Defaults to ``False``.
            eval_shift (bool): Whether to evaluate on shifted data. Defaults to ``False``.
            num_tta (int): Number of test-time augmentations (TTA). Defaults to ``1`` (no TTA).
            shift_severity (int): Severity of the shift. Defaults to ``1``.
            val_split (float or Path): Share of samples to use for validation
                or path to a yaml file containing a list of validation images
                ids. Defaults to ``0.0``.
            postprocess_set (str, optional): The post-hoc calibration dataset to
                use for the post-processing method. Defaults to ``val``.
            train_transform (nn.Module | None): Custom training transform. Defaults
                to ``None``. If not provided, a default transform is used.
            test_transform (nn.Module | None): Custom test transform. Defaults to
                ``None``. If not provided, a default transform is used.
            ood_ds (str): Which out-of-distribution dataset to use. Defaults to
                ``"openimage-o"``.
            test_alt (str): Which test set to use. Defaults to ``None``.
            procedure (str): Which procedure to use. Defaults to ``None``.
            train_size (int): Size of training images. Defaults to ``224``.
            interpolation (str): Interpolation method for the Resize Crops. Defaults to ``"bilinear"``.
            basic_augment (bool): Whether to apply base augmentations. Defaults to ``True``.
            rand_augment_opt (str): Which RandAugment to use. Defaults to ``None``.
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
        self.shift_severity = shift_severity

        self.ood_ds = ood_ds
        self.interpolation = interpolation_modes_from_str(interpolation)

        self.dataset = TinyImageNet

        if ood_ds == "imagenet-o":
            self.ood_dataset = ImageNetO
        elif ood_ds == "svhn":
            self.ood_dataset = SVHN
        elif ood_ds == "textures":
            self.ood_dataset = DTD
        elif ood_ds == "openimage-o":
            self.ood_dataset = OpenImageO
        else:
            raise ValueError(f"OOD dataset {ood_ds} not supported for TinyImageNet.")
        self.shift_dataset = TinyImageNetC

        if train_transform is not None:
            self.train_transform = train_transform
        else:
            if basic_augment:
                basic_transform = v2.Compose(
                    [
                        v2.RandomCrop(64, padding=4),
                        v2.RandomHorizontalFlip(),
                    ]
                )
            else:
                basic_transform = nn.Identity()

            if rand_augment_opt is not None:
                main_transform = v2.Compose(
                    [
                        v2.ToPILImage(),
                        rand_augment_transform(rand_augment_opt, {}),
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
                    v2.Resize(64, interpolation=self.interpolation),
                    v2.ToDtype(dtype=torch.float32, scale=True),
                    v2.Normalize(mean=self.mean, std=self.std),
                ]
            )

    def _verify_splits(self, split: str) -> None:  # coverage: ignore
        if split not in list(self.root.iterdir()):
            raise FileNotFoundError(
                f"a {split} TinyImagenet split was not found in {self.root},"
                f" make sure the folder contains a subfolder named {split}"
            )

    def prepare_data(self) -> None:  # coverage: ignore
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
                split="train",
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
                    split="val",
                    transform=self.test_transform,
                )
        if stage == "test" or stage is None:
            self.test = self.dataset(
                self.root,
                split="val",
                transform=self.test_transform,
            )
        if stage not in ["fit", "test", None]:
            raise ValueError(f"Stage {stage} is not supported.")

        if self.eval_ood:
            self.ood = self.ood_dataset(
                self.root,
                split="test",
                transform=self.test_transform,
            )

        if self.eval_shift:
            self.shift = self.shift_dataset(
                self.root,
                download=False,
                shift_severity=self.shift_severity,
                transform=self.test_transform,
            )

    def test_dataloader(self) -> list[DataLoader]:
        r"""Get test dataloaders for TinyImageNet.

        Return:
            list[DataLoader]: test set for in distribution data, OOD data, and/or
            TinyImageNetC data.
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
            return self.train.dataset.samples[self.train.indices]
        return self.train.samples

    def _get_train_targets(self) -> ArrayLike:
        if self.val_split:
            return np.array(self.train.dataset.label_data)[self.train.indices]
        return np.array(self.train.label_data)
