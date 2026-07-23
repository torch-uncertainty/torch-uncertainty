import copy
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import yaml
from numpy.typing import ArrayLike
from timm.data.auto_augment import rand_augment_transform
from torch import nn
from torch.utils.data import Subset
from torchvision.transforms import v2

from torch_uncertainty.datamodules import TUDataModule
from torch_uncertainty.datasets.classification import ImageNet200
from torch_uncertainty.datasets.utils import create_train_val_split
from torch_uncertainty.utils import interpolation_modes_from_str


class ImageNet200DataModule(TUDataModule):
    """DataModule for the ImageNet-200 subset of ImageNet-1K.

    ImageNet-200 uses the 200 ImageNet-1K classes represented in ImageNet-R. The
    underlying images are loaded from an existing torchvision-compatible ImageNet
    directory. This datamodule only provides in-distribution train, validation, and
    test loaders.

    Args:
        root: Root directory of the ImageNet dataset.
        batch_size: Number of samples per training batch.
        eval_batch_size: Number of samples per validation and test batch. Set to
            ``batch_size`` when ``None``. Defaults to ``None``.
        val_split: Share of training samples used for validation, or a path to a YAML
            file containing ``train`` and ``val`` index lists. When ``None``, the
            ImageNet validation split is used. Defaults to ``None``.
        num_tta: Number of test-time augmentations. Defaults to ``1``.
        postprocess_set: Split used to fit post-processing methods. Defaults to
            ``"val"``.
        train_transform: Custom training transform. A default ImageNet transform is
            used when ``None``. Defaults to ``None``.
        test_transform: Custom evaluation transform. A default ImageNet transform is
            used when ``None``. Defaults to ``None``.
        train_size: Size of the random crop used for training. Defaults to ``224``.
        interpolation: Interpolation mode used by resize operations. Defaults to
            ``"bilinear"``.
        basic_augment: Whether to apply random resized crop and horizontal flip when
            using the default training transform. Defaults to ``True``.
        rand_augment_opt: timm RandAugment configuration string. Defaults to ``None``.
        num_workers: Number of data-loading workers. Defaults to ``1``.
        pin_memory: Whether to pin memory in data loaders. Defaults to ``True``.
        persistent_workers: Whether data-loading workers persist across epochs.
            Defaults to ``True``.
    """

    num_classes = 200
    num_channels = 3
    input_shape = (3, 224, 224)
    training_task = "classification"
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    dataset: type[ImageNet200]

    def __init__(
        self,
        root: str | Path,
        batch_size: int,
        eval_batch_size: int | None = None,
        val_split: float | str | Path | None = None,
        num_tta: int = 1,
        postprocess_set: Literal["val", "test"] = "val",
        train_transform: nn.Module | None = None,
        test_transform: nn.Module | None = None,
        train_size: int = 224,
        interpolation: str = "bilinear",
        basic_augment: bool = True,
        rand_augment_opt: str | None = None,
        num_workers: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ) -> None:
        super().__init__(
            root=Path(root),
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            val_split=val_split,
            num_tta=num_tta,
            postprocess_set=postprocess_set,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        self.train_indices: list[int] | None = None
        self.val_indices: list[int] | None = None
        if val_split is not None and not isinstance(val_split, float):
            val_split = Path(val_split)
            self.train_indices, self.val_indices = _read_indices(val_split)
        self.val_split = val_split
        self.dataset = ImageNet200

        interpolation_mode = interpolation_modes_from_str(interpolation)

        if train_transform is not None:
            self.train_transform = train_transform
        else:
            if basic_augment:
                basic_transform = v2.Compose(
                    [
                        v2.RandomResizedCrop(train_size, interpolation=interpolation_mode),
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
                    v2.Resize(256, interpolation=interpolation_mode),
                    v2.CenterCrop(224),
                    v2.ToDtype(dtype=torch.float32, scale=True),
                    v2.Normalize(mean=self.mean, std=self.std),
                ]
            )

    def _verify_splits(self, split: str) -> None:
        if not (self.root / split).is_dir():
            raise FileNotFoundError(
                f"An ImageNet {split} split was not found in {self.root}. "
                f"Make sure {self.root / split} exists."
            )

    def prepare_data(self) -> None:
        """ImageNet must be downloaded manually."""

    def setup(self, stage: str | None = None) -> None:
        if stage not in ("fit", "test", None):
            raise ValueError(f"Stage {stage} is not supported.")

        if stage == "fit" or stage is None:
            full = self.dataset(
                self.root,
                split="train",
                transform=self.train_transform,
            )
            if isinstance(self.val_split, float) and self.val_split:
                self.train, self.val = create_train_val_split(
                    full,
                    self.val_split,
                    self.test_transform,
                )
            elif isinstance(self.val_split, Path):
                self.train = Subset(full, self.train_indices or [])
                self.val = copy.deepcopy(Subset(full, self.val_indices or []))
                self.val.dataset.transform = self.test_transform
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

    def _get_train_data(self) -> ArrayLike:
        if isinstance(self.train, Subset):
            samples = np.asarray(self.train.dataset.samples, dtype=object)
            return samples[self.train.indices]
        return np.asarray(self.train.samples, dtype=object)

    def _get_train_targets(self) -> ArrayLike:
        if isinstance(self.train, Subset):
            targets = np.asarray(self.train.dataset.targets)
            return targets[self.train.indices]
        return np.asarray(self.train.targets)


def _read_indices(path: Path) -> tuple[list[int], list[int]]:
    if not path.is_file():
        raise ValueError(f"{path} is not a file.")
    with path.open() as file:
        indices = yaml.safe_load(file)
    return indices["train"], indices["val"]
