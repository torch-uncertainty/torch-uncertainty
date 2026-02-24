import copy
from pathlib import Path
from typing import Literal

import torch
import yaml
from timm.data.auto_augment import rand_augment_transform
from timm.data.mixup import Mixup
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import DTD, SVHN, ImageNet, INaturalist
from torchvision.transforms import v2

from torch_uncertainty.datamodules import TUDataModule
from torch_uncertainty.datasets.classification import (
    ImageNetA,
    ImageNetC,
    ImageNetO,
    ImageNetR,
    OpenImageO,
)
from torch_uncertainty.datasets.utils import create_train_val_split
from torch_uncertainty.utils import (
    interpolation_modes_from_str,
)


class ImageNetDataModule(TUDataModule):
    num_classes = 1000
    num_channels = 3
    test_datasets = ["r", "o", "a"]
    ood_datasets = [
        "inaturalist",
        "imagenet-o",
        "svhn",
        "textures",
        "openimage-o",
    ]
    training_task = "classification"
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_indices = None
    val_indices = None

    def __init__(
        self,
        root: str | Path,
        batch_size: int,
        eval_batch_size: int | None = None,
        eval_ood: bool = False,
        eval_shift: bool = False,
        num_tta: int = 1,
        shift_severity: int = 1,
        val_split: float | Path | None = None,
        postprocess_set: Literal["val", "test"] = "val",
        train_transform: nn.Module | None = None,
        test_transform: nn.Module | None = None,
        ood_ds: str = "openimage-o",
        test_alt: str | None = None,
        procedure: str | None = None,
        train_size: int = 224,
        interpolation: str = "bilinear",
        basic_augment: bool = True,
        rand_augment_opt: str | None = None,
        num_workers: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ) -> None:
        """DataModule for the ImageNet dataset.

        This datamodule uses ImageNet as In-distribution dataset, OpenImage-O, INaturalist,
        ImageNet-0, SVHN or DTD as Out-of-distribution dataset and ImageNet-C as shifted dataset.

        Args:
            root (str): Root directory of the datasets.
            batch_size (int): Number of samples per batch during training.
            eval_batch_size (int | None) : Number of samples per batch during evaluation (val
                and test). Set to batch_size if ``None``. Defaults to ``None``.
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
                Only used if ``train_transform`` is not provided.
            train_size (int): Size of training images. Defaults to ``224``.
            interpolation (str): Interpolation method for the Resize Crops.
                Defaults to ``"bilinear"``. Only used if ``train_transform`` is not
                provided.
            basic_augment (bool): Whether to apply base augmentations. Defaults to
                ``True``. Only used if ``train_transform`` is not provided.
            rand_augment_opt (str): Which RandAugment to use. Defaults to ``None``.
                Only used if ``train_transform`` is not provided.
            num_workers (int): Number of workers to use for data loading. Defaults
                to ``1``.
            pin_memory (bool): Whether to pin memory. Defaults to ``True``.
            persistent_workers (bool): Whether to use persistent workers. Defaults to ``True``.
        """
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

        self.eval_ood = eval_ood
        self.eval_shift = eval_shift
        self.shift_severity = shift_severity
        if val_split and not isinstance(val_split, float):
            val_split = Path(val_split)
            self.train_indices, self.val_indices = read_indices(val_split)
        self.val_split = val_split
        self.ood_ds = ood_ds
        self.test_alt = test_alt
        self.interpolation = interpolation_modes_from_str(interpolation)

        if test_alt is None:
            self.dataset = ImageNet
        elif test_alt == "r":
            self.dataset = ImageNetR
        elif test_alt == "o":
            self.dataset = ImageNetO
        elif test_alt == "a":
            self.dataset = ImageNetA
        else:
            raise ValueError(f"The alternative {test_alt} is not known.")

        if ood_ds == "inaturalist":
            self.ood_dataset = INaturalist
        elif ood_ds == "imagenet-o":
            self.ood_dataset = ImageNetO
        elif ood_ds == "svhn":
            self.ood_dataset = SVHN
        elif ood_ds == "textures":
            self.ood_dataset = DTD
        elif ood_ds == "openimage-o":
            self.ood_dataset = OpenImageO
        else:
            raise ValueError(f"The dataset {ood_ds} is not supported.")
        self.shift_dataset = ImageNetC

        self.procedure = procedure

        if train_transform is not None:
            self.train_transform = train_transform
        else:
            if basic_augment:
                basic_transform = v2.Compose(
                    [
                        v2.RandomResizedCrop(train_size, interpolation=self.interpolation),
                        v2.RandomHorizontalFlip(),
                    ]
                )
            else:
                basic_transform = nn.Identity()

            if self.procedure is None:
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
            elif self.procedure == "ViT":
                train_size = 224
                main_transform = v2.Compose(
                    [
                        v2.ToPILImage(),
                        Mixup(mixup_alpha=0.2, cutmix_alpha=1.0),
                        rand_augment_transform("rand-m9-n2-mstd0.5", {}),
                        v2.ToImage(),
                    ]
                )
            elif self.procedure == "A3":
                train_size = 160
                main_transform = v2.Compose(
                    [
                        v2.ToPILImage(),
                        rand_augment_transform("rand-m6-mstd0.5-inc1", {}),
                        v2.ToImage(),
                    ]
                )
            else:
                raise ValueError("The procedure is unknown")

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
                    v2.Resize(256, interpolation=self.interpolation),
                    v2.CenterCrop(224),
                    v2.ToDtype(dtype=torch.float32, scale=True),
                    v2.Normalize(mean=self.mean, std=self.std),
                ]
            )

    def _verify_splits(self, split: str) -> None:
        if split not in list(self.root.iterdir()):
            raise FileNotFoundError(
                f"a {split} Imagenet split was not found in {self.root},"
                f" make sure the folder contains a subfolder named {split}"
            )

    def prepare_data(self) -> None:  # coverage: ignore
        if self.test_alt is not None:
            self.data = self.dataset(
                self.root,
                split="val",
                download=True,
            )
        if self.eval_ood:
            if self.ood_ds == "inaturalist":
                self.ood = self.ood_dataset(
                    self.root,
                    version="2021_valid",
                    download=True,
                    transform=self.test_transform,
                )
            elif self.ood_ds != "textures":
                self.ood = self.ood_dataset(
                    self.root,
                    split="test",
                    download=True,
                    transform=self.test_transform,
                )
            else:
                self.ood = self.ood_dataset(
                    self.root,
                    split="train",
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
            if self.test_alt is not None:
                raise ValueError("The test_alt argument is not supported for training.")
            full = self.dataset(
                self.root,
                split="train",
                transform=self.train_transform,
            )
            if self.val_split and isinstance(self.val_split, float):
                self.train, self.val = create_train_val_split(
                    full,
                    self.val_split,
                    self.test_transform,
                )
            elif isinstance(self.val_split, Path):
                self.train = Subset(full, self.train_indices)
                # TODO: improve the performance
                self.val = copy.deepcopy(Subset(full, self.val_indices))
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
        if stage not in ["fit", "test", None]:
            raise ValueError(f"Stage {stage} is not supported.")

        if self.eval_ood:
            if self.ood_ds == "inaturalist":
                self.ood = self.ood_dataset(
                    self.root,
                    version="2021_valid",
                    transform=self.test_transform,
                )
            else:
                self.ood = self.ood_dataset(
                    self.root,
                    transform=self.test_transform,
                    download=True,
                )

        if self.eval_shift:
            self.shift = self.shift_dataset(
                self.root,
                download=False,
                transform=self.test_transform,
                shift_severity=self.shift_severity,
            )

    def test_dataloader(self) -> list[DataLoader]:
        """Get the test dataloaders for ImageNet.

        Return:
            list[DataLoader]: ImageNet test set (in distribution data), OOD dataset test split
            (out-of-distribution data), and/or ImageNetC data.
        """
        dataloader = [self._data_loader(self.get_test_set(), training=False, shuffle=False)]
        if self.eval_ood:
            dataloader.append(self._data_loader(self.get_ood_set(), training=False, shuffle=False))
        if self.eval_shift:
            dataloader.append(
                self._data_loader(self.get_shift_set(), training=False, shuffle=False)
            )
        return dataloader


def read_indices(path: Path) -> list[str]:  # coverage: ignore
    """Read a file and return its lines as a list.

    Args:
        path (Path): Path to the file.

    Returns:
        list[str]: list of filenames.
    """
    if not path.is_file():
        raise ValueError(f"{path} is not a file.")
    with path.open("r") as f:
        indices = yaml.safe_load(f)
        return indices["train"], indices["val"]
