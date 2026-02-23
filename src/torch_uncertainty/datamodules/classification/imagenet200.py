import logging
from pathlib import Path
from typing import Literal

import torch
import yaml
from timm.data.auto_augment import rand_augment_transform
from timm.data.mixup import Mixup
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

from torch_uncertainty.datamodules import TUDataModule
from torch_uncertainty.datasets.classification import (
    ImageNetA,
    ImageNetC,
    ImageNetO,
    ImageNetR,
)
from torch_uncertainty.datasets.ood.utils import (
    FileListDataset,
    download_and_extract_hf_dataset,
    download_and_extract_splits_from_hf,
    get_ood_datasets,
)
from torch_uncertainty.utils import (
    interpolation_modes_from_str,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logging.getLogger("faiss").setLevel(logging.WARNING)


class ImageNet200DataModule(TUDataModule):
    num_classes = 200
    num_channels = 3
    test_datasets = ["r", "o", "a"]
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
        shift_severity: int = 1,
        val_split: float | Path | None = None,
        postprocess_set: Literal["val", "test"] = "val",
        test_alt: str | None = None,
        procedure: str | None = None,
        train_size: int = 224,
        interpolation: str = "bilinear",
        basic_augment: bool = True,
        rand_augment_opt: str | None = None,
        num_workers: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        near_ood_datasets: list | None = None,
        far_ood_datasets: list | None = None,
    ) -> None:
        """DataModule for the ImageNet200 dataset.

        This datamodule uses ImageNet200 as In-distribution dataset, OpenImage-O, INaturalist,
        ImageNet-0, SVHN or DTD as Out-of-distribution dataset and ImageNet-C as shifted dataset.

        Args:
            root (str): Root directory of the datasets.
            batch_size (int): Number of samples per batch during training.
            eval_batch_size (int | None) : Number of samples per batch during evaluation (val
                and test). Set to batch_size if None. Defaults to None.
            eval_ood (bool): Whether to evaluate out-of-distribution performance. Defaults to ``False``.
            near_ood_datasets (list, optional): list of near OOD dataset classes must be subclass of torch.utils.data.Dataset. Defaults to SSB-hard, NINCO (OpenOOD splits)
            far_ood_datasets (list, optional): list of far OOD dataset classes must be subclass of torch.utils.data.Dataset. Defaults to iNaturalist, Textures, OpenImage-O (OpenOOD splits)
            eval_shift (bool): Whether to evaluate on shifted data. Defaults to ``False``.
            shift_severity (int): Severity of the shift. Defaults to ``1``.
            val_split (float or Path): Share of samples to use for validation
                or path to a yaml file containing a list of validation images
                ids. Defaults to ``0.0``.
            postprocess_set (str, optional): The post-hoc calibration dataset to
                use for the post-processing method. Defaults to ``val``.
            ood_ds (str): Which out-of-distribution dataset to use. Defaults to
                ``"openimage-o"``.
            test_alt (str): Which test set to use. Defaults to ``None``.
            procedure (str): Which procedure to use. Defaults to ``None``.
            train_size (int): Size of training images. Defaults to ``224``.
            interpolation (str): Interpolation method for the Resize Crops.
                Defaults to ``"bilinear"``.
            basic_augment (bool): Whether to apply base augmentations. Defaults to
                ``True``.
            rand_augment_opt (str): Which RandAugment to use. Defaults to ``None``.
            num_workers (int): Number of workers to use for data loading. Defaults
                to ``1``.
            pin_memory (bool): Whether to pin memory. Defaults to ``True``.
            persistent_workers (bool): Whether to use persistent workers. Defaults
                to ``True``.
        """
        super().__init__(
            root=Path(root),
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            val_split=val_split,
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
        self.test_alt = test_alt
        self.interpolation = interpolation_modes_from_str(interpolation)

        if self.test_alt is not None and eval_ood:
            raise ValueError("For now test_alt argument is not supported when ood_eval=True.")

        if test_alt is None:
            self.dataset = None
        elif test_alt == "r":
            self.dataset = ImageNetR
        elif test_alt == "o":
            self.dataset = ImageNetO
        elif test_alt == "a":
            self.dataset = ImageNetA
        else:
            raise ValueError(f"The alternative {test_alt} is not known.")

        self.near_ood_datasets = near_ood_datasets or []
        self.far_ood_datasets = far_ood_datasets or []

        self.shift_dataset = ImageNetC

        self.procedure = procedure

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
                main_transform = rand_augment_transform(rand_augment_opt, {})
            else:
                main_transform = nn.Identity()
        elif self.procedure == "ViT":
            train_size = 224
            main_transform = v2.Compose(
                [
                    Mixup(mixup_alpha=0.2, cutmix_alpha=1.0),
                    rand_augment_transform("rand-m9-n2-mstd0.5", {}),
                ]
            )
        elif self.procedure == "A3":
            train_size = 160
            main_transform = rand_augment_transform("rand-m6-mstd0.5-inc1", {})
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
            self.test = self.dataset(
                self.root,
                split="val",
                download=True,
            )
        if self.eval_shift:
            self.shift_dataset(
                self.root,
                download=True,
                transform=self.test_transform,
                shift_severity=self.shift_severity,
            )

    def setup(self, stage: Literal["fit", "test"] | None = None) -> None:
        if stage not in (None, "fit", "test"):
            raise ValueError(f"Stage {stage} is not supported.")
        splits_base = download_and_extract_splits_from_hf(root=self.root)
        if stage == "fit":
            if self.test_alt is not None:
                raise ValueError("The test_alt argument is not supported for training.")

            # To change for more flexible splits later
            self.data_dir = download_and_extract_hf_dataset("imagenet1k", self.root)
            imagenet1k_splits = splits_base / "imagenet200"
            val_txt = imagenet1k_splits / "val_imagenet200.txt"
            self.val = FileListDataset(
                root=self.data_dir,
                list_file=val_txt,
                transform=self.test_transform,
            )
            self.train = None

        if stage == "test":
            if self.test_alt is not None:
                self.test = self.dataset(
                    self.root,
                    split="val",
                    transform=self.test_transform,
                    download=False,
                )
            else:
                self.data_dir = getattr(
                    self, "data_dir", download_and_extract_hf_dataset("imagenet1k", self.root)
                )
                imagenet1k_splits = splits_base / "imagenet200"
                test_txt = imagenet1k_splits / "test_imagenet200.txt"
                self.test = FileListDataset(
                    root=self.data_dir,
                    list_file=test_txt,
                    transform=self.test_transform,
                )

            if self.eval_ood:
                self.test_ood, self.val_ood, near_default, far_default = get_ood_datasets(
                    root=self.root,
                    dataset_id="imagenet200",
                    transform=self.test_transform,
                )

                if self.near_ood_datasets:
                    if not all(isinstance(ds, Dataset) for ds in self.near_ood_datasets):
                        raise TypeError("All entries in near_ood_datasets must be Dataset objects")
                    self.near_oods = self.near_ood_datasets
                else:
                    self.near_oods = list(near_default.values())

                if self.far_ood_datasets:
                    if not all(isinstance(ds, Dataset) for ds in self.far_ood_datasets):
                        raise TypeError("All entries in far_ood_datasets must be Dataset objects")
                    self.far_oods = self.far_ood_datasets
                else:
                    self.far_oods = list(far_default.values())

                for ds in [self.val_ood, *self.near_oods, *self.far_oods]:
                    if not hasattr(ds, "dataset_name"):
                        ds.dataset_name = ds.__class__.__name__.lower()

                self.near_ood_names = [ds.dataset_name for ds in self.near_oods]
                self.far_ood_names = [ds.dataset_name for ds in self.far_oods]

        if stage not in ["fit", "test", None]:
            raise ValueError(f"Stage {stage} is not supported.")

        if self.eval_shift:
            self.shift = self.shift_dataset(
                self.root,
                download=False,
                transform=self.test_transform,
                shift_severity=self.shift_severity,
            )

    def train_dataloader(self) -> DataLoader:
        # look for a train/ folder under the HF extraction root
        train_dir = Path(self.data_dir) / "train"
        if train_dir.is_dir():
            ds_train = ImageFolder(train_dir, transform=self.train_transform)
            return self._data_loader(ds_train, training=True, shuffle=True)
        raise RuntimeError(
            "ImageNet training data not found under:\n"
            f"    {train_dir}\n"
            "Please download the ILSVRC2012 train split manually from\n"
            "https://www.image-net.org/download/ and unpack it under that folder."
        )

    def val_dataloader(self) -> DataLoader:
        return self._data_loader(self.val, training=False)

    def test_dataloader(self):
        loaders = [self._data_loader(self.get_test_set(), training=False)]
        if self.eval_ood:
            loaders.append(self._data_loader(self.get_test_ood_set(), training=False))

            loaders.append(self._data_loader(self.get_val_ood_set(), training=False))

            loaders.extend(self._data_loader(ds, training=False) for ds in self.get_near_ood_set())

            loaders.extend(self._data_loader(ds, training=False) for ds in self.get_far_ood_set())
        if self.eval_shift:
            loaders.append(self._data_loader(self.get_shift_set(), training=False))
        return loaders

    def get_indices(self):
        idx = 0
        indices = {}
        indices["test"] = [idx]
        idx += 1
        if self.eval_ood:
            indices["test_ood"] = [idx]
            idx += 1
            indices["val_ood"] = [idx]
            idx += 1
            n_near = len(self.near_oods)
            indices["near_oods"] = list(range(idx, idx + n_near))
            idx += n_near
            n_far = len(self.far_oods)
            indices["far_oods"] = list(range(idx, idx + n_far))
            idx += n_far
        else:
            indices["test_ood"] = []
            indices["val_ood"] = []
            indices["near_oods"] = []
            indices["far_oods"] = []
        if self.eval_shift:
            indices["shift"] = [idx]
        else:
            indices["shift"] = []
        return indices


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
