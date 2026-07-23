from pathlib import Path

import pytest
from torch import nn
from torch.utils.data import Subset
from torchvision.transforms import v2

from tests._dummies.dataset import DummyClassificationDataset
from torch_uncertainty.datamodules.classification.imagenet200 import (
    ImageNet200DataModule,
)
from torch_uncertainty.datasets.classification import ImageNet200


def test_imagenet200() -> None:
    dm = ImageNet200DataModule(
        root="./data/",
        batch_size=128,
        train_transform=nn.Identity(),
        test_transform=nn.Identity(),
    )

    assert dm.dataset == ImageNet200
    assert isinstance(dm.train_transform, nn.Identity)
    assert isinstance(dm.test_transform, nn.Identity)

    dm.dataset = DummyClassificationDataset
    dm.prepare_data()
    dm.setup()

    dm.train_dataloader()
    dm.val_dataloader()
    dm.test_dataloader()
    assert len(dm._get_train_data()) == len(dm.train)
    assert len(dm._get_train_targets()) == len(dm.train)

    with pytest.raises(ValueError):
        dm.setup("other")


def test_imagenet200_default_transforms() -> None:
    dm = ImageNet200DataModule(root="./data/", batch_size=128)
    assert isinstance(dm.train_transform, v2.Compose)
    assert isinstance(dm.test_transform, v2.Compose)

    dm = ImageNet200DataModule(
        root="./data/",
        batch_size=128,
        basic_augment=False,
        rand_augment_opt="rand-m9-n1",
        num_tta=2,
    )
    assert dm.test_transform is dm.train_transform


def test_imagenet200_float_val_split() -> None:
    dm = ImageNet200DataModule(
        root="./data/",
        batch_size=2,
        val_split=0.5,
        train_transform=nn.Identity(),
        test_transform=nn.Identity(),
    )
    dm.dataset = DummyClassificationDataset
    dm.setup("fit")

    assert isinstance(dm.train, Subset)
    assert isinstance(dm.val, Subset)
    assert len(dm._get_train_data()) == len(dm.train)
    assert len(dm._get_train_targets()) == len(dm.train)


def test_imagenet200_indices_val_split() -> None:
    path = Path(__file__).parent.resolve() / "../../assets/dummy_indices.yaml"
    dm = ImageNet200DataModule(
        root="./data/",
        batch_size=2,
        val_split=path,
        train_transform=nn.Identity(),
        test_transform=nn.Identity(),
    )
    dm.dataset = lambda root, split, transform: DummyClassificationDataset(
        root,
        split=split,
        transform=transform,
        num_images=10,
    )
    dm.setup("fit")
    dm.setup("test")

    assert isinstance(dm.train, Subset)
    assert isinstance(dm.val, Subset)
    assert len(dm.train) == 6
    assert len(dm.val) == 2
    assert len(dm._get_train_data()) == len(dm.train)
    assert len(dm._get_train_targets()) == len(dm.train)
    dm.train_dataloader()
    dm.val_dataloader()
    dm.test_dataloader()


def test_imagenet200_verify_splits(tmp_path: Path) -> None:
    dm = ImageNet200DataModule(
        root=tmp_path,
        batch_size=2,
        train_transform=nn.Identity(),
        test_transform=nn.Identity(),
    )

    (tmp_path / "train").mkdir()
    dm._verify_splits("train")

    with pytest.raises(FileNotFoundError, match="ImageNet val split"):
        dm._verify_splits("val")

    with pytest.raises(ValueError, match="is not a file"):
        ImageNet200DataModule(
            root=tmp_path,
            batch_size=2,
            val_split=tmp_path / "missing.yaml",
        )
