import pytest
from torch import nn

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

    with pytest.raises(ValueError):
        dm.setup("other")
