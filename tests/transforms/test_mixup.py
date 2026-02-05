import pytest
import torch

from torch_uncertainty.transforms import Mixup, MixupIO, RegMixup, WarpingMixup


@pytest.fixture
def batch_input() -> tuple[torch.Tensor, torch.Tensor]:
    imgs = torch.rand(2, 3, 28, 28)
    return imgs, torch.tensor([0, 1])


class TestMixup:
    """Testing Mixup augmentation."""

    def test_batch_mixup(self, batch_input) -> None:
        mixup = Mixup(alpha=1.0, isobatch=True, num_classes=2)
        _ = mixup(*batch_input)

    def test_elem_mixup(self, batch_input) -> None:
        mixup = Mixup(alpha=1.0, isobatch=False, num_classes=2)
        _ = mixup(*batch_input)


class TestMixupIO:
    """Testing MixupIO augmentation."""

    def test_batch_mixupio(self, batch_input) -> None:
        mixup = MixupIO(alpha=1.0, isobatch=True, num_classes=2)
        _ = mixup(*batch_input)

    def test_elem_mixupio(self, batch_input) -> None:
        mixup = MixupIO(alpha=1.0, isobatch=False, num_classes=2)
        _ = mixup(*batch_input)


class TestRegMixup:
    """Testing RegMixup augmentation."""

    def test_batch_regmixup(self, batch_input) -> None:
        mixup = RegMixup(alpha=1.0, isobatch=True, num_classes=2)
        _ = mixup(*batch_input)

    def test_elem_regmixup(self, batch_input) -> None:
        mixup = RegMixup(alpha=1.0, isobatch=False, num_classes=2)
        _ = mixup(*batch_input)


class TestWarpingMixup:
    """Testing WarpingMixup augmentation."""

    def test_batch_kernel_warpingmixup(self, batch_input) -> None:
        mixup = WarpingMixup(alpha=1.0, isobatch=True, num_classes=2, apply_kernel=True)
        _ = mixup(*batch_input, batch_input[0])

    def test_elem_kernel_warpingmixup(self, batch_input) -> None:
        mixup = WarpingMixup(alpha=1.0, isobatch=False, num_classes=2, apply_kernel=True)
        _ = mixup(*batch_input, batch_input[0])

    def test_elem_warpingmixup(self, batch_input) -> None:
        mixup = WarpingMixup(alpha=1.0, isobatch=False, num_classes=2, apply_kernel=False)
        _ = mixup(*batch_input, batch_input[0])
