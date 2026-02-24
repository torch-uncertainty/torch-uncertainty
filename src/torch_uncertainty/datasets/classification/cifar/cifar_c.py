import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
from torch import Tensor
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
)


class CIFAR10C(VisionDataset):
    base_folder = "CIFAR-10-C"
    tgz_md5 = "56bf5dcef84df0e2308c6dcbcbbd8499"
    cifarc_subsets = [
        "brightness",
        "contrast",
        "defocus_blur",
        "elastic_transform",
        "fog",
        "frost",
        "gaussian_blur",
        "gaussian_noise",
        "glass_blur",
        "impulse_noise",
        "jpeg_compression",
        "motion_blur",
        "pixelate",
        "saturate",
        "shot_noise",
        "snow",
        "spatter",
        "speckle_noise",
        "zoom_blur",
    ]

    ctest_list = [
        ["fog.npy", "7b397314b5670f825465fbcd1f6e9ccd"],
        ["jpeg_compression.npy", "2b9cc4c864e0193bb64db8d7728f8187"],
        ["zoom_blur.npy", "6ea8e63f1c5cdee1517533840641641b"],
        ["speckle_noise.npy", "ef00b87611792b00df09c0b0237a1e30"],
        ["glass_blur.npy", "7361fb4019269e02dbf6925f083e8629"],
        ["spatter.npy", "8a5a3903a7f8f65b59501a6093b4311e"],
        ["shot_noise.npy", "3a7239bb118894f013d9bf1984be7f11"],
        ["defocus_blur.npy", "7d1322666342a0702b1957e92f6254bc"],
        ["elastic_transform.npy", "9421657c6cd452429cf6ce96cc412b5f"],
        ["gaussian_blur.npy", "c33370155bc9b055fb4a89113d3c559d"],
        ["frost.npy", "31f6ab3bce1d9934abfb0cc13656f141"],
        ["saturate.npy", "1cfae0964219c5102abbb883e538cc56"],
        ["brightness.npy", "0a81ef75e0b523c3383219c330a85d48"],
        ["snow.npy", "bb238de8555123da9c282dea23bd6e55"],
        ["gaussian_noise.npy", "ecaf8b9a2399ffeda7680934c33405fd"],
        ["motion_blur.npy", "fffa5f852ff7ad299cfe8a7643f090f4"],
        ["contrast.npy", "3c8262171c51307f916c30a3308235a8"],
        ["impulse_noise.npy", "2090e01c83519ec51427e65116af6b1a"],
        ["labels.npy", "c439b113295ed5254878798ffe28fd54"],
        ["pixelate.npy", "0f14f7e2db14288304e1de10df16832f"],
    ]
    url = "https://huggingface.co/datasets/torch-uncertainty/CIFAR-C/resolve/main/CIFAR-10-C.tar"
    filename = "CIFAR-10-C.tar"

    def __init__(
        self,
        root: Path | str,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        subset: str = "all",
        shift_severity: int = 1,
        download: bool = False,
    ) -> None:
        """The corrupted CIFAR-10-C Dataset.

        Args:
            root (str): Root directory of the datasets.
            transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``. Defaults to ``None``.
            target_transform (callable, optional): A function/transform that takes in the target and transforms it. Defaults to ``None``.
            subset (str): The subset to use, one of ``all`` or the keys in ``cifarc_subsets``.
            shift_severity (int): The shift_severity of the corruption, between ``1`` and ``5``.
            download (bool, optional): If ``True``, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again. Defaults to ``False``.

        References:
            [1] `Benchmarking neural network robustness to common corruptions and perturbations. Dan Hendrycks and Thomas Dietterich. In ICLR, 2019 <https://arxiv.org/abs/1903.12261>`_.
        """
        self.root = Path(root)
        # Download the new targets
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found. You can use download=True to download it.")

        super().__init__(
            root=self.root / self.base_folder,
            transform=transform,
            target_transform=target_transform,
        )
        if subset not in ["all", *self.cifarc_subsets]:
            raise ValueError(f"The subset '{subset}' does not exist in CIFAR-C.")
        self.subset = subset
        self.shift_severity = shift_severity

        if shift_severity not in list(range(1, 6)):
            raise ValueError(
                "Corruptions shift_severity should be chosen between 1 and 5 included."
            )
        samples, labels = self.make_dataset(self.root, self.subset, self.shift_severity)

        self.samples = samples
        self.labels = labels.astype(np.int64)

    def make_dataset(
        self, root: Path, subset: str, shift_severity: int
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Make the CIFAR-C dataset.

        Build the corrupted dataset according to the chosen subset and
        shift_severity. If the subset is 'all', gather all corruption types
        in the dataset.

        Args:
            root (Path): The path to the dataset.
            subset (str): The name of the corruption subset to be used. Choose
                `all` for the dataset to contain all subsets.
            shift_severity (int): The shift_severity of the corruption applied to the
                images.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The samples and labels of the chosen.
        """
        if subset == "all":
            labels: np.ndarray = np.load(root / "labels.npy")[
                (shift_severity - 1) * 10000 : shift_severity * 10000
            ]
            sample_arrays = [
                np.load(root / (cifar_subset + ".npy"))[
                    (shift_severity - 1) * 10000 : shift_severity * 10000
                ]
                for cifar_subset in self.cifarc_subsets
            ]
            samples = np.concatenate(sample_arrays, axis=0)
            labels = np.tile(labels, len(self.cifarc_subsets))

        else:
            samples: np.ndarray = np.load(root / (subset + ".npy"))[
                (shift_severity - 1) * 10000 : shift_severity * 10000
            ]
            labels: np.ndarray = np.load(root / "labels.npy")[
                (shift_severity - 1) * 10000 : shift_severity * 10000
            ]
        return samples, labels

    def __len__(self) -> int:
        """The number of samples in the dataset."""
        return self.labels.shape[0]

    def __getitem__(self, index: int) -> tuple[np.ndarray | Tensor, int]:
        """Get the samples and targets of the dataset.

        Args:
            index (int): The index of the sample to get.
        """
        sample, target = (
            self.samples[index],
            self.labels[index],
        )

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def _check_integrity(self) -> bool:
        """Check the integrity of the dataset."""
        for filename, md5 in self.ctest_list:
            fpath = self.root / self.base_folder / filename
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        """Download the dataset."""
        if self._check_integrity():
            logging.info("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)


class CIFAR100C(CIFAR10C):
    base_folder = "CIFAR-100-C"
    tgz_md5 = "11f0ed0f1191edbf9fa23466ae6021d3"
    ctest_list = [
        ["fog.npy", "4efc7ebd5e82b028bdbe13048e3ea564"],
        ["jpeg_compression.npy", "c851b7f1324e1d2ffddeb76920576d11"],
        ["zoom_blur.npy", "0204613400c034a81c4830d5df81cb82"],
        ["speckle_noise.npy", "e3f215b1a0f9fd9fd6f0d1cf94a7ce99"],
        ["glass_blur.npy", "0bf384f38e5ccbf8dd479d9059b913e1"],
        ["spatter.npy", "12ccf41d62564d36e1f6a6ada5022728"],
        ["shot_noise.npy", "b0a1fa6e1e465a747c1b204b1914048a"],
        ["defocus_blur.npy", "d923e3d9c585a27f0956e2f2ad832564"],
        ["elastic_transform.npy", "a0792bd6581f6810878be71acedfc65a"],
        ["gaussian_blur.npy", "5204ba0d557839772ef5a4196a052c3e"],
        ["frost.npy", "3a39c6823bdfaa0bf8b12fe7004b8117"],
        ["saturate.npy", "c0697e9fdd646916a61e9c312c77bf6b"],
        ["brightness.npy", "f22d7195aecd6abb541e27fca230c171"],
        ["snow.npy", "0237be164583af146b7b144e73b43465"],
        ["gaussian_noise.npy", "ecc4d366eac432bdf25c024086f5e97d"],
        ["motion_blur.npy", "732a7e2e54152ff97c742d4c388c5516"],
        ["contrast.npy", "322bb385f1d05154ee197ca16535f71e"],
        ["impulse_noise.npy", "3b3c210ddfa0b5cb918ff4537a429fef"],
        ["labels.npy", "bb4026e9ce52996b95f439544568cdb2"],
        ["pixelate.npy", "96c00c60f144539e14cffb02ddbd0640"],
    ]
    url = "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar"
    filename = "CIFAR-100-C.tar"

    def __init__(
        self,
        root: Path | str,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        subset: str = "all",
        shift_severity: int = 1,
        download: bool = False,
    ) -> None:
        """The corrupted CIFAR-100-C Dataset.

        Args:
            root (str): Root directory of the datasets.
            transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``. Defaults to None.
            target_transform (callable, optional): A function/transform that takes in the target and transforms it. Defaults to None.
            subset (str): The subset to use, one of ``all`` or the keys in ``cifarc_subsets``.
            shift_severity (int): The shift_severity of the corruption, between 1 and 5.
            download (bool, optional): If True, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again. Defaults to False.
            kwargs: Additional keyword arguments passed to the parent class (CIFAR-10-C).

        .. seealso::
            - :class:`CIFAR10C` - The corrupted CIFAR-10-C dataset.

        References:
            [1] `Benchmarking neural network robustness to common corruptions and perturbations. Dan Hendrycks and Thomas Dietterich. In ICLR, 2019 <https://arxiv.org/abs/1903.12261>`_.
        """
        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            subset=subset,
            shift_severity=shift_severity,
            download=download,
        )
