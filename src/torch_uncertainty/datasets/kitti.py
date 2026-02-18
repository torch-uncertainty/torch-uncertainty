import json
import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from PIL import Image
from torchvision import tv_tensors
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import (
    download_and_extract_archive,
    download_url,
)
from torchvision.transforms import functional as F
from tqdm import tqdm


class KITTIDepth(VisionDataset):
    root: Path
    depth_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip"
    depth_md5 = "7d1ce32633dc2f43d9d1656a1f875e47"
    raw_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/"
    raw_filenames_url = "https://raw.githubusercontent.com/torch-uncertainty/dataset-metadata/main/download/kitti/raw_filenames.json"
    raw_filenames_md5 = "e5b7fad5ecd059488ef6c02dc9e444c1"
    _num_samples = {
        "train": 42949,
        "val": 3426,
        "test": ...,
    }

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val"],
        min_depth: float = 0.0,
        max_depth: float = 80.0,
        transforms: Callable | None = None,
        download: bool = False,
        remove_unused: bool = False,
    ) -> None:
        """KITTI Depth Estimation dataset.

        PyTorch ``VisionDataset`` wrapper for depth estimation
        benchmark derived from the KITTI raw dataset. This dataset provides RGB
        images from the left color camera (``leftImg8bit``) paired with projected
        LiDAR depth maps (``leftDepth``).

        The dataset is automatically structured as:

            root/
                KITTIDepth/
                    train/
                        leftImg8bit/*.png
                        leftDepth/*.png
                    val/
                        leftImg8bit/*.png
                        leftDepth/*.png

        Args:
            root (str or Path): Root directory where the dataset will be stored.
            split (Literal["train", "val"]): Dataset split to use.
            min_depth (float, default=0.0): Minimum valid depth value (in meters). Depth values
                smaller than or equal to this threshold are set to NaN.
            max_depth (float, default=80.0): Maximum valid depth value (in meters). Depth values
                greater than this threshold are set to NaN.
            transforms (Callable, optional): A function/transform that takes an ``(image, target)``
                pair and returns the transformed pair.
            download (bool, default=False): If True, downloads and restructures the depth
                annotations and raw KITTI data if not already present.
            remove_unused (bool, default=False): If True, removes the extracted raw files after
                restructuring to save disk space.

        Returns:
            tuple[tv_tensors.Image, tv_tensors.Mask]:
                - image: RGB image as ``tv_tensors.Image``.
                - target: Depth map as ``tv_tensors.Mask`` in meters (float),
                with invalid values set to NaN.

        Notes:
            - Depth maps are stored as 16-bit PNG files and converted to meters
            by dividing by 256.0.
            - Usage of this dataset is subject to the original KITTI license
            (CC BY-NC-SA 3.0).
        """
        logging.info(
            "KITTIDepth is copyrighted by the Karlsruhe Institute of Technology "
            "(KIT) and the Toyota Technological Institute at Chicago (TTIC). "
            "By using KITTIDepth, you agree to the terms and conditions of the "
            "CC BY-NC-SA 3.0 License. "
            "This means that you must attribute the work in the manner specified "
            "by the authors, you may not use this work for commercial purposes "
            "and if you alter, transform, or build upon this work, you may "
            "distribute the resulting work only under the same license."
        )

        super().__init__(
            root=Path(root) / "KITTIDepth",
            transforms=transforms,
        )
        self.min_depth = min_depth
        self.max_depth = max_depth

        if split not in ["train", "val"]:
            raise ValueError(f"split must be one of ['train', 'val']. Got {split}.")

        self.split = split

        if not self.check_split_integrity("leftDepth"):
            if download:
                self._download_depth()
            else:
                raise FileNotFoundError(
                    f"KITTI {split} split not found or incomplete. Set download=True to download it."
                )

        if not self.check_split_integrity("leftImg8bit"):
            if download:
                self._download_raw(remove_unused)
            else:
                raise FileNotFoundError(
                    f"KITTI {split} split not found or incomplete. Set download=True to download it."
                )

        self._make_dataset()

    def check_split_integrity(self, folder: str) -> bool:
        split_path = self.root / self.split
        return (
            split_path.is_dir()
            and len(list((split_path / folder).glob("*.png"))) == self._num_samples[self.split]
        )

    def __getitem__(self, index: int) -> tuple[tv_tensors.Image, tv_tensors.Mask]:
        """Get the sample at the given index.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a depth map.
        """
        image = tv_tensors.Image(Image.open(self.samples[index]).convert("RGB"))
        target = tv_tensors.Mask(
            F.pil_to_tensor(Image.open(self.targets[index])).squeeze(0) / 256.0
        )
        target[(target <= self.min_depth) | (target > self.max_depth)] = float("nan")

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        """The number of samples in the dataset."""
        return self._num_samples[self.split]

    def _make_dataset(self) -> None:
        self.samples = sorted((self.root / self.split / "leftImg8bit").glob("*.png"))
        self.targets = sorted((self.root / self.split / "leftDepth").glob("*.png"))

    def _download_depth(self) -> None:
        """Download and extract the depth annotation dataset."""
        if not (self.root / "tmp").exists():
            download_and_extract_archive(
                self.depth_url,
                download_root=self.root,
                extract_root=self.root / "tmp",
                md5=self.depth_md5,
            )

        logging.info("Re-structuring the depth annotations...")

        if (self.root / "train" / "leftDepth").exists():
            shutil.rmtree(self.root / "train" / "leftDepth")

        (self.root / "train" / "leftDepth").mkdir(parents=True, exist_ok=False)

        depth_files = list((self.root).glob("**/tmp/train/**/image_02/*.png"))
        logging.info("Train files...")
        for file in tqdm(depth_files):
            exp_code = file.parents[3].name.split("_")
            filecode = "_".join([exp_code[0], exp_code[1], exp_code[2], exp_code[4], file.name])
            shutil.copy(file, self.root / "train" / "leftDepth" / filecode)

        if (self.root / "val" / "leftDepth").exists():
            shutil.rmtree(self.root / "val" / "leftDepth")

        (self.root / "val" / "leftDepth").mkdir(parents=True, exist_ok=False)

        depth_files = list((self.root).glob("**/tmp/val/**/image_02/*.png"))
        logging.info("Validation files...")
        for file in tqdm(depth_files):
            exp_code = file.parents[3].name.split("_")
            filecode = "_".join([exp_code[0], exp_code[1], exp_code[2], exp_code[4], file.name])
            shutil.copy(file, self.root / "val" / "leftDepth" / filecode)

        shutil.rmtree(self.root / "tmp")

    def _download_raw(self, remove_unused: bool) -> None:
        """Download and extract the raw dataset."""
        download_url(
            self.raw_filenames_url,
            self.root,
            "raw_filenames.json",
            self.raw_filenames_md5,
        )
        with (self.root / "raw_filenames.json").open() as file:
            raw_filenames = json.load(file)

        for filename in tqdm(raw_filenames):
            logging.info("%s", self.raw_url + filename)
            download_and_extract_archive(
                self.raw_url + filename,
                download_root=self.root,
                extract_root=self.root / "raw",
                md5=None,
            )

        logging.info("Re-structuring the raw data...")

        samples_to_keep = list((self.root / "train" / "leftDepth").glob("*.png"))

        if (self.root / "train" / "leftImg8bit").exists():
            shutil.rmtree(self.root / "train" / "leftImg8bit")

        (self.root / "train" / "leftImg8bit").mkdir(parents=True, exist_ok=False)

        logging.info("Train files...")
        for sample in tqdm(samples_to_keep):
            filecode = sample.name.split("_")
            first_level = "_".join([filecode[0], filecode[1], filecode[2]])
            second_level = "_".join(
                [
                    filecode[0],
                    filecode[1],
                    filecode[2],
                    "drive",
                    filecode[3],
                    "sync",
                ]
            )
            raw_path = (
                self.root / "raw" / first_level / second_level / "image_02" / "data" / filecode[4]
            )
            shutil.copy(raw_path, self.root / "train" / "leftImg8bit" / sample.name)

        samples_to_keep = list((self.root / "val" / "leftDepth").glob("*.png"))

        if (self.root / "val" / "leftImg8bit").exists():
            shutil.rmtree(self.root / "val" / "leftImg8bit")

        (self.root / "val" / "leftImg8bit").mkdir(parents=True, exist_ok=False)

        logging.info("Validation files...")
        for sample in tqdm(samples_to_keep):
            filecode = sample.name.split("_")
            first_level = "_".join([filecode[0], filecode[1], filecode[2]])
            second_level = "_".join(
                [
                    filecode[0],
                    filecode[1],
                    filecode[2],
                    "drive",
                    filecode[3],
                    "sync",
                ]
            )
            raw_path = (
                self.root / "raw" / first_level / second_level / "image_02" / "data" / filecode[4]
            )
            shutil.copy(raw_path, self.root / "val" / "leftImg8bit" / sample.name)

        if remove_unused:
            shutil.rmtree(self.root / "raw")
