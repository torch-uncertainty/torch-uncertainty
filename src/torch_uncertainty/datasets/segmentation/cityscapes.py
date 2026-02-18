from collections.abc import Callable
from typing import Any

import torch
from einops import rearrange
from PIL import Image
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from torchvision import tv_tensors
from torchvision.datasets import Cityscapes as TVCityscapes
from torchvision.transforms.v2 import functional as F


class Cityscapes(TVCityscapes):
    color_palette = [
        (128, 64, 128),  # 0: road
        (244, 35, 232),  # 1: sidewalk
        (70, 70, 70),  # 2: building
        (102, 102, 156),  # 3: wall
        (190, 153, 153),  # 4: fence
        (153, 153, 153),  # 5: pole
        (250, 170, 30),  # 6: traffic light
        (220, 220, 0),  # 7: traffic sign
        (107, 142, 35),  # 8: vegetation
        (152, 251, 152),  # 9: terrain
        (70, 130, 180),  # 10: sky
        (220, 20, 60),  # 11: person
        (255, 0, 0),  # 12: rider
        (0, 0, 142),  # 13: car
        (0, 0, 70),  # 14: truck
        (0, 60, 100),  # 15: bus
        (0, 80, 100),  # 16: train
        (0, 0, 230),  # 17: motorcycle
        (119, 11, 32),  # 18: bicycle
        (0, 0, 0),  # 19: void
    ]

    def __init__(
        self,
        root: str,
        split: str = "train",
        mode: str = "fine",
        target_type: list[str] | str = "instance",
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        transforms: Callable[..., Any] | None = None,
    ) -> None:
        """Cityscapes dataset wrapper with train ID color mapping.

        This class extends the `TVCityscapes` dataset to provide a fixed
        color palette for visualization and encode/decode utilities for
        semantic segmentation targets using Cityscapes train IDs. It also
        sets up mapping from train IDs to RGB colors for easier interpretation
        of predicted masks.

        Attributes:
            color_palette (list[tuple[int, int, int]]):
                List of RGB tuples for each class label in the Cityscapes dataset.
            train_id_to_color (torch.Tensor):
                Tensor mapping train IDs to RGB colors for output decoding.

        Args:
            root (str): Root directory of the Cityscapes dataset.
            split (str, optional): Dataset split to use, such as "train", "val", or "test".
            mode (str, optional): Annotation mode, e.g., "fine" or "coarse".
            target_type (list[str] | str, optional):
                One or more target types to load ("instance", "semantic", etc.).
            transform (Callable[..., Any] | None, optional):
                Transformation applied to the input image.
            target_transform (Callable[..., Any] | None, optional):
                Transformation applied to the target.
            transforms (Callable[..., Any] | None, optional):
                Combined transformation for image and target.

        """
        super().__init__(
            root,
            split,
            mode,
            target_type,
            transform,
            target_transform,
            transforms,
        )
        train_id_to_color = [
            c.color for c in self.classes if (c.train_id != -1 and c.train_id != 255)
        ]
        train_id_to_color.append([0, 0, 0])
        self.train_id_to_color = torch.tensor(train_id_to_color)

    @classmethod
    def encode_target(cls, target: Image.Image) -> Image.Image:
        """Encode target image to tensor.

        Args:
            target (Image.Image): Target PIL image.

        Returns:
            torch.Tensor: Encoded target.
        """
        colored_target = F.pil_to_tensor(target)
        colored_target = rearrange(colored_target, "c h w -> h w c")
        target = torch.zeros_like(colored_target[..., :1])
        # convert target color to index
        for cityscapes_class in cls.classes:
            target[
                (colored_target == torch.tensor(cityscapes_class.id, dtype=target.dtype)).all(
                    dim=-1
                )
            ] = cityscapes_class.train_id

        return F.to_pil_image(rearrange(target, "h w c -> c h w"))

    def decode_target(self, target: torch.Tensor) -> torch.Tensor:
        """Decode target tensor to RGB tensor.

        Args:
            target (torch.Tensor): Target RGB tensor.

        Returns:
            Image.Image: Decoded target.
        """
        target[target == 255] = -1
        return self.train_id_to_color[target]

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """Get the sample at the given index.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a tuple of all target types if ``target_type``
                is a list with more than one item. Otherwise, target is a json object if
                ``target_type="polygon"``, else the image segmentation.
        """
        image = tv_tensors.Image(Image.open(self.images[index]).convert("RGB"))

        targets: Any = []
        for i, t in enumerate(self.target_type):
            if t == "polygon":
                target = self._load_json(self.targets[index][i])
            elif t == "semantic":
                target = tv_tensors.Mask(self.encode_target(Image.open(self.targets[index][i])))
            else:
                target = Image.open(self.targets[index][i])

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def plot_sample(self, index: int, ax: _AX_TYPE | None = None) -> _PLOT_OUT_TYPE:
        """Plot a sample from the dataset.

        Args:
            index: The index of the sample to plot.
            ax: Optional matplotlib axis to plot on.

        Returns:
            The axis on which the sample was plotted.
        """
        raise NotImplementedError("This method is not implemented yet.")
