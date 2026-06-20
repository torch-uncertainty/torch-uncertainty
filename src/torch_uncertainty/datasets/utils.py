import copy
import gzip
import io
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from torch.utils.data import Dataset, random_split


def load_arff(path: Path) -> pd.DataFrame:
    """Parse an ARFF file into a pandas DataFrame.

    Handles both plain text and gzip-compressed ARFF files.
    """
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            content = f.read()
    except (gzip.BadGzipFile, OSError):
        with path.open(encoding="utf-8") as f:
            content = f.read()

    col_names = []
    data_start = 0
    lines = content.splitlines()

    for i, line in enumerate(lines):
        stripped = line.strip()
        lower = stripped.lower()
        if not stripped or stripped.startswith("%"):
            continue
        if lower.startswith("@relation"):
            continue
        if lower.startswith("@attribute"):
            parts = stripped.split(None, 2)
            col_names.append(parts[1].strip("'\""))
        elif lower.startswith("@data"):
            data_start = i + 1
            break

    if not col_names or data_start == 0:
        raise ValueError(
            f"Could not parse ARFF file '{path}': no @attribute or @data section found. "
            "The file may be corrupt or not a valid ARFF file. "
            "Delete the cached file and re-download."
        )

    data_content = "\n".join(lines[data_start:])
    return pd.read_csv(
        io.StringIO(data_content),
        header=None,
        names=col_names,
        na_values=["?", ""],
        skipinitialspace=True,
        quotechar="'",
    )


def create_train_val_split(
    dataset: Dataset,
    val_split_rate: float,
    val_transforms: Callable | None = None,
) -> tuple[Dataset, Dataset]:
    """Split a dataset for training and validation.

    Args:
        dataset: The dataset to be split.
        val_split_rate: The amount of the original dataset to use as validation split.
        val_transforms: The transformations to apply on the validation set.
            Defaults to ``None``.

    Returns:
        tuple[Dataset, Dataset]: The training and the validation splits.
    """
    train, val = random_split(dataset, [1 - val_split_rate, val_split_rate])
    val = copy.deepcopy(val)  # Ensure train.dataset.transform is not modified next line
    val.dataset.transform = val_transforms
    return train, val


class TTADataset(Dataset):
    def __init__(self, dataset: Dataset, num_augmentations: int) -> None:
        """Create a version of the dataset that returns the same sample multiple times.

        This is useful for test-time augmentation (TTA).

        Args:
            dataset: The dataset to be adapted for TTA.
            num_augmentations: The number of augmentations to apply.
        """
        super().__init__()
        self.dataset = dataset
        self.num_augmentations = num_augmentations

    def __len__(self) -> int:
        """Get the virtual length of the dataset."""
        return len(self.dataset) * self.num_augmentations

    def __getitem__(self, index) -> Any:
        """Get the item corresponding to idx // :attr:`self.num_augmentations`."""
        return self.dataset[index // self.num_augmentations]
