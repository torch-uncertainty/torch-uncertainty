from typing import Literal

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def _check_classes(num_classes):
    """Check that the number of classes is an int and is striclty positive."""
    if not isinstance(num_classes, int):
        raise TypeError(f"num_classes must be an integer. Got {num_classes}.")
    if num_classes <= 0:
        raise ValueError(f"The number of classes must be strictly positive. Got {num_classes}.")


def _extract_data(
    dataloader: DataLoader,
    model: nn.Module,
    device: Literal["cpu", "cuda"] | torch.device,
    progress: bool,
) -> tuple[Tensor, Tensor]:
    """Extract logits and labels from the dataloader.

    Args:
        dataloader (DataLoader): The calibration dataloader.
        model (nn.Module): Model to calibrate.
        device (Optional[Literal["cpu", "cuda"]], optional): Device to use for
            tensor operations.
        progress (bool): Whether to show the progress bar.

    Returns:
        tuple[Tensor, Tensor]: Tensors containing all logits and labels
            from the dataset.
    """
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, disable=not progress):
            logits = model(inputs.to(device))
            all_logits.append(logits)
            all_labels.append(labels)

    all_logits = torch.cat(all_logits).to(device)
    all_labels = torch.cat(all_labels).to(device)

    if all_labels.ndim == 1:
        all_labels = all_labels.long()

    return all_logits, all_labels
