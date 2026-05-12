import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import Tensor


def show_segmentation_predictions(prediction: Tensor, target: Tensor) -> Figure:
    imgs = [prediction, target]
    fig, axs = plt.subplots(ncols=2, figsize=(12, 6), dpi=300)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[i].imshow(np.asarray(img))
        axs[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    axs[0].set(title="Prediction")
    axs[1].set(title="Ground Truth")
    return fig


def plot_per_class_accuracy(
    per_class_acc: Tensor,
    class_names: list[str] | None = None,
    dpi: int = 60,
) -> tuple[Figure, Axes]:
    """Plot per-class accuracy as a horizontal bar chart.

    Args:
        per_class_acc (Tensor): Per-class accuracy tensor of shape ``(num_classes,)``.
        class_names (list[str] | None): Names of the classes. If ``None``, uses
            class indices. Defaults to ``None``.
        dpi (int): The dpi of the plot. Defaults to ``60``.

    Returns:
        Tuple[Figure, Axes]: The figure and axes of the plot.
    """
    num_classes = len(per_class_acc)
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    elif len(class_names) != num_classes:
        raise ValueError(
            "class_names must have the same length as per_class_acc: "
            f"expected {num_classes}, got {len(class_names)}."
        )
    acc_values = per_class_acc.cpu().float().numpy()
    mean_acc = float(acc_values.mean())

    fig_height = max(4, num_classes * 0.3)
    fig, ax = plt.subplots(1, figsize=(8, fig_height), dpi=dpi)
    ax.barh(range(num_classes), acc_values, color="#1f77b4", alpha=0.8)
    ax.axvline(mean_acc, color="#d45f00", linestyle="--", label=f"Mean: {mean_acc:.3f}")
    ax.set_yticks(range(num_classes))
    ax.set_yticklabels(class_names, fontsize=max(4, min(10, 120 // num_classes)))
    ax.set_xlim(0, 1)
    ax.set_xlabel("Accuracy")
    ax.set_title("Per-Class Accuracy")
    ax.legend()
    plt.grid(True, linestyle="--", alpha=0.7, axis="x", zorder=0)
    fig.tight_layout()
    return fig, ax


def plot_hist(
    conf: list[torch.Tensor],
    bins: int = 20,
    title: str = "Histogram with 'auto' bins",
    dpi: int = 60,
) -> tuple[Figure, Axes]:
    """Plot a confidence histogram.

    Args:
        conf: The confidence values.
        bins: The number of bins. Defaults to ``20``.
        title: The title of the plot. Defaults to ``"Histogram with 'auto' bins"``.
        dpi: The dpi of the plot. Defaults to ``60``.

    Returns:
        Tuple[Figure, Axes]: The figure and axes of the plot.
    """
    plt.rc("axes", axisbelow=True)
    fig, ax = plt.subplots(1, figsize=(7, 5), dpi=dpi)
    for i in [1, 0]:
        ax.hist(
            conf[i],
            bins=bins,
            density=True,
            label=["In-distribution", "Out-of-Distribution"][i],
            alpha=0.4,
            linewidth=1,
            edgecolor=["#0d559f", "#d45f00"][i],
            color=["#1f77b4", "#ff7f0e"][i],
        )

    ax.set_title(title)
    plt.grid(True, linestyle="--", alpha=0.7, zorder=0)
    plt.legend()
    fig.tight_layout()
    return fig, ax
