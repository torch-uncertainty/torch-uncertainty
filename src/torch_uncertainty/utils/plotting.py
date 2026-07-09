import numpy as np
import torch
import torchvision.transforms.functional as F
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor


def show_segmentation_predictions(prediction: Tensor, target: Tensor) -> Figure:
    imgs = [prediction, target]
    fig = Figure(figsize=(12, 6), dpi=300)
    axs = fig.subplots(ncols=2)
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
    top_k: int | None = 100,
    dpi: int = 60,
) -> tuple[Figure, Axes]:
    """Plot per-class accuracy as a grid of colored squares (worst classes first).

    Classes are sorted by ascending accuracy and laid out in a near-square grid.
    Each cell is colored red (low) to green (high) via the RdYlGn colormap.
    When ``top_k`` is set and fewer than ``top_k`` classes exist, all classes
    are shown. Otherwise the ``top_k`` lowest-accuracy classes are displayed.

    Args:
        per_class_acc (Tensor): Per-class accuracy tensor of shape ``(num_classes,)``.
        class_names (list[str]): Names of the classes. If ``None``, uses class indices.
            Defaults to ``None``.
        top_k (int): Maximum number of classes to display, chosen as the lowest-accuracy
            classes. If ``None``, all classes are shown. Defaults to ``100``.
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

    # Always sort ascending so the worst class is always at top-left
    if top_k is not None and num_classes > top_k:
        indices = np.argsort(acc_values)[:top_k]
        title = f"Per-Class Accuracy (worst {top_k}/{num_classes}, mean: {mean_acc:.3f})"
    else:
        indices = np.argsort(acc_values)
        title = f"Per-Class Accuracy (mean: {mean_acc:.3f})"
    acc_values = acc_values[indices]
    class_names = [class_names[i] for i in indices]

    n = len(acc_values)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))

    raw = np.full(nrows * ncols, np.nan)
    raw[:n] = acc_values
    grid = np.ma.masked_invalid(raw.reshape(nrows, ncols))

    # Cell size in inches, capped so the figure stays near 10x10
    cell_in = min(10.0 / max(ncols, nrows), 0.8)
    figw = ncols * cell_in + 1.2
    figh = nrows * cell_in + 0.5

    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#cccccc")

    fig, ax = plt.subplots(figsize=(figw, figh), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#cccccc")

    mesh = ax.pcolormesh(grid, cmap=cmap, vmin=0, vmax=1, edgecolors="white", linewidth=1.5)
    # set_ylim(nrows, 0): y decreases downward, so row 0 is at the top
    ax.set_xlim(0, ncols)
    ax.set_ylim(nrows, 0)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label("Accuracy", fontsize=8)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.tick_params(labelsize=7)

    cell_px = cell_in * dpi
    label_fontsize = int(cell_px // 6)
    if label_fontsize >= 6:
        for idx in range(n):
            r, c = divmod(idx, ncols)
            v = float(acc_values[idx])
            rgba = cmap(v)
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(
                c + 0.5,
                r + 0.5,
                class_names[idx],
                ha="center",
                va="center",
                fontsize=label_fontsize,
                fontweight="bold",
                color="white" if lum < 0.45 else "black",
            )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
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
    fig = Figure(figsize=(7, 5), dpi=dpi)
    ax = fig.add_subplot()
    ax.set_axisbelow(True)
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
    ax.grid(True, linestyle="--", alpha=0.7, zorder=0)
    ax.legend()
    fig.tight_layout()
    return fig, ax
