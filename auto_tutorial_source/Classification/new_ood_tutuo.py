"""# Simple OOD Evaluation

In this tutorial, we demonstrate out-of-distribution (OOD) evaluation with TorchUncertainty
datamodules and routines. You will learn to:

1. **Set up a CIFAR-100 datamodule** with in-distribution, near-OOD, and far-OOD splits.
2. **Run `ClassificationRoutine`** to report in-distribution accuracy and OOD metrics (AUROC, AUPR, FPR95).
3. **Plug in custom OOD datasets** for fully custom evaluation.

## Foreword on out-of-distribution detection

OOD detection measures how well a model recognizes inputs outside its training distribution.
TorchUncertainty integrates common OOD metrics in the Lightning test loop:

- **AUROC** — area under the ROC curve
- **AUPR** — area under the precision–recall curve
- **FPR95** — false positive rate at 95% true positive rate

By default, near and far OOD splits follow the
[OpenOOD](https://arxiv.org/pdf/2306.09301) benchmark. You can override them with
`near_ood_datasets` and `far_ood_datasets` on the datamodule.

## Supported datamodules and default OOD splits

| Datamodule | In-domain | Default near-OOD (hard) | Default far-OOD (easy) |
| --- | --- | --- | --- |
| `CIFAR10DataModule` | CIFAR-10 | CIFAR-100, Tiny ImageNet | MNIST, SVHN, Textures, Places365 |
| `CIFAR100DataModule` | CIFAR-100 | CIFAR-10, Tiny ImageNet | MNIST, SVHN, Textures, Places365 |
| `ImageNetDataModule` | ImageNet-1K | SSB-hard, NINCO | iNaturalist, Textures, OpenImage-O |
| `ImageNet200DataModule` | ImageNet-200 | SSB-hard, NINCO | iNaturalist, Textures, OpenImage-O |

## Supported OOD criteria

| Criterion | Reference |
| --- | --- |
| `msp` | Hendrycks & Gimpel — [ICLR Workshop 2017](https://arxiv.org/abs/1610.02136) |
| `maxlogit` | — |
| `energy` | Liu et al. — [NeurIPS 2020](https://arxiv.org/abs/2010.03759) |
| `odin` | Liang, Li & Srikant — [ICML 2018](https://arxiv.org/abs/1706.02690) |
| `entropy` | — |
| `mutual_information` | — |
| `variation_ratio` | — |
| `scale` | Hendrycks et al. — [ICML 2022](https://proceedings.mlr.press/v162/hendrycks22a/hendrycks22a.pdf) |
| `ash` | Djurisic et al. (ASH) — [ICLR 2023](https://arxiv.org/pdf/2209.09858) |
| `react` | Sun et al. — [NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/01894d6f048493d2cacde3c579c315a3-Paper.pdf) |
| `adascale_a` | Regmi et al. — [arXiv 2025](https://arxiv.org/pdf/2503.08023) |
| `vim` | Wang et al. — [CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_ViM_Out-of-Distribution_With_Virtual-Logit_Matching_CVPR_2022_paper.pdf) |
| `knn` | Sun et al. — [ICML 2022](https://arxiv.org/abs/2106.01477) |
| `gen` | Liu et al. — [CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_GEN_Pushing_the_Limits_of_Softmax-Based_Out-of-Distribution_Detection_CVPR_2023_paper.pdf) |
| `nnguide` | Park et al. — [ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Park_Nearest_Neighbor_Guidance_for_Out-of-Distribution_Detection_ICCV_2023_paper.pdf) |
| `neco` | Ammar et al. — [arXiv 2023](https://arxiv.org/abs/2310.06823) |

> **Note:** Pass any criterion as `ood_criterion` to `ClassificationRoutine`. Ensemble-only
> methods need multiple stochastic forward passes.

> **Note:** **Near-OOD** splits are semantically close to the in-domain data; **far-OOD** splits
> are more distant (e.g. other datasets or domains). Override defaults with
> `near_ood_datasets` / `far_ood_datasets`.

## 1. Loading the utilities

To evaluate OOD with TorchUncertainty, load:

- the model: ResNet-18 (CIFAR-style) trained on CIFAR-100
- `ClassificationRoutine` from `torch_uncertainty.routines`
- `CIFAR100DataModule` from `torch_uncertainty.datamodules`
"""

# %%
from pathlib import Path

# %%
# 2. Load the trained model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# In this tutorial we will be loading a pretrained model, but you can also train your own using the same classification routine and still get ood related metrics at test phase.
import torch
from huggingface_hub import hf_hub_download

from torch_uncertainty.models.classification import resnet

net = resnet(in_channels=3, arch=18, num_classes=100, style="cifar", conv_bias=False)

# load the model
path = hf_hub_download(repo_id="torch-uncertainty/resnet18_c100", filename="resnet18_c100.ckpt")
net.load_state_dict(torch.load(path))

net.cuda()
net.eval()


# %%
# 3. Defining the necessary datamodules
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In the following, we instantiate our trainer, define the root of the datasets and the logs.
# We also create the datamodule that handles the cifar100 dataset, dataloaders and transforms.
# Datamodules can also handle OOD detection by setting the eval_ood parameter to True.

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from lightning.pytorch.loggers import TensorBoardLogger

from torch_uncertainty import TUTrainer
from torch_uncertainty.datamodules import CIFAR100DataModule
from torch_uncertainty.routines import ClassificationRoutine

# Inline backend only (avoid plt.ion() — it double-renders with display()).
try:
    from IPython import get_ipython

    if get_ipython() is not None:
        get_ipython().run_line_magic("matplotlib", "inline")
except ImportError:
    pass
def plot_ood_histograms(
    routine,
    tag: str,
    out_dir: Path = Path("figures/ood_tutorial"),
    *,
    show: bool = True,
    save: bool = True,
    bins: int = 24,
    max_cols: int = 3,
):
    """Plot ID vs OOD score histograms in a single grid (requires ``log_plots=True``)."""
    if routine.id_score_storage is None or routine.ood_score_storage is None:
        print(
            "Skipping plots: score buffers are empty. Set log_plots=True on ClassificationRoutine "
            "and run trainer.test() in this session before plot_ood_histograms()."
        )
        return

    id_scores = torch.cat(routine.id_score_storage, dim=0).numpy()
    panels: list[tuple[str, np.ndarray]] = []
    for ds_name, batches in routine.ood_score_storage.items():
        if not batches:
            print(f"  No scores for '{ds_name}', skipping.")
            continue
        panels.append((ds_name, torch.cat(batches, dim=0).numpy()))

    if not panels:
        print("No histograms produced (near/far OOD buffers were empty).")
        return

    plt.close("all")  # clear figures left open by trainer.test()

    n = len(panels)
    ncols = min(max_cols, n)
    nrows = (n + ncols - 1) // ncols
    fig_w, fig_h = 4.2 * ncols, 3.4 * nrows
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, fig_h),
        dpi=110,
        squeeze=False,
        layout="constrained",
    )
    fig.suptitle(f"OOD scores — {tag}", fontsize=15, fontweight="semibold")

    id_style = {"color": "#4C72B0", "edgecolor": "#2A4A6F", "label": "In-distribution"}
    ood_style = {"color": "#DD8452", "edgecolor": "#9E4F2A", "label": "Out-of-distribution"}
    axes_flat = axes.flatten()

    for ax, (ds_name, ood_scores) in zip(axes_flat, panels, strict=False):
        ax.hist(
            id_scores,
            bins=bins,
            density=True,
            alpha=0.55,
            linewidth=0.8,
            **id_style,
        )
        ax.hist(
            ood_scores,
            bins=bins,
            density=True,
            alpha=0.55,
            linewidth=0.8,
            **ood_style,
        )
        ax.set_title(ds_name.replace("_", " "), fontsize=11)
        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        ax.grid(True, linestyle="--", alpha=0.35, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=2,
        frameon=False,
        fontsize=10,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    if save:
        path = out_dir / f"hist_grid_{tag}.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        print(f"Saved {path}")
    if show:
        try:
            from IPython.display import display

            display(fig)
        except ImportError:
            plt.show()
    plt.close(fig)


root = Path("data1")
datamodule = CIFAR100DataModule(root=root, batch_size=200, eval_ood=True, eval_shift=True)
trainer = TUTrainer(
    accelerator="gpu",
    enable_progress_bar=True,
    devices=1,
    logger=TensorBoardLogger(save_dir="logs/ood_tutorial", name="run"),
)


# %%
# 4. Define the classification routine and launch the test
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define the classification routine for evaluation. We use the CrossEntropyLoss
# as the loss function since we are working on a classification task.
# The routine is configured to handle OOD detection and distributional shifts using the specified model, loss function, and evaluation criteria.

routine = ClassificationRoutine(
    num_classes=datamodule.num_classes,
    eval_ood=True,
    model=net,
    loss=nn.CrossEntropyLoss(),
    eval_shift=True,
    ood_criterion="neco",
    log_plots=True,
)

# Perform testing using the defined routine and datamodule.
results = trainer.test(model=routine, datamodule=datamodule)
plot_ood_histograms(routine, "neco")

# We can test also different ood techniques simpy by changing the ood_criterion argument below is an example using ASH technique
routine = ClassificationRoutine(
    num_classes=datamodule.num_classes,
    eval_ood=True,
    model=net,
    loss=nn.CrossEntropyLoss(),
    eval_shift=True,
    ood_criterion="ash",
    log_plots=True,
)
results = trainer.test(model=routine, datamodule=datamodule)
plot_ood_histograms(routine, "ash")


# Example using REACT technique
routine = ClassificationRoutine(
    num_classes=datamodule.num_classes,
    eval_ood=True,
    model=net,
    loss=nn.CrossEntropyLoss(),
    eval_shift=True,
    ood_criterion="react",
    log_plots=True,
)
results = trainer.test(model=routine, datamodule=datamodule)
plot_ood_histograms(routine, "react")


# %%
# Here, we show the various test metrics along with the ood eval metrics, auroc,aupr and fpr95 on Near and far ood datasets defined per defualt according to OpenOOD splits (link to library)


# %%
# 5. Defining custom ood datasets
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# If you don't want to use the open ood datasets or dataset splits, you can pass your own datasets in a list to near_ood_datasets or far_ood_datasets datamodule arguments
# and use them for ood evaluation but make sure they inherit from the
# Dataset class from torch.utils.data, below is an example of such a case.

from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import v2

test_transform = v2.Compose(
    [
        v2.ToImage(),
        v2.Resize(32),
        v2.CenterCrop(32),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.5071, 0.4867, 0.4408)),
    ]
)

custom_dataset1 = CIFAR10(root=root, train=False, download=True, transform=test_transform)
custom_dataset2 = MNIST(root=root, train=False, download=True, transform=test_transform)

datamodule = CIFAR100DataModule(
    root=root,
    batch_size=200,
    eval_ood=True,
    eval_shift=True,
    near_ood_datasets=[custom_dataset1],
    far_ood_datasets=[custom_dataset2],
)

# Perform testing using the CUSTOM defined ood datasets.
results = trainer.test(model=routine, datamodule=datamodule)
plot_ood_histograms(routine, "react_custom")








# %%
# ## References
#
# - **OpenOOD:** Jingyang Zhang et al. — [NeurIPS 2025](https://arxiv.org/pdf/2306.09301). OpenOOD v1.5: Enhanced Benchmark for Out-of-Distribution Detection.
