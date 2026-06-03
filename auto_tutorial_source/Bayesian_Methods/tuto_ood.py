"""Simple Ood Evaluation
================================================


In this tutorial, we’ll demonstrate how to perform out-of-distribution (OOD) evaluation using TorchUncertainty’s datamodules and routines. You’ll learn to:

1. **Set up a CIFAR-100 datamodule** that automatically handles in-distribution, near-OOD, and far-OOD splits.
2. **Run the `ClassificationRoutine`** to compute both in-distribution accuracy and OOD metrics (AUROC, AUPR, FPR95).
3. **Plug in your own OOD datasets** for fully custom evaluation.

Foreword on Out-of-Distribution Detection
-----------------------------------------

Out-of-Distribution (OOD) detection measures a model’s ability to recognize inputs that differ from its training distribution. TorchUncertainty integrates common OOD metrics directly into the Lightning test loop, including:

- **AUROC** (Area Under the ROC Curve)
- **AUPR** (Area Under the Precision-Recall Curve)
- **FPR95** (False Positive Rate at 95% True Positive Rate)

With just a few lines of code you can compare in-distribution performance to OOD detection performance under both “near” and “far” shifts. Per default, TorchUncertainty uses the
popular OpenOOD library to define the near and far OOD datasets and splits. You can also use your own datasets by passing them to the datamodule.

Supported Datamodules and Default OOD Splits
--------------------------------------------

.. list-table:: Datamodules & Default OOD Splits
   :header-rows: 1
   :widths: 20 15 20 20

   * - **Datamodule**
     - **In-Domain**
     - **Default Near-OOD (Hard)**
     - **Default Far-OOD (Easy)**
   * - ``CIFAR10DataModule``
     - CIFAR-10
     - CIFAR-100, Tiny ImageNet
     - MNIST, SVHN, Textures, Places365
   * - ``CIFAR100DataModule``
     - CIFAR-100
     - CIFAR-10, Tiny ImageNet
     - MNIST, SVHN, Textures, Places365
   * - ``ImageNetDataModule``
     - ImageNet-1K
     - SSB-hard, NINCO
     - iNaturalist, Textures, OpenImage-O
   * - ``ImageNet200DataModule``
     - ImageNet200
     - SSB-hard, NINCO
     - iNaturalist, Textures, OpenImage-O

Supported OOD Criteria
----------------------

.. list-table:: Supported OOD Criteria
   :header-rows: 1
   :widths: 15 50

   * - **Criterion**
     - **Original Reference (Year, Venue)**
   * - ``msp``
     - Hendrycks & Gimpel, A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks `ICLR Workshop 2017 <https://arxiv.org/abs/1610.02136>`_.
   * - ``Maxlogit``
     - /
   * - ``energy``
     - Liu et al., Energy-based Out-of-Distribution Detection `NeurIPS 2020 <https://arxiv.org/abs/2010.03759>`_.
   * - ``odin``
     - Liang, Li & Srikant, Enhancing The Reliability of Out-of-Distribution Image Detection in Neural Networks `ICML 2018 <https://arxiv.org/abs/1706.02690>`_.
   * - ``entropy``
     - /
   * - ``mutual_information``
     - /
   * - ``variation_ratio``
     - /
   * - ``scale``
     - Scaling Out-of-Distribution Detection for Real-World Settings Hendrycks et al. `ICML 2022 <https://proceedings.mlr.press/v162/hendrycks22a/hendrycks22a.pdf>`_.
   * - ``ash``
     - AASH: Extremely Simple Activation Shaping for OOD Detection, Djurisic et al. `ICLR 2023 <https://arxiv.org/pdf/2209.09858>`_.
   * - ``react``
     - ReAct: Out-of-distribution Detection with Rectified Activations, Sun et al. `NeurIPS 2021 <https://proceedings.neurips.cc/paper/2021/file/01894d6f048493d2cacde3c579c315a3-Paper.pdf>`_.
   * - ``adascale_a``
     - AdaSCALE: Adaptive Scaling for OOD Detection `Regmi et al. <https://arxiv.org/pdf/2503.08023>`_.
   * - ``vim``
     - ViM: Out-of-Distribution with Virtual-Logit Matching, Wang et al. `CVPR 2022 <https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_ViM_Out-of-Distribution_With_Virtual-Logit_Matching_CVPR_2022_paper.pdf>`_.
   * - ``knn``
     - Out-of-Distribution Detection with Deep Nearest Neighbors, Sun et al. `ICML 2022 <https://arxiv.org/abs/2106.01477>`_.
   * - ``gen``
     - GEN: Generalized ENtropy Score for OOD Detection, Liu et al. `CVPR 2023 <https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_GEN_Pushing_the_Limits_of_Softmax-Based_Out-of-Distribution_Detection_CVPR_2023_paper.pdf>`_.
   * - ``nnguide``
     - NNGuide: Nearest-Neighbor Guidance for OOD Detection, Park et al. `ICCV 2023 <https://openaccess.thecvf.com/content/ICCV2023/papers/Park_Nearest_Neighbor_Guidance_for_Out-of-Distribution_Detection_ICCV_2023_paper.pdf>`_.

.. note::

   - All of these criteria can be passed as the `ood_criterion` argument to
     `ClassificationRoutine`.
   - Methods marked “ensemble-only” will require multiple stochastic passes.



.. note::

   - **Near-OOD** splits are semantically similar to the in-domain data.
   - **Far-OOD** splits come from more distant distributions (e.g., ImageNet variants).
   - Override defaults by passing your own ``near_ood_datasets`` / ``far_ood_datasets``.


1. Loading the utilities
~~~~~~~~~~~~~~~~~~~~~~~~

To eval ood using TorchUncertainty, we have to load the following:

- the model:ResNet18_32x32 trained on in-domain data cifar100
- the classification routine from torch_uncertainty.routines
- the datamodule that handles dataloaders: CIFAR100DataModule from torch_uncertainty.datamodules.
"""

# %%
from pathlib import Path

# %%
# 2. Load the trained model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# In this tutorial we will be loading a pretrained model, but you can also train your own using the same classification routine and still get ood related metrics at test phase.
import torch
from huggingface_hub import hf_hub_download

from torch_uncertainty.models.resnet import resnet

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

import torch.nn as nn

from torch_uncertainty import TUTrainer
from torch_uncertainty.datamodules import CIFAR100DataModule
from torch_uncertainty.routines import ClassificationRoutine

root = Path("data1")
datamodule = CIFAR100DataModule(root=root, batch_size=200, eval_ood=True, eval_shift=True)
trainer = TUTrainer(accelerator="gpu", enable_progress_bar=True)


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
    ood_criterion="ash",
)

# Perform testing using the defined routine and datamodule.
results = trainer.test(model=routine, datamodule=datamodule)


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


# %%
# References
# ----------
# - **OpenOOD:** Jingyang Zhang & al. (`Neurips 2025 <https://arxiv.org/pdf/2306.09301>`_). OpenOOD v1.5: Enhanced Benchmark for Out-of-Distribution Detection.
