# ruff: noqa: E402, D212, D415, T201
"""
Improved Ensemble parameter-efficiency with Packed-Ensembles
============================================================

*This tutorial is adapted from a notebook part of a lecture given at the* |conference|_ *by Sebastian Starke, Peter Steinbach, Gianni Franchi, and Olivier Laurent.*

.. _conference: https://haicon24.de/

.. |conference| replace:: *Helmholtz AI Conference*

In this notebook will work on the MNIST dataset that was introduced by Corinna Cortes, Christopher J.C. Burges, and later modified by Yann LeCun in the foundational paper:

- `Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE. <http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf>`_.

The MNIST dataset consists of 70 000 images of handwritten digits from 0 to 9. The images are grayscale and 28x28-pixel sized. The task is to classify the images into their respective digits. The dataset can be automatically downloaded using the `torchvision` library.

In this notebook, we will train a model and an ensemble on this task and evaluate their performance. The performance will consist in the following metrics:

- Accuracy: the proportion of correctly classified images,
- Brier score: a measure of the quality of the predicted probabilities,
- Calibration error: a measure of the calibration of the predicted probabilities,
- Negative Log-Likelihood: the value of the loss on the test set.

Throughout this notebook, we abstract the training and evaluation process using `PyTorch Lightning <https://lightning.ai/docs/pytorch/stable/>`_
and `TorchUncertainty <https://torch-uncertainty.github.io/>`_.

Similarly to keras for tensorflow, PyTorch Lightning is a high-level interface for PyTorch that simplifies the training and evaluation process using a Trainer.
TorchUncertainty is partly built on top of PyTorch Lightning and provides tools to train and evaluate models with uncertainty quantification.

TorchUncertainty includes datamodules that handle the data loading and preprocessing. We don't use them here for tutorial purposes.

1. Download, instantiate and visualize the datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The dataset is automatically downloaded using torchvision. We then visualize a few images to see a bit what we are working with.
"""

# %%
import torch
import torchvision.transforms as T

# We set the number of epochs to some very low value for the sake of time
MAX_EPOCHS = 3

# Create the transforms for the images
train_transform = T.Compose(
    [
        T.ToTensor(),
        # We perform random cropping as data augmentation
        T.RandomCrop(28, padding=4),
        # As for the MNIST1d dataset, we normalize the data
        T.Normalize((0.1307,), (0.3081,)),
    ]
)
test_transform = T.Compose(
    [
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
        T.CenterCrop(28),
        T.Normalize((0.1307,), (0.3081,)),
    ]
)

# Download and instantiate the dataset
from torch.utils.data import Subset
from torchvision.datasets import MNIST, FashionMNIST

train_data = MNIST(root="./data/", download=True, train=True, transform=train_transform)
test_data = MNIST(root="./data/", train=False, transform=test_transform)
# We only take the first 10k images to have the same number of samples as the test set using torch Subsets
ood_data = Subset(
    FashionMNIST(root="./data/", download=True, transform=test_transform),
    indices=range(10000),
)

# Create the corresponding dataloaders
from torch.utils.data import DataLoader

train_dl = DataLoader(train_data, batch_size=512, shuffle=True, num_workers=8)
test_dl = DataLoader(test_data, batch_size=2048, shuffle=False, num_workers=4)
ood_dl = DataLoader(ood_data, batch_size=2048, shuffle=False, num_workers=4)

# %%
# You could replace all this cell by simply loading the MNIST datamodule from TorchUncertainty.
# Now, let's visualize a few images from the dataset. For this task, we use the viz_data dataset that applies no transformation to the images.

# Datasets without transformation to visualize the unchanged data
viz_data = MNIST(root="./data/", train=False)
ood_viz_data = FashionMNIST(root="./data/", download=True)

print("In distribution data:")
viz_data[0][0]
# %%
print("Out of distribution data:")
ood_viz_data[0][0]

# %%
# 2. Create & train the model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We will create a simple convolutional neural network (CNN): the LeNet model (also introduced by LeCun).
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, (5, 5))
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        self.pooling = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = torch.flatten(out, 1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        return self.fc3(out)  # No softmax in the model!


# Instantiate the model, the images are in grayscale so the number of channels is 1
model = LeNet(in_channels=1, num_classes=10)

# %%
# We now need to define the optimization recipe:
# - the optimizer, here the standard stochastic gradient descent (SGD) with a learning rate of 0.05
# - the scheduler, here cosine annealing.


def optim_recipe(model, lr_mult: float = 1.0):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05 * lr_mult)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    return {"optimizer": optimizer, "scheduler": scheduler}


# %%
# To train the model, we use `TorchUncertainty <https://torch-uncertainty.github.io/>`_, a library that we have developed to ease
# the training and evaluation of models with uncertainty.
#
# **Note:** To train supervised classification models we most often use the cross-entropy loss.
# With weight-decay, minimizing this loss amounts to finding a Maximum a posteriori (MAP) estimate of the model parameters.
# This means that the model is trained to predict the most likely class for each input given a diagonal Gaussian prior on the weights.


from torch_uncertainty import TUTrainer
from torch_uncertainty.routines import ClassificationRoutine

# Create the trainer that will handle the training
trainer = TUTrainer(accelerator="gpu", devices=1, max_epochs=MAX_EPOCHS, enable_progress_bar=False)

# The routine is a wrapper of the model that contains the training logic with the metrics, etc
routine = ClassificationRoutine(
    num_classes=10,
    model=model,
    loss=nn.CrossEntropyLoss(),
    optim_recipe=optim_recipe(model),
    eval_ood=True,
)

# In practice, avoid performing the validation on the test set (if you do model selection)
trainer.fit(routine, train_dataloaders=train_dl, val_dataloaders=test_dl)

# %%
# Evaluate the trained model on the test set - pay attention to the cls/Acc metric
perf = trainer.test(routine, dataloaders=[test_dl, ood_dl])

# %%
# This table provides a lot of information:
#
# **OOD Detection: Binary Classification MNIST vs. FashionMNIST**
#
# - AUPR/AUROC/FPR95: Measures the quality of the OOD detection. The higher the better for AUPR and AUROC, the lower the better for FPR95.
#
# **Calibration: Reliability of the Predictions**
#
# - ECE: Expected Calibration Error. The lower the better.
# - aECE: Adaptive Expected Calibration Error. The lower the better. (~More precise version of the ECE)
#
# **Classification Performance**
#
# - Accuracy: The ratio of correctly classified images. The higher the better.
# - Brier: The quality of the predicted probabilities (Mean Squared Error of the predictions vs. ground-truth). The lower the better.
# - Negative Log-Likelihood: The value of the loss on the test set. The lower the better.
#
# **Selective Classification & Grouping Loss**
# - We talk about these points later in the "To go further" section.
#
# By setting `eval_shift` to True, we could also evaluate the performance of the models on MNIST-C, a dataset close to MNIST but with perturbations.
#
# 3. Training an ensemble of models with TorchUncertainty
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# You have two options here, you can either train the ensemble directly if you have enough memory,
# otherwise, you can train independent models and do the ensembling during the evaluation (sometimes called inference).
#
# In this case, we will do it sequentially. In this tutorial, you have the choice between training multiple models,
# which will take time if you have no GPU, or downloading the pre-trained models that we have prepared for you.
#
# **Training the ensemble**
#
# To train the ensemble, you will have to use the "deep_ensembles" function from TorchUncertainty, which will
# replicate and change the initialization of your networks to ensure diversity.

from torch_uncertainty.methods import deep_ensembles
from torch_uncertainty.transforms import RepeatTarget

# Create the ensemble model
ensemble = deep_ensembles(
    LeNet(in_channels=1, num_classes=10),
    num_estimators=2,
    task="classification",
    reset_model_parameters=True,
)

trainer = TUTrainer(accelerator="gpu", devices=1, max_epochs=MAX_EPOCHS)
ens_routine = ClassificationRoutine(
    is_ensemble=True,
    num_classes=10,
    model=ensemble,
    loss=nn.CrossEntropyLoss(),  # The loss for the training
    format_batch_fn=RepeatTarget(2),  # How to handle the targets when comparing the predictions
    optim_recipe=optim_recipe(
        ensemble, 2.0
    ),  # The optimization scheme with the optimizer and the scheduler as a dictionnary
    eval_ood=True,  # We want to evaluate the OOD-related metrics
)
trainer.fit(ens_routine, train_dataloaders=train_dl, val_dataloaders=test_dl)
ens_perf = trainer.test(ens_routine, dataloaders=[test_dl, ood_dl])

# %%
# Feel free to run the notebook on your machine for a longer duration.
#
# We need to multiply the learning rate by 2 to account for the fact that we have 2 models
# in the ensemble and that we average the loss over all the predictions.
#
# **Downloading the pre-trained models**
#
# We have put the pre-trained models on Hugging Face that you can download with the utility function
# "hf_hub_download" imported just below. These models are trained for 75 epochs and are therefore not
# comparable to the all the others trained in this notebook. The pretrained models can be seen
# on `HuggingFace <https://huggingface.co/ENSTA-U2IS/tutorial-models>`_ and TorchUncertainty's are `there <https://huggingface.co/torch-uncertainty>`_.

from torch_uncertainty.utils.hub import hf_hub_download

all_models = []
for i in range(8):
    hf_hub_download(
        repo_id="ENSTA-U2IS/tutorial-models",
        filename=f"version_{i}.ckpt",
        local_dir="./models/",
    )
    model = LeNet(in_channels=1, num_classes=10)
    state_dict = torch.load(f"./models/version_{i}.ckpt", map_location="cpu", weights_only=True)[
        "state_dict"
    ]
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    all_models.append(model)

from torch_uncertainty.models import deep_ensembles
from torch_uncertainty.transforms import RepeatTarget

ensemble = deep_ensembles(
    all_models,
    num_estimators=None,
    task="classification",
    reset_model_parameters=True,
)

ens_routine = ClassificationRoutine(
    is_ensemble=True,
    num_classes=10,
    model=ensemble,
    loss=nn.CrossEntropyLoss(),  # The loss for the training
    format_batch_fn=RepeatTarget(8),  # How to handle the targets when comparing the predictions
    optim_recipe=None,  # No optim recipe as the model is already trained
    eval_ood=True,  # We want to evaluate the OOD-related metrics
)

trainer = TUTrainer(accelerator="gpu", devices=1, max_epochs=MAX_EPOCHS)

ens_perf = trainer.test(ens_routine, dataloaders=[test_dl, ood_dl])

# %%
# 4. From Deep Ensembles to Packed-Ensembles
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In the paper `Packed-Ensembles for Efficient Uncertainty Quantification <https://arxiv.org/abs/2210.09184>`_
# published at the International Conference on Learning Representations (ICLR) in 2023, we introduced a
# modification of Deep Ensembles to make it more computationally-efficient. The idea is to pack the ensemble
# members into a single model, which allows us to train the ensemble in a single forward pass.
# This modification is particularly useful when the ensemble size is large, as it is often the case in practice.
#
# We will need to update the model and replace the layers with their Packed equivalents. You can find the
# documentation of the Packed-Linear layer using this `link <https://torch-uncertainty.github.io/generated/torch_uncertainty.layers.PackedLinear.html>`_,
# and the Packed-Conv2D, `here <https://torch-uncertainty.github.io/generated/torch_uncertainty.layers.PackedLinear.html>`_.

import torch
import torch.nn as nn

from torch_uncertainty.layers import PackedConv2d, PackedLinear


class PackedLeNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        alpha: int,
        num_estimators: int,
    ) -> None:
        super().__init__()
        self.num_estimators = num_estimators
        self.conv1 = PackedConv2d(
            in_channels,
            6,
            (5, 5),
            alpha=alpha,
            num_estimators=num_estimators,
            first=True,
        )
        self.conv2 = PackedConv2d(
            6,
            16,
            (5, 5),
            alpha=alpha,
            num_estimators=num_estimators,
        )
        self.pooling = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = PackedLinear(256, 120, alpha=alpha, num_estimators=num_estimators)
        self.fc2 = PackedLinear(120, 84, alpha=alpha, num_estimators=num_estimators)
        self.fc3 = PackedLinear(
            84,
            num_classes,
            alpha=alpha,
            num_estimators=num_estimators,
            last=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = torch.flatten(out, 1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        return self.fc3(out)  # Again, no softmax in the model


# Instantiate the model, the images are in grayscale so the number of channels is 1
packed_model = PackedLeNet(in_channels=1, num_classes=10, alpha=2, num_estimators=4)

# Create the trainer that will handle the training
trainer = TUTrainer(accelerator="gpu", devices=1, max_epochs=MAX_EPOCHS)

# The routine is a wrapper of the model that contains the training logic with the metrics, etc
packed_routine = ClassificationRoutine(
    is_ensemble=True,
    num_classes=10,
    model=packed_model,
    loss=nn.CrossEntropyLoss(),
    format_batch_fn=RepeatTarget(4),
    optim_recipe=optim_recipe(packed_model, 4.0),
    eval_ood=True,
)

# In practice, avoid performing the validation on the test set
trainer.fit(packed_routine, train_dataloaders=train_dl, val_dataloaders=test_dl)

packed_perf = trainer.test(packed_routine, dataloaders=[test_dl, ood_dl])

# %%
# The training time should be approximately similar to the one of the single model that you trained before. However, please note that we are working with very small models, hence completely underusing your GPU. As such, the training time is not representative of what you would observe with larger models.
#
# You can read more on Packed-Ensembles in the `paper <https://arxiv.org/abs/2210.09184>`_ or the `Medium <https://medium.com/@adrien.lafage/make-your-neural-networks-more-reliable-with-packed-ensembles-7ad0b737a873>`_ post.
#
# To Go Further & More Concepts of Uncertainty in ML
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# **Question 1:** Have a look at the models in the "lightning_logs". If you are on your own machine, try to visualize the learning curves with `tensorboard --logdir lightning_logs`.
#
# **Question 2:** Add a cell below and try to find the errors made by packed-ensembles on the test set. Visualize the errors and their labels and look at the predictions of the different sub-models. Are they similar? Can you think of uncertainty scores that could help you identify these errors?
#
# Selective Classification
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# Selective classification or "prediction with rejection" is a paradigm in uncertainty-aware machine learning where the model can decide not to make a prediction if the confidence score given by the model is below some pre-computed threshold. This can be useful in real-world applications where the cost of making a wrong prediction is high.
#
# In constrast to calibration, the values of the confidence scores are not important, only the order of the scores. *Ideally, the best model will order all the correct predictions first, and all the incorrect predictions last.* In this case, there will be a threshold so that all the predictions above the threshold are correct, and all the predictions below the threshold are incorrect.
#
# In TorchUncertainty, we look at 3 different metrics for selective classification:
#
# - **AURC**: The area under the Risk (% of errors) vs. Coverage (% of classified samples) curve. This curve expresses how the risk of the model evolves as we increase the coverage (the proportion of predictions that are above the selection threshold). This metric will be minimized by a model able to perfectly separate the correct and incorrect predictions.
#
# The following metrics are computed at a fixed risk and coverage level and that have practical interests. The idea of these metrics is that you can set the selection threshold to achieve a certain level of risk and coverage, as required by the technical constraints of your application:
#
# - **Coverage at 5% Risk**: The proportion of predictions that are above the selection threshold when it is set for the risk to egal 5%. Set the risk threshold to your application constraints. The higher the better.
# - **Risk at 80% Coverage**: The proportion of errors when the coverage is set to 80%. Set the coverage threshold to your application constraints. The lower the better.
#
# Grouping Loss
# ^^^^^^^^^^^^^
#
# The grouping loss is a measure of uncertainty orthogonal to calibration. Have a look at `this paper <https://arxiv.org/abs/2210.16315>`_ to learn about it. Check out their small library `GLest <https://github.com/aperezlebel/glest>`_. TorchUncertainty includes a wrapper of the library to compute the grouping loss with eval_grouping_loss parameter.
