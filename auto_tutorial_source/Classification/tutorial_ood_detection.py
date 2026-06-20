# ruff: noqa: E402, D212, D415
"""
Out-of-distribution detection with TorchUncertainty
===================================================

This tutorial demonstrates how to perform OOD detection using
TorchUncertainty's ClassificationRoutine with a ResNet18 model trained on CIFAR-10,
evaluating its performance with SVHN as the OOD dataset.

We will:

- Set up the CIFAR-10 datamodule.
- Initialize and shortly train a ResNet18 model using the ClassificationRoutine.
- Evaluate the model's performance on both in-distribution and out-of-distribution data.
- Analyze uncertainty metrics for OOD detection.
"""

# %%
# Imports and Setup
# ------------------
#
# First, we need to import the necessary libraries and set up our environment.
# This includes importing PyTorch, TorchUncertainty components, and TorchUncertainty's Trainer (built on top of Lightning's),
# as well as two criteria for OOD detection, the maximum softmax probability [1] and the Max Logit [2].
import os

from torch import nn, optim

from torch_uncertainty import TUTrainer
from torch_uncertainty.datamodules import CIFAR10DataModule
from torch_uncertainty.models.classification.resnet import resnet
from torch_uncertainty.ood_criteria import MaxLogitCriterion, MaxSoftmaxCriterion
from torch_uncertainty.routines.classification import ClassificationRoutine

# %%
# DataModule Setup
# ----------------
#
# TorchUncertainty provides convenient DataModules for standard datasets like CIFAR-10.
# DataModules handle data loading, preprocessing, and batching, simplifying the data pipeline. Each datamodule
# also include the corresponding out-of-distribution and distribution shift datasets, which are then used by the routine.
# For CIFAR-10, the corresponding OOD-detection dataset is SVHN as used in the community.
# To enable OOD evaluation, activate the `eval_ood` flag as done below.

datamodule = CIFAR10DataModule(
    root=os.environ.get("TU_DATA_DIR", "data"), batch_size=512, num_workers=8, eval_ood=True
)

# %%
# Model Initialization
# --------------------
#
# We use the ResNet18 architecture, a widely adopted convolutional neural network known for its deep residual learning capabilities.
# The model is initialized with 10 output classes corresponding to the CIFAR-10 dataset categories. When training on CIFAR, do not forget to
# set the style of the resnet to CIFAR, otherwise it will lose more information in the first convolution.

# Initialize the ResNet18 model
model = resnet(arch=18, in_channels=3, num_classes=10, style="cifar", conv_bias=False)

# %%
# Define the Classification Routine
# ---------------------------------
#
# The `ClassificationRoutine` is one of the most crucial building blocks in TorchUncertainty.
# It streamlines the training and evaluation processes.
# It integrates the model, loss function, and optimizer into a cohesive routine compatible with PyTorch Lightning's Trainer.
# This abstraction simplifies the implementation of standard training loops and evaluation protocols.
# To come back to what matters in this tutorial, the routine also handles OOD detection. To enable it,
# just activate the `eval_ood` flag. Note that you can also evaluate the distribution-shift performance
# of the model at the same time by also setting `eval_shift` to True.

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize the ClassificationRoutine, you could replace MaxSoftmaxCriterion by "msp"
routine = ClassificationRoutine(
    model=model,
    num_classes=10,
    loss=criterion,
    optim_recipe=optimizer,
    eval_ood=True,
    ood_criterion=MaxSoftmaxCriterion,
)

# %%
# Test the Training of the Model
# ------------------------------
#
# With the routine defined, we can now set up the Trainer and commence training.
# The Trainer handles the training loop, including epoch management, logging, and checkpointing.
# We specify the maximum number of epochs, the precision and the device to be used. To reduce the tutorial building time,
# we will train for a single epoch and load a model from `TorchUncertainty's HuggingFace <https://huggingface.co/torch-uncertainty>`_.

# Initialize the TUTrainer
trainer = TUTrainer(
    max_epochs=1, precision="16-mixed", accelerator="cuda", devices=1, enable_progress_bar=False
)

# Train the model for 1 epoch using the CIFAR-10 DataModule
trainer.fit(routine, datamodule=datamodule)

# %%
# Load the model from HuggingFace
# -------------------------------
#
# We simply download a ResNet-18 trained on CIFAR-10 from `TorchUncertainty's HuggingFace <https://huggingface.co/torch-uncertainty>`_ and load it with
# the `load_from_checkpoint` method.

import torch
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="torch-uncertainty/resnet18_c10",
    filename="resnet18_c10.ckpt",
)
state_dict = torch.load(path, map_location="cpu", weights_only=True)
routine.model.load_state_dict(state_dict)

# %%
# Evaluating on In-Distribution and Out-of-distribution Data
# ----------------------------------------------------------
#
# Now that the model is trained, we can evaluate its performance on the original in-distribution test set,
# as well as the OOD set. Typing the next line will automatically compute the in-distribution and OOD detection metrics.

# Evaluate the model on the CIFAR-10 (IID) and SVHN (OOD) test sets
results = trainer.test(routine, datamodule=datamodule)

# %%
# Changing the OOD Criterion
# --------------------------
#
# The previous metrics for Out-of-distribution detection have been computed using the maximum softmax probability score [1],
# which corresponds to the likelihood of the prediction. We could use other scores such as the maximum logit [2]. To do this,
# just change the routine's `ood_criterion` and perform a second test.
routine.ood_criterion = MaxLogitCriterion()

results = trainer.test(routine, datamodule=datamodule)

# %%
# Note that you could create your own class if you want to implement a custom OOD detection score. When changing the
# Out-of-distribution criterion, all the In-distribution metrics remain the same. The only values that change
# are those of the regrouped in the OOD Detection category. Here we see that the AUPR, AUROC and FPR95 are worse using the maximum
# logit score compared to the maximum softmax probability but it could depend on the model you are using.
#
# References
# ----------
#
# [1] Hendrycks, D., & Gimpel, K. (2016). A baseline for detecting misclassified and out-of-distribution examples in neural networks. In ICLR 2017.
#
# [2] Hendrycks, D., Basart, S., Mazeika, M., Zou, A., Kwon, J., Mostajabi, M., ... & Song, D. (2019). Scaling out-of-distribution detection for real-world settings. In ICML 2022.
