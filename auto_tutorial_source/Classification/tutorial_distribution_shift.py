# ruff: noqa: D212, D415, T201
"""
Evaluating Model Performance Under Distribution Shift with TorchUncertainty
===========================================================================

In this tutorial, we explore how to assess a model's robustness when faced with distribution shifts.
Specifically, we will:

- Shortly train a **ResNet18** model on the standard **CIFAR-10** dataset.
- Evaluate its performance on both the original CIFAR-10 test set and a corrupted version of CIFAR-10 to simulate distribution shift.
- Analyze the model's performance and robustness under these conditions.

By the end of this tutorial, you will understand how to use TorchUncertainty to evaluate and interpret model behavior under distribution shifts.
"""

# %%
# Imports and Setup
# -----------------
#
# First, we need to import the necessary libraries and set up our environment.
# This includes importing PyTorch, TorchUncertainty components, and TorchUncertainty's Trainer (built on top of Lightning's).

import os

from torch import nn, optim

from torch_uncertainty import TUTrainer
from torch_uncertainty.datamodules import CIFAR10DataModule
from torch_uncertainty.models.classification.resnet import resnet
from torch_uncertainty.routines.classification import ClassificationRoutine

# %%
# DataModule Setup
# ----------------
#
# TorchUncertainty provides convenient DataModules for standard datasets like CIFAR-10.
# DataModules handle data loading, preprocessing, and batching, simplifying the data pipeline. Each datamodule
# also include the corresponding out-of-distribution and distribution shift datasets, which are then used by the routine.
# For CIFAR-10, the corresponding distribution-shift dataset is CIFAR-10C as used in the community.
# To enable Distribution Shift evaluation, activate the `eval_shift` flag as done below.

# Initialize the CIFAR-10 DataModule
datamodule = CIFAR10DataModule(
    root=os.environ.get("TU_DATA_DIR", "data"),
    batch_size=512,
    num_workers=8,
    eval_shift=True,
    shift_severity=5,  # Set severity level of the corruption (1 to 5): max-strength!
)

# %%
# CIFAR-10C
# ---------
#
# CIFAR-10C is a transformed version of CIFAR-10 test set. Dan Hendrycks and Thomas Dietterich applied computer vision
# transforms, known as corruptions to degrade the quality of the image and test deep learning models in adverse conditions.
# There are 15 (+4 optional) corruptions in total, including noise, blur, weather effects, etc. Each corruption has 5 different
# levels of severity ranging from small corruptions to very strong effects on the image. You can set the desired corruption level with
# the shift-severity argument. We refer to [1] for more details.
# You can get a more detailed overview and examples of the corruptions on the corresponding tutorial.

# These lines are usually not necessary (they are called by the Trainer),
# but we want to get access to the dataset before training
datamodule.prepare_data()
datamodule.setup("test")

# Let's check the CIFAR-10C, it should contain (15+4)*10000 images for the selected severity level.
print(datamodule.shift)

# %%
# Model Initialization
# --------------------
#
# We will use the ResNet18 architecture, a widely adopted convolutional neural network known for its deep residual learning capabilities.
# The model is initialized with 10 output classes corresponding to the CIFAR-10 dataset categories.

# Initialize the ResNet18 model with 10 output classes
model = resnet(arch=18, in_channels=3, num_classes=10)

# %%
# Define the Classification Routine
# ---------------------------------
#
# The `ClassificationRoutine` is one of the most crucial building blocks in TorchUncertainty.
# It streamlines the training and evaluation processes.
# It integrates the model, loss function, and optimizer into a cohesive routine compatible with PyTorch Lightning's Trainer.
# This abstraction simplifies the implementation of standard training loops and evaluation protocols.
# To come back to what matters in this tutorial, the routine also handles the evaluation of the performance
# of the model under distribution shift detection. To enable it, activate the `eval_shift` flag. Note that you can also evaluate
# the Out-of-distribution detection at the same time by also setting `eval_ood` to True.

# Define the loss function: Cross-Entropy Loss for multi-class classification
criterion = nn.CrossEntropyLoss()

# Define the optimizer: Adam optimizer with a learning rate of 0.001
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize the ClassificationRoutine with the model, number of classes, loss function, and optimizer
routine = ClassificationRoutine(
    model=model, num_classes=10, loss=criterion, optim_recipe=optimizer, eval_shift=True
)

# %%
# Training the Model
# ------------------
#
# With the routine defined, we can now set up the TUTrainer and commence training.
# The TUTrainer handles the training loop, including epoch management, logging, and checkpointing.
# We specify the maximum number of epochs, the precision and the device to be used.

# Initialize the TUTrainer with a maximum of 10 epochs and the specified device
trainer = TUTrainer(
    max_epochs=10, precision="16-mixed", accelerator="cuda", devices=1, enable_progress_bar=False
)

# Begin training the model using the CIFAR-10 DataModule
trainer.fit(routine, datamodule=datamodule)

# %%
# Evaluating on In-Distribution and Distribution-shifted Data
# -----------------------------------------------------------
#
# Now that the model is trained, we can evaluate its performance on the original in-distribution test set,
# as well as the distribution-shifted set. Typing the next line will automatically compute the in-distribution
# metrics as well as their values on the distribution-shifted set.

# Evaluate the trained model on the original CIFAR-10 test set and on CIFAR-10C
results = trainer.test(routine, datamodule=datamodule)

# %%
# Distribution-shift metrics
# --------------------------
#
# The distribution shift metrics are computed only when the `eval_shift` flag of the routine is True.
# In this case, the values of the metrics are shown last. They correspond to the in-distribution metrics but
# computed on the distribution-shifted datasets, hence the worse results.
