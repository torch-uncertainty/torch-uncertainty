# ruff: noqa: E402, D212, D415
"""
Training an MLP for Tabular Regression with TorchUncertainty
============================================================

In this tutorial, we will train a multi-layer-perceptron on a UCI regression dataset using TorchUncertainty.
You will discover two of the core tools from TorchUncertainty, namely

- the routine: a model wrapper, which handles the training and evaluation logics, here for regression
- the datamodules: python classes, which provide the dataloaders used by the routine

The regression routine is not as complete as the classification routine, so reach out if you would like the
team to implement more features or if you want to contribute!

1. Loading the utilities
~~~~~~~~~~~~~~~~~~~~~~~~

First, we have to load the following utilities from TorchUncertainty:

- the TUTrainer which mostly handles the link with the hardware (accelerators, precision, etc)
- the regression training & evaluation routine from torch_uncertainty.routines
- the datamodule handling dataloaders: TabularRegressionDatamodule from torch_uncertainty.datamodules
- the model: mlp from torch_uncertainty.models
"""

# %%
import os
from pathlib import Path

import torch
from torch import nn

from torch_uncertainty import TUTrainer
from torch_uncertainty.datamodules.tabular_regression import YachtDataModule
from torch_uncertainty.models.mlp import mlp
from torch_uncertainty.routines import RegressionRoutine

# %%
# 2. Defining the Trainer and the DataModule
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In the following, we first create the trainer and instantiate the datamodule that handles the UCI regression dataset,
# dataloaders and transforms.

trainer = TUTrainer(accelerator="gpu", devices=1, max_epochs=20, enable_progress_bar=False)

# datamodule providing the dataloaders to the trainer, specifically we use the yacht dataset
datamodule = YachtDataModule(
    root=Path(os.environ.get("TU_DATA_DIR", "data")), batch_size=32, num_workers=4
)

# %%
# 3. Instantiating the Pointwise Model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We create the model easily using the blueprint from torch_uncertainty.models. In this first case, we will just
# try to predict a pointwise prediction. Later, we will predict a distribution instead to model the uncertainty.

model = mlp(in_features=6, num_outputs=1, hidden_dims=[10, 10])

# %%
# 4. The Loss and the Training Routine
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# This is a classification problem, and we use CrossEntropyLoss as the (negative-log-)likelihood.
# We define the training routine using the classification routine from torch_uncertainty.routines.
# We provide the number of classes, the model, the optimization recipe, the loss, and tell the routine
# that our model is an ensemble at evaluation time with the `is_ensemble` flag to get the corresponding metrics.
routine = RegressionRoutine(
    output_dim=1,
    model=model,
    loss=nn.MSELoss(),
    optim_recipe=torch.optim.Adam(model.parameters(), lr=0.01),
    is_ensemble=True,
)

# %%
# 5. Gathering Everything and Training the Model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We can now train the model using the trainer. We pass the routine and the datamodule
# to the fit and test methods of the trainer. It will automatically evaluate uncertainty
# metrics that you will find in the table below.

trainer.fit(model=routine, datamodule=datamodule)
results = trainer.test(model=routine, datamodule=datamodule)

# %%
# The performance of the model is much better than random, this means that we have learnt
# patterns from the input data. However, we just provide pointwise estimates and do not model
# the uncertainty. To predict estimations of uncertainty in regression, we can try to optimize
# the parameters of some distributions, for instance, we could assume that the samples follow a
# Laplace distribution of unknown parameters.
#
# 6. Instantiating the Uncertain Model & its Routine
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# As before, we create the model easily using the blueprint from torch_uncertainty.models. This time,
# we state that we want to predict a Laplace distribution.
from torch_uncertainty.losses import DistributionNLLLoss

prob_routine = mlp(in_features=6, num_outputs=1, hidden_dims=[10, 10], dist_family="laplace")

loss = DistributionNLLLoss()

# We create a new routine, this time probabilistic
prob_routine = RegressionRoutine(
    output_dim=1,
    model=prob_routine,
    loss=loss,
    optim_recipe=torch.optim.Adam(prob_routine.parameters(), lr=0.01),
    dist_family="laplace",
)

# %%
# 7. Training & Testing the Probabilistic Model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

trainer = TUTrainer(accelerator="gpu", devices=1, max_epochs=20, enable_progress_bar=False)

trainer.fit(model=prob_routine, datamodule=datamodule)
results = trainer.test(model=prob_routine, datamodule=datamodule)

# %%
# In this case, we get another metric, the negative log-likelihood (the lower the better) that
# represents how likely it would be to obtain the evaluation set with the parameters of the Laplace
# distribution predicted by the model.
# You will get more information on probabilistic regression in the dedicated tutorial.
