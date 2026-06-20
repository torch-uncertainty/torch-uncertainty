# ruff: noqa: E402, D212, D415, T201
"""
Deep Probabilistic Regression
=============================

This tutorial aims to provide an overview of some utilities in TorchUncertainty for probabilistic regression.
Contrary to pointwise prediction, probabilistic regression consists - in TorchUncertainty's context - in predicting
the parameters of a predefined distribution that fit best some training dataset. The distribution's formulation
is fixed but the parameters are different for all data points, we say that the distribution is heteroscedastic.

Building a MLP for Probabilistic Regression using TorchUncertainty Distribution Layers
--------------------------------------------------------------------------------------

In this section we cover the building of a very simple MLP outputting Normal distribution parameters,
the mean and the standard deviation. These values will depend on the data point given as input.

1. Loading the Utilities
~~~~~~~~~~~~~~~~~~~~~~~~

First, we disable some logging and warnings to keep the output clean.
"""

# %%
import logging
import os
import warnings

import torch
from torch import nn

logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

# Here are the trainer and dataloader main hyperparameters
MAX_EPOCHS = 10
BATCH_SIZE = 128

# %%
# 2. Building the NormalMLP Model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# To create a NormalMLP model estimating a Normal distribution, we use the NormalLinear layer.
# This layer is a wrapper around the nn.Linear layer, which outputs the location and scale of a Normal distribution in a dictionnary.
# As you will see in the following, any other distribution layer from TU can be used in the same way. Check out the regression tutorial
# to learn how to create a NormalMLP more easily using the blueprints from torch_uncertainty.models
from torch_uncertainty.layers.distributions import NormalLinear


class NormalMLP(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, 50)
        self.fc2 = NormalLinear(
            base_layer=nn.Linear,
            event_dim=out_features,
            in_features=50,
        )

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# %%
# 3. Setting up the Data
# ~~~~~~~~~~~~~~~~~~~~~~
#
# We use the UCI Kin8nm dataset, which is a regression dataset with 8 features and 8192 samples.
from torch_uncertainty.datamodules.tabular_regression import Kin8NMDataModule

# datamodule
datamodule = Kin8NMDataModule(
    root=os.environ.get("TU_DATA_DIR", "data"), batch_size=BATCH_SIZE, num_workers=4
)

# %%
# 4. Setting up the Model and Trainer
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from torch_uncertainty import TUTrainer

trainer = TUTrainer(
    accelerator="gpu",
    devices=1,
    max_epochs=MAX_EPOCHS,
    enable_progress_bar=False,
)

model = NormalMLP(in_features=8, out_features=1)


# %%
# 5. The Loss, the Optimizer and the Training Routine
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We use the DistributionNLLLoss to compute the negative log-likelihood of the Normal distribution.
# Note that this loss can be used with any Distribution from torch.distributions.
# For the optimizer, we use the Adam optimizer with a learning rate of 5e-2.
# Finally, we create a RegressionRoutine to train the model.
# We indicate that the output dimension is 1 and the distribution family is "normal".

from torch_uncertainty.losses import DistributionNLLLoss
from torch_uncertainty.routines import RegressionRoutine

loss = DistributionNLLLoss()

routine = RegressionRoutine(
    output_dim=1,
    model=model,
    loss=loss,
    optim_recipe=torch.optim.Adam(model.parameters(), lr=5e-2),
    dist_family="normal",
)


# %%
# 6. Training and Testing the Model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Thanks to the RegressionRoutine, we get the values from 4 metrics, the mean absolute error,
# the mean squared error, its square root (RMSE) and the negative-log-likelihood (NLL). For all these metrics,
# lower is better.


trainer.fit(model=routine, datamodule=datamodule)
results = trainer.test(model=routine, datamodule=datamodule)

# %%
# 7. Benchmarking Different Distributions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Our NormalMLP model assumes a Normal distribution as the output. However, we could be interested in comparing the performance of different distributions.
# TorchUncertainty provides a simple way to do this using the get_dist_linear_layer() function.
# Let us rewrite the NormalMLP model to use it.

from torch_uncertainty.layers.distributions import get_dist_linear_layer


class DistMLP(nn.Module):
    def __init__(self, in_features: int, out_features: int, dist_family: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, 50)
        dist_layer = get_dist_linear_layer(dist_family)
        self.fc2 = dist_layer(
            base_layer=nn.Linear,
            event_dim=out_features,
            in_features=50,
        )

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# %%
# We can now train the model with different distributions.
# Let us train the model with a Laplace, Student's t, and Cauchy distribution.
# Note that we use the mode as the point-wise estimate of the distribution as the mean
# is not defined for the Cauchy distribution.
for dist_family in ["laplace", "student", "cauchy"]:
    print("#" * 38)
    print(f">>> Training with {dist_family.capitalize()} distribution")
    print("#" * 38)
    trainer = TUTrainer(
        accelerator="gpu",
        devices=1,
        max_epochs=MAX_EPOCHS,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    model = DistMLP(in_features=8, out_features=1, dist_family=dist_family)
    routine = RegressionRoutine(
        output_dim=1,
        model=model,
        loss=loss,
        optim_recipe=torch.optim.Adam(model.parameters(), lr=5e-2),
        dist_family=dist_family,
        dist_estimate="mode",
    )
    trainer.fit(model=routine, datamodule=datamodule)
    trainer.test(model=routine, datamodule=datamodule)
# %%
# The Negative Log-Likelihood (NLL) is a good score to encompass the correctness of the predicted
# distributions, evaluating both the correctness of the mode (the point prediction) and of the predicted uncertainty
# around the mode ("represented" by the variance). Although there is a lot of variability, in this case, it seems that
# the Normal distribution often performs better.
