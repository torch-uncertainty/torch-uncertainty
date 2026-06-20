# ruff: noqa: E402, D212, D415, T201
"""
Training a Bayesian Neural Network in 20 seconds
================================================

In this tutorial, we will train a variational inference Bayesian Neural Network (viBNN) LeNet classifier on the MNIST dataset.

Foreword on Bayesian Neural Networks
------------------------------------

Bayesian Neural Networks (BNNs) are a class of neural networks that estimate the uncertainty on their predictions via uncertainty
on their weights. This is achieved by considering the weights of the neural network as random variables, and by learning their
posterior distribution. This is in contrast to standard neural networks, which only learn a single set of weights (this can be
seen as Dirac distributions on the weights).

For more information on Bayesian Neural Networks, we refer to the following resources:

- Weight Uncertainty in Neural Networks `ICML2015 <https://arxiv.org/pdf/1505.05424.pdf>`_
- Hands-on Bayesian Neural Networks - a Tutorial for Deep Learning Users `IEEE Computational Intelligence Magazine <https://arxiv.org/pdf/2007.06823.pdf>`_

Training a Bayesian LeNet using TorchUncertainty models and Lightning
---------------------------------------------------------------------

In this first part, we train a Bayesian LeNet, based on the model and routines already implemented in TU.

1. Loading the utilities
~~~~~~~~~~~~~~~~~~~~~~~~

To train a BNN using TorchUncertainty, we have to load the following modules:

- our TUTrainer to improve the display of our metrics
- the model: bayesian_lenet, which lies in the torch_uncertainty.model.classification.lenet module
- the classification training routine from torch_uncertainty.routines module
- the Bayesian objective: the ELBOLoss, which lies in the torch_uncertainty.losses file
- the datamodule that handles dataloaders: MNISTDataModule from torch_uncertainty.datamodules

We will also need to define an optimizer using torch.optim and Pytorch's
neural network utils from torch.nn.
"""

# %%
import os
from pathlib import Path

from torch import nn, optim

from torch_uncertainty import TUTrainer
from torch_uncertainty.datamodules import MNISTDataModule
from torch_uncertainty.losses import ELBOLoss
from torch_uncertainty.models.classification.lenet import bayesian_lenet
from torch_uncertainty.routines import ClassificationRoutine

# We also define the main hyperparameters, with just one epoch for the sake of time
BATCH_SIZE = 512
MAX_EPOCHS = 2

# %%
# 2. Creating the necessary variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In the following, we instantiate our trainer, define the root of the datasets and the logs.
# We also create the datamodule that handles the MNIST dataset, dataloaders and transforms.
# Please note that the datamodules can also handle OOD detection by setting the `eval_ood`
# parameter to True, as well as distribution shift with `eval_shift`.
# Finally, we create the model using the blueprint from torch_uncertainty.models.

trainer = TUTrainer(accelerator="gpu", devices=1, enable_progress_bar=False, max_epochs=MAX_EPOCHS)

# datamodule
root = Path(os.environ.get("TU_DATA_DIR", "data"))
datamodule = MNISTDataModule(root=root, batch_size=BATCH_SIZE, num_workers=8)

# model
model = bayesian_lenet(datamodule.num_channels, datamodule.num_classes)

# %%
# 3. The Loss and the Training Routine
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Then, we just define the loss to be used during training, which is a bit special and called
# the evidence lower bound. We use the hyperparameters proposed in the blitz
# library. As we are training a classification model, we use the CrossEntropyLoss
# as the negative log likelihood. We then define the training routine using the classification
# training routine from torch_uncertainty.classification. We provide the model, the ELBO
# loss and the optimizer to the routine.
# We use an Adam optimizer with a learning rate of 0.02.

loss = ELBOLoss(
    model=model,
    inner_loss=nn.CrossEntropyLoss(),
    kl_weight=1 / 10000,
    num_samples=3,
)

routine = ClassificationRoutine(
    model=model,
    num_classes=datamodule.num_classes,
    loss=loss,
    optim_recipe=optim.Adam(model.parameters(), lr=2e-2),
    is_ensemble=True,
)

# %%
# 4. Gathering Everything and Training the Model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Now that we have prepared all of this, we just have to gather everything in
# the main function and to train the model using our wrapper of Lightning Trainer.
# Specifically, it needs the routine, that includes the model as well as the
# training/eval logic and the datamodule.
# The dataset will be downloaded automatically in the root/data folder, and the
# logs will be saved in the root/logs folder.

trainer.fit(model=routine, datamodule=datamodule)
trainer.test(model=routine, datamodule=datamodule)
# %%
# 5. Testing the Model
# ~~~~~~~~~~~~~~~~~~~~
#
# Now that the model is trained, let's test it on MNIST.
# Please note that we apply a reshape to the logits to determine the dimension corresponding to the ensemble
# and to the batch. As for TorchUncertainty 0.5.2, the ensemble dimension is merged with the batch dimension
# in this order (num_estimator x batch, classes).

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from einops import rearrange


def imshow(img) -> None:
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")
    plt.tight_layout()
    plt.show()


images, labels = next(iter(datamodule.val_dataloader()))

# print images
imshow(torchvision.utils.make_grid(images[:4, ...]))
print("Ground truth: ", " ".join(f"{labels[j]}" for j in range(4)))

# Put the model in eval mode to use several samples
model = routine.eval()
logits = routine(images[:4, ...])
print("Output logit shape (Num predictions x Batch) x Classes: ", logits.shape)
logits = rearrange(logits, "(m b) c -> b m c", b=4)  # batch_size, num_estimators, num_classes

# We apply the softmax on the classes then average over the estimators
probs = torch.nn.functional.softmax(logits, dim=-1)
avg_probs = probs.mean(dim=1)
var_probs = probs.std(dim=1)

predicted = torch.argmax(avg_probs, -1)

print("Predicted digits: ", " ".join(f"{predicted[j]}" for j in range(4)))
print(
    "Std. dev. of the scores over the posterior samples",
    " ".join(f"{var_probs[j][predicted[j]]:.3f}" for j in range(4)),
)
# %%
# Here, we show the variance of the top prediction. This is a non-standard but intuitive way to show the diversity of the predictions
# of the ensemble. Ideally, the variance should be high when the prediction is incorrect.
#
# References
# ----------
#
# - **LeNet & MNIST:** LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. `Proceedings of the IEEE <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_.
# - **Bayesian Neural Networks:** Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015). Weight Uncertainty in Neural Networks. `ICML 2015 <https://arxiv.org/pdf/1505.05424.pdf>`_.
# - **The Adam optimizer:** Kingma, D. P., & Ba, J. (2014). "Adam: A method for stochastic optimization." `ICLR 2015 <https://arxiv.org/pdf/1412.6980.pdf>`_.
# - **The Blitz** `library <https://github.com/piEsposito/blitz-bayesian-deep-learning>`_ (for the hyperparameters).
