# ruff: noqa: E402, D212, D415, T201
"""
From a Standard Classifier to a Packed-Ensemble
===============================================

This tutorial is heavily inspired by PyTorch's `Training a Classifier <https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#test-the-network-on-the-test-data>`_
tutorial.

Let's dive step by step into the process to modify a standard classifier into a
packed-ensemble classifier.

Dataset
-------

In this tutorial we will use the CIFAR10 dataset available in the torchvision
package. The CIFAR10 dataset consists of 60,000 32x32 colour images in 10
classes, with 6000 images per class. There are 50000 training images and 10000
test images.

Here is an example of what the data looks like:

.. figure:: /_static/img/cifar10.png
   :alt: cifar10
   :figclass: figure-caption

   Sample of the CIFAR-10 dataset

Training an image Packed-Ensemble classifier
--------------------------------------------

Here is the outline of the process:

1. Load and normalizing the CIFAR10 training and test datasets using
   ``torchvision``
2. Define a Packed-Ensemble from a standard classifier
3. Define a loss function
4. Train the Packed-Ensemble on the training data
5. Test the Packed-Ensemble on the test data and evaluate its performance
   w.r.t. uncertainty quantification and OOD detection

1. Load and normalize CIFAR10
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""

# %%
import os

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# %%
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

# %%
# .. note::
#     If running on Windows and you get a BrokenPipeError, try setting
#     the num_worker of torch.utils.data.DataLoader() to 0.

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

MAX_EPOCHS = 3
BATCH_SIZE = 256

trainset = torchvision.datasets.CIFAR10(
    root=os.environ.get("TU_DATA_DIR", "data"), train=True, download=True, transform=transform
)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(
    root=os.environ.get("TU_DATA_DIR", "data"), train=False, download=True, transform=transform
)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# %%
# Let us show some of the training images.

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img) -> None:
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.figure(figsize=(10, 3))
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images[:4], pad_value=1))
# print labels
print(" ".join(f"{classes[labels[j]]:5s}" for j in range(4)))


# %%
# 2. Define a Packed-Ensemble from a standard classifier
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# First we define a standard classifier for CIFAR10 for reference. We will use a
# convolutional neural network.

import torch.nn.functional as F
from torch import nn


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


net = Net()

# %%
# Let's modify the standard classifier into a Packed-Ensemble classifier of
# parameters :math:`M=4,\ \alpha=2\text{ and }\gamma=1`.

from einops import rearrange

from torch_uncertainty.layers import PackedConv2d, PackedLinear


class PackedNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        num_estimators = 4
        alpha = 2
        gamma = 1
        self.conv1 = PackedConv2d(
            3, 6, 5, alpha=alpha, num_estimators=num_estimators, gamma=gamma, first=True
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = PackedConv2d(6, 16, 5, alpha=alpha, num_estimators=num_estimators, gamma=gamma)
        self.fc1 = PackedLinear(
            16 * 5 * 5, 120, alpha=alpha, num_estimators=num_estimators, gamma=gamma
        )
        self.fc2 = PackedLinear(120, 84, alpha=alpha, num_estimators=num_estimators, gamma=gamma)
        self.fc3 = PackedLinear(
            84,
            10 * num_estimators,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            last=True,
        )

        self.num_estimators = num_estimators

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


packed_net = PackedNet()

# %%
# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum.

from torch import optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(packed_net.parameters(), lr=0.2, momentum=0.9)

# %%
# 4. Train the Packed-Ensemble on the training data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's train the Packed-Ensemble on the training data.

for epoch in range(MAX_EPOCHS):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = packed_net(inputs)
        loss = criterion(outputs, labels.repeat(packed_net.num_estimators))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:  # print every 20 mini-batches
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}")
            running_loss = 0.0

print("Finished Training")

# %%
# Save our trained model:

PATH = "./cifar_packed_net.pth"
torch.save(packed_net.state_dict(), PATH)

# %%
# 5. Test the Packed-Ensemble on the test data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let us display an image from the test set to get familiar.

dataiter = iter(testloader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images[:6], pad_value=1))
print(
    "GroundTruth: ",
    " ".join(f"{classes[labels[j]]:5s}" for j in range(6)),
)

# %%
# Next, let us load back in our saved model (note: saving and re-loading the
# model wasn't necessary here, we only did it to illustrate how to do so):

packed_net = PackedNet()
packed_net.load_state_dict(torch.load(PATH, weights_only=True))
# %%
# Let us see what the Packed-Ensemble predicts these examples above are:

logits = packed_net(images[:6])
logits = rearrange(logits, "(m b) c -> b m c", m=packed_net.num_estimators)
probs_per_est = F.softmax(logits, dim=-1)
outputs = probs_per_est.mean(dim=1)

_, predicted = torch.max(outputs, 1)

print(
    "Predicted: ",
    " ".join(f"{classes[predicted[j]]:5s}" for j in range(6)),
)

# %%
# The results seem pretty good.
