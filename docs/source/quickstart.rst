Quickstart
==========

.. role:: bash(code)
    :language: bash

TorchUncertainty is centered around **uncertainty-aware** training and evaluation routines.
These routines make it very easy to:

- train ensemble-like methods (Deep Ensembles, Packed-Ensembles, MIMO, Masksembles, etc.),
- compute and monitor uncertainty metrics: calibration, out-of-distribution detection, proper scores, grouping loss, etc.,
- leverage post-processing methods automatically during evaluation.

That said, we know there will be as many uses of TorchUncertainty as there are users.
This page outlines how to benefit from TorchUncertainty at every level, from ready-to-train
Lightning-based models all the way down to individual PyTorch layers.

.. figure:: _static/images/structure_torch_uncertainty.jpg
  :alt: TorchUncertainty structure
  :align: center
  :figclass: figure-caption
  :width: 70%

  **Structure of TorchUncertainty**

Training with TorchUncertainty's Uncertainty-aware Routines
-----------------------------------------------------------

TorchUncertainty provides a set of Lightning training and evaluation routines that wrap PyTorch models. Let's have a look at the
`Classification routine <https://github.com/torch-uncertainty/torch-uncertainty/blob/main/torch_uncertainty/routines/classification.py>`_
and its parameters.

.. code:: python

  from lightning.pytorch import LightningModule

  class ClassificationRoutine(LightningModule):
    def __init__(
      self,
      model: nn.Module,
      num_classes: int,
      loss: nn.Module,
      num_estimators: int = 1,
      format_batch_fn: nn.Module | None = None,
      optim_recipe: dict | Optimizer | None = None,
      # ...
      eval_ood: bool = False,
      eval_grouping_loss: bool = False,
      ood_criterion: TUOODCriterion | str = "msp",
      log_plots: bool = False,
      save_to_csv: bool = False,
    ) -> None:
      ...


Building your First Routine
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This routine wraps any custom or TorchUncertainty classification model. To use it, build
your model and pass it to the routine along with an optimization recipe, the loss, and the
number of classes (which is forwarded to ``torchmetrics``).

.. code:: python

  from torch import nn, optim

  model = MyModel(num_classes=10)
  routine = ClassificationRoutine(
    model,
    num_classes=10,
    loss=nn.CrossEntropyLoss(),
    optim_recipe=optim.Adam(model.parameters(), lr=1e-3),
  )


Training with the Routine
^^^^^^^^^^^^^^^^^^^^^^^^^

To train with this routine, you will need to create a Lightning ``Trainer`` and either a Lightning
``DataModule`` or PyTorch ``DataLoader``\ s. When benchmarking models, we recommend using
Lightning datamodules: they automatically handle the train/val/test splits, the
out-of-distribution dataloader and the dataset-shift dataloader. For this example, let us use
TorchUncertainty's CIFAR10 datamodule.

.. code:: python

  from torch_uncertainty.datamodules import CIFAR10DataModule
  from lightning.pytorch import TUTrainer

  dm = CIFAR10DataModule(root="data", batch_size=32)
  trainer = TUTrainer(gpus=1, max_epochs=100)
  trainer.fit(routine, dm)
  trainer.test(routine, dm)

That's it — you have trained your first model with TorchUncertainty! As a result, you get
access to a wide range of metrics measuring the ability of your model to handle uncertainty.
For more examples of training with Lightning ``Trainer``\ s, check the
`Tutorials <auto_tutorials/index.html>`_. For a complete overview of what is logged at
evaluation time, see the `Evaluating Models <evaluation.html>`_ page.

More metrics
^^^^^^^^^^^^

With TorchUncertainty datamodules, you can easily evaluate models on out-of-distribution
datasets by setting ``eval_ood=True``, and on distribution-shifted versions of the test set
by setting ``eval_shift=True``. You can also evaluate the grouping loss with
``eval_grouping_loss=True``. Finally, you can calibrate your model by passing a
``post_processing`` method (e.g., temperature scaling) — in that case the metrics for the
post-processed predictions will be logged under the ``test/post/`` prefix in addition to the
uncalibrated ones.

----

Using the Lightning CLI tool
----------------------------------

Procedure
^^^^^^^^^

The library leverages the `Lightning CLI tool <https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html>`_
to provide a simple way to train and evaluate models, while ensuring reproducibility via
configuration files. Under the ``experiments`` folder, you will find scripts for the four
application tasks covered by the library: classification, regression, segmentation, and
pixel regression. To get the most out of the CLI, check our `CLI Guide <cli_guide.html>`_.

.. note::

  In particular, the ``experiments/classification`` folder contains scripts to reproduce the experiments covered
  in the paper: *Packed-Ensembles for Efficient Uncertainty Estimation*, O. Laurent & A. Lafage, et al., in ICLR 2023.



Example
^^^^^^^

Training a model with the Lightning CLI tool is as simple as running the following command:

.. parsed-literal::

  # in torch-uncertainty/experiments/classification/cifar10
  python resnet.py fit --config configs/resnet18/standard.yaml

Which trains a classic ResNet18 model on CIFAR10 with the settings used in the Packed-Ensembles paper.

----

Using the PyTorch-based models
------------------------------

Procedure
^^^^^^^^^

If you prefer classic PyTorch pipelines, we provide PyTorch ``Module``\ s that do not
require Lightning.

1. Check the *Models* section of the API reference to ensure that your architecture of choice is supported.
2. Instantiate a ``torch.nn.Module`` in your training/testing script using one of the provided builder functions listed in the API reference.

Example
^^^^^^^

You can initialize a Packed-Ensemble out of a ResNet18
backbone with the following code:

.. code:: python

    from torch_uncertainty.models.classification import packed_resnet

    model = packed_resnet(
        in_channels = 3,
        arch=18,
        num_estimators = 4,
        alpha = 2,
        gamma = 2,
        num_classes = 10,
    )

----

Using the PyTorch-based layers
------------------------------

Procedure
^^^^^^^^^

It is likely that your desired architecture is not supported by our library.
In that case, you can directly use the underlying layers.

1. Check the API reference for the specific layers you need.
2. Import the layers and use them as you would any standard PyTorch layer.

If you think that your architecture should be added to the package, raise an
issue on the GitHub repository!

.. tip::

  Do not hesitate to head to the `API Reference <api.html#layers>`_ for more detailed explanations on layer usage.

Example
^^^^^^^

You can create a Packed-Ensemble ``torch.nn.Module`` model with the following
code:

.. code:: python

  from einops import rearrange
  from torch_uncertainty.layers import PackedConv2d, PackedLinear

  class PackedNet(nn.Module):
      def __init__(self) -> None:
          super().__init__()
          M = 4
          alpha = 2
          gamma = 1
          self.conv1 = PackedConv2d(3, 6, 5, alpha=alpha, num_estimators=M, gamma=gamma, first=True)
          self.pool = nn.MaxPool2d(2, 2)
          self.conv2 = PackedConv2d(6, 16, 5, alpha=alpha, num_estimators=M, gamma=gamma)
          self.fc1 = PackedLinear(16 * 5 * 5, 120, alpha=alpha, num_estimators=M, gamma=gamma)
          self.fc2 = PackedLinear(120, 84, alpha=alpha, num_estimators=M, gamma=gamma)
          self.fc3 = PackedLinear(84, 10, alpha=alpha, num_estimators=M, gamma=gamma, last=True)

          self.num_estimators = M

      def forward(self, x):
          x = self.pool(F.relu(self.conv1(x)))
          x = self.pool(F.relu(self.conv2(x)))
          x = rearrange(
              x, "e (m c) h w -> (m e) c h w", m=self.num_estimators
          )
          x = x.flatten(1)
          x = F.relu(self.fc1(x))
          x = F.relu(self.fc2(x))
          x = self.fc3(x)
          return x

  packed_net = PackedNet()

----

Other usage
-----------

Feel free to use any classes described in the API reference such as the metrics, datasets, etc.
