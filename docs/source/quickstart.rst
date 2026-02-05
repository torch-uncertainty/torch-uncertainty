Quickstart
==========

.. role:: bash(code)
    :language: bash

TorchUncertainty is centered around **uncertainty-aware** training and evaluation routines.
These routines make it very easy to:

- train ensembles-like methods (Deep Ensembles, Packed-Ensembles, MIMO, Masksembles, etc)
- compute and monitor uncertainty metrics: calibration, out-of-distribution detection, proper scores, grouping loss, etc.
- leverage post-processing methods automatically during evaluation

Yet, we take account that their will be as many different uses of TorchUncertainty as there are of users.
This page provides ideas on how to benefit from TorchUncertainty at all levels: from ready-to-train lightning-based models to using only specific
PyTorch layers.

.. figure:: _static/images/structure_torch_uncertainty.jpg
  :alt: TorchUncertainty structure
  :align: center
  :figclass: figure-caption
  :width: 70%

  **Structure of TorchUncertainty**

Training with TorchUncertainty's Uncertainty-aware Routines
-----------------------------------------------------------

TorchUncertainty provides a set of Ligthning training and evaluation routines that wrap PyTorch models. Let's have a look at the
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
      save_in_csv: bool = False,
    ) -> None:
      ...


Building your First Routine
^^^^^^^^^^^^^^^^^^^^^^^^^^^
This routine is a wrapper of any custom or TorchUncertainty classification model. To use it,
just build your model and pass it to the routine as argument along with an optimization recipe
and the loss as well as the number of classes that we use for torch metrics.

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

To train with this routine, you will first need to create a lightning Trainer and have either a lightning datamodule
or PyTorch dataloaders. When benchmarking models, we advise to use lightning datamodules that will automatically handle
train/val/test splits, out-of-distribution detection and dataset shift. For this example, let us use TorchUncertainty's
CIFAR10 datamodule.

.. code:: python

  from torch_uncertainty.datamodules import CIFAR10DataModule
  from lightning.pytorch import TUTrainer

  dm = CIFAR10DataModule(root="data", batch_size=32)
  trainer = TUTrainer(gpus=1, max_epochs=100)
  trainer.fit(routine, dm)
  trainer.test(routine, dm)

Here it is, you have trained your first model with TorchUncertainty! As a result, you will get access to various metrics
measuring the ability of your model to handle uncertainty. You can get other examples of training with lightning Trainers
looking at the `Tutorials <auto_tutorials/index.html>`_.

More metrics
^^^^^^^^^^^^

With TorchUncertainty datamodules, you can easily test models on out-of-distribution datasets, by
setting the ``eval_ood`` parameter to ``True``. You can also evaluate the grouping loss by setting ``eval_grouping_loss`` to ``True``.
Finally, you can calibrate your model using the ``calibration_set`` parameter. In this case, you will get
metrics for but the uncalibrated and calibrated models: the metrics corresponding to the temperature scaled
model will begin with ``ts_``.

----

Using the Lightning CLI tool
----------------------------------

Procedure
^^^^^^^^^

The library leverages the `Lightning CLI tool <https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html>`_
to provide a simple way to train models and evaluate them, while insuring reproducibility via configuration files.
Under the ``experiment`` folder, you will find scripts for the three application tasks covered by the library:
classification, regression and segmentation. Take the most out of the CLI by checking our `CLI Guide <cli_guide.html>`_.

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

If you prefer classic PyTorch pipelines, we provide PyTorch Modules that do not
require Lightning.

1. Check the API reference under the *Models* section to ensure the architecture of your choice is supported by the library.
2. Create a ``torch.nn.Module`` in your training/testing script using one of the provided building functions listed in the API reference.

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
In that case, you might be interested in directly using the actual layers.

1. Check the API reference for specific layers of your choosing.
2. Import the layers and use them as you would for any standard PyTorch layer.

If you think that your architecture should be added to the package, raise an
issue on the GitHub repository!

.. tip::

  Do not hesitate to go to the `API Reference <api.html#layers>`_ to get better explanations on the
  layer usage.

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
