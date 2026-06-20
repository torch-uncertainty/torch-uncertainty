CLI Guide
=========

Introduction to the Lightning CLI
---------------------------------

The Lightning CLI tool simplifies the implementation of a CLI that instantiates models and trains
or evaluates them on data. The CLI is a wrapper around the ``Trainer`` class and provides a set of
subcommands to train and test a ``LightningModule`` on a ``LightningDataModule``. To better match
our needs, we subclass ``LightningCLI`` into ``TULightningCLI``.

.. note::
    ``TULightningCLI`` adds (among others) a new argument to ``LightningCLI``: :attr:`eval_after_fit`, which
    controls whether an evaluation on the test set is performed after the training phase. We
    recommend restricting the use of this parameter to single-node training to avoid small
    test-performance inconsistencies.

Let's see how to implement the CLI, by checking out the ``experiments/classification/cifar10/main.py``.

.. code:: python

    import torch
    from torch_uncertainty import TULightningCLI
    from torch_uncertainty.datamodules import CIFAR10DataModule
    from torch_uncertainty.routines import ClassificationRoutine


    def cli_main() -> TULightningCLI:
        return TULightningCLI(ClassificationRoutine, CIFAR10DataModule)


    if __name__ == "__main__":
        torch.set_float32_matmul_precision("medium")
        cli = cli_main()
        if (
            (not cli.trainer.fast_dev_run)
            and cli.subcommand == "fit"
            and cli._get(cli.config, "eval_after_fit")
        ):
            cli.trainer.test(datamodule=cli.datamodule, ckpt_path="best")

This file enables both training and testing on the CIFAR-10 dataset. The model, optimizer and
learning-rate scheduler are set in the configuration file as shown below.

.. code:: python

    def cli_main() -> TULightningCLI:
        return TULightningCLI(ClassificationRoutine, CIFAR10DataModule)

Depending on the CLI subcommand, calling ``cli_main()`` will either train or test the model on the
CIFAR-10 dataset. But what are these subcommands?

.. parsed-literal::

    python main.py --help

This command will display the available subcommands of the CLI tool.

.. parsed-literal::

    subcommands:
    For more details of each subcommand, add it as an argument followed by --help.

    Available subcommands:
        fit                 Runs the full optimization routine.
        validate            Perform one evaluation epoch over the validation set.
        test                Perform one evaluation epoch over the test set.
        predict             Run evaluation on your data.

You can run any of these subcommands and set up all your hyperparameters directly on the command line.

Given the large number of hyperparameters, we recommend using configuration files instead. Let's see how.


Configuration files
-------------------

By default, ``LightningCLI`` supports configuration files in the YAML format (learn more about this format
`here <https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_faq.html#what-is-a-yaml-config-file>`_).
Taking the previous example, we can create a configuration file named ``config.yaml`` with the following content:

.. code:: yaml

    # config.yaml
    eval_after_fit: true
    trainer:
      max_epochs: 75
      accelerators: gpu
      devices: 1
    routine:
      model:
        class_path: torch_uncertainty.models.classification.resnet
        init_args:
        in_channels: 3
        num_classes: 10
        arch: 18
        style: cifar
      num_classes: 10
      loss: CrossEntropyLoss
    data:
      root: ./data
      batch_size: 128
  optimizer:
    class_path: torch.optim.SGD
    init_args:
      lr: 0.05
      momentum: 0.9
      weight_decay: 5e-4
  lr_scheduler:
    class_path: torch.optim.lr_scheduler.MultiStepLR
    init_args:
      milestones:
        - 25
        - 50
      gamma: 0.1

Then, we can run the following command to train the model:

.. parsed-literal::

    python main.py fit --config config.yaml

By default, executing the command above stores the experiment results in a directory named ``lightning_logs``,
and the last state of the model is saved in ``lightning_logs/version_{int}/checkpoints``.
In addition, all arguments passed to instantiate the ``Trainer``, ``CIFAR10DataModule``,
``torch.optim.SGD``, and ``torch.optim.lr_scheduler.MultiStepLR`` classes are saved in
``lightning_logs/version_{int}/config.yaml``. When testing the model, we recommend reusing this
configuration file to ensure that the same hyperparameters are used for training and testing.

.. parsed-literal::

    python main.py test --config lightning_logs/version_{int}/config.yaml --ckpt_path lightning_logs/version_{int}/checkpoints/{filename}.ckpt

Experiment folder usage
-----------------------

Now that we have seen how to implement the CLI and how to use configuration files, let's explore
the configurations available in the ``experiments`` directory, which is organized roughly as follows:

.. code:: bash

    experiments
    ├── classification
    │   ├── cifar10
    │   │   ├── configs
    │   │   │   ├── resnet18
    │   │   │   │   ├── standard.yaml
    │   │   │   │   ├── packed.yaml
    │   │   │   │   ├── ...
    │   │   │   ├── resnet50
    │   │   │   ├── wideresnet28x10
    │   │   ├── main.py
    │   └── cifar100
    │       ├── configs
    │       │   ├── ...
    │       ├── main.py
    ├── regression
    │   └── uci_datasets
    │       ├── configs
    │       └── main.py
    └── segmentation
        ├── cityscapes
        │   ├── configs
        │   └── main.py
        └── muad
            ├── configs
            └── main.py

For each task (**classification**, **regression**, and **segmentation**), we have a directory containing the datasets
(e.g., CIFAR10, CIFAR100, UCI datasets, Cityscapes, and MUAD) and for each dataset, we have a directory containing
the configuration files and the CLI files for different backbones.

You can directly use the CLI files with the command line or use the predefined configuration files to train and test
the models. The configuration files are stored in the ``configs``. For example, the configuration file for the classic
ResNet-18 model on the CIFAR-10 dataset is stored in the ``experiments/classification/cifar10/configs/resnet18/standard.yaml``
file. For the Packed ResNet-18 model on the CIFAR-10 dataset, the configuration file is stored in the
``experiments/classification/cifar10/configs/resnet18/packed.yaml`` file.

If you are interested in using a ResNet model but want to choose some of the hyperparameters using the command line,
you can use the configuration file and override the hyperparameters using the command line. For example, to train
a ResNet-18 model on the CIFAR-10 dataset with a batch size of :math:`256`, you can use the following command:

.. parsed-literal::

    python main.py fit --config configs/resnet18/standard.yaml --data.batch_size 256

To use the weights argument of the ``torch.nn.CrossEntropyLoss`` class, you can use the following command:

.. parsed-literal::

    python main.py fit --config configs/resnet18/standard.yaml --routine.loss CrossEntropyLoss --routine.loss.weight Tensor --routine.loss.weight.dict_kwargs.data [1,2,3,4,5,6,7,8,9,10]


In addition, we provide a default configuration file for some backbones in the ``configs`` directory. For example,
``experiments/classification/cifar10/configs/resnet18/standard.yaml`` contains the default hyperparameters to train a ResNet-18 model
on the CIFAR-10 dataset. Yet, some hyperparameters are purposely missing to be set by the user using the command line.

For instance, to train a Packed ResNet-34 model on the CIFAR-10 dataset with :math:`4` estimators and a :math:`\alpha` value of :math:`2`,
you can use the following command:

.. parsed-literal::

    python main.py fit --config configs/resnet18/standard.yaml --trainer.max_epochs 75 --routine.model.version packed --routine.model.arch 34 --routine.model.num_estimators 4 --routine.model.alpha 2 --optimizer.lr 0.05 --lr_scheduler.milestones [25,50]


.. tip::

    Explore the `Lightning CLI docs <https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html>`_ to learn more about the CLI tool,
    the available arguments, and how to use them with configuration files.
