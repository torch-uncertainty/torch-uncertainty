# ruff: noqa: D212, D415
"""
Mixup and MixupMP Training & Ensembles with TorchUncertainty
============================================================

*This tutorial illustrates how to train models using Mixup and MixupMP*
*in the same style as the Packed-Ensembles tutorial you provided.*

In this notebook we will show:
1. Standard Mixup training
2. Assembling Mixup-trained models into a Deep Ensemble
3. MixupMP training
4. Assembling MixupMP models the same way

Throughout this notebook we use `PyTorch Lightning <https://lightning.ai/docs/pytorch/stable/>`_
and `TorchUncertainty <https://torch-uncertainty.github.io/>`_ for training.

Note: This script does *not* actually run training — it shows the configuration and conceptual usage only.
"""

# %%
# 1. Training with Mixup
# ~~~~~~~~~~~~~~~~~~~~~~
#
# To train a Mixup model using TorchUncertainty with Lightning,
# you need to configure the routine's `mixup_params`:
#
# - `mixtype: "mixup"` tells the routine to use standard Mixup augmentation.
# - `mixup_alpha` controls the strength of mixing (Beta(α,α)).
#
# Mixup internally creates convex combinations of images and labels
# and computes the soft-label cross entropy under the hood.
#
# This config snippet focuses on *only* the mixup setup.
# Refer to your standard training setup for other keys.

# Example Lightning CLI config for Mixup training:
mixup_config = r"""
# lightning.pytorch==2.1.3
seed_everything: false
eval_after_fit: true

trainer:
  accelerator: gpu
  devices: 1
  precision: 16-mixed
  max_epochs: 200
  logger:
    class_path: lightning.pytorch.loggers.TensorBoardLogger
    init_args:
      save_dir: logs/wideresnet28x10
      name: mixup
      default_hp_metric: false
  callbacks:
    - class_path: torch_uncertainty.callbacks.TUClsCheckpoint
    - class_path: lightning.pytorch.callbacks.LearningRateMonitor
      init_args:
        logging_interval: step
    - class_path: lightning.pytorch.callbacks.EarlyStopping
      init_args:
        monitor: val/cls/Acc
        patience: 1000
        check_finite: true

routine:
  model:
    class_path: torch_uncertainty.models.classification.wideresnet28x10
    init_args:
      in_channels: 3
      num_classes: 10
      dropout_rate: 0.0
      style: cifar
  num_classes: 10
  loss: CrossEntropyLoss

  # Mixup-specific parameters
  mixup_params:
    mixtype: "mixup"
    mixup_alpha: 2

data:
  root: ./data
  batch_size: 128
  num_workers: 4

optimizer:
  class_path: torch.optim.SGD
  init_args:
    lr: 0.1
    momentum: 0.9
    weight_decay: 5e-4
    nesterov: true

lr_scheduler:
  class_path: torch.optim.lr_scheduler.MultiStepLR
  init_args:
    milestones:
      - 60
      - 120
      - 160
    gamma: 0.2
"""

# %%
# 2. How to Ensemble Mixup Models
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Once you have trained multiple Mixup models (e.g., using the previous config
# under different versions / runs), you can assemble them with `deep_ensembles`.
# This is identical to ensembling vanilla models, except the ckpt paths point
# to your Mixup-trained checkpoints.
#
# TorchUncertainty's `deep_ensembles` helper will load each checkpoint
# and produce an ensemble model that averages predictions across all members.:contentReference[oaicite:0]{index=0}

ensemble_mixup_config = r"""
# lightning.pytorch==2.1.3
seed_everything: false
eval_after_fit: true

trainer:
  accelerator: gpu
  devices: 1
  precision: 16-mixed
  max_epochs: 200
  logger:
    class_path: lightning.pytorch.loggers.TensorBoardLogger
    init_args:
      save_dir: logs/wideresnet28x10
      name: mixup_ensemble
      default_hp_metric: false
  callbacks:
    - class_path: torch_uncertainty.callbacks.TUClsCheckpoint
    - class_path: lightning.pytorch.callbacks.LearningRateMonitor
      init_args:
        logging_interval: step
    - class_path: lightning.pytorch.callbacks.EarlyStopping
      init_args:
        monitor: val/cls/Acc
        patience: 1000
        check_finite: true

routine:
  model:
    class_path: torch_uncertainty.models.deep_ensembles
    init_args:
      core_models:
        class_path: torch_uncertainty.models.classification.wideresnet28x10
        init_args:
          in_channels: 3
          num_classes: 10
          style: cifar
          dropout_rate: 0.0
      num_estimators: 4
      task: classification
      # Replace with your trained mixup Checkpoint paths
      ckpt_paths:
        - path/to/mixup/version_0.ckpt
        - path/to/mixup/version_1.ckpt
        - path/to/mixup/version_2.ckpt
        - path/to/mixup/version_3.ckpt

  num_classes: 10
  is_ensemble: true
  format_batch_fn:
    class_path: torch_uncertainty.transforms.RepeatTarget
    init_args:
      num_repeats: 4

data:
  root: ./data
  batch_size: 128
"""

# %%
# 3. Training with MixupMP
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# MixupMP is a variant of mixup that produces posterior samples from a predictive
# distribution that is more realistic than deep ensembles alone, according to the
# `MixupMP` paper (Martingale posterior based construction).
#
# The two key differences vs standard Mixup:
#   - You use a specialized loss: `MixupMPLoss`
#   - You set `mixtype: "mixupmp"` in mixup_params
#
# This causes the routine to sample augmented predictive variations as part of
# the MixupMP methodology for uncertainty.

mixupmp_config = r"""
# lightning.pytorch==2.1.3
seed_everything: false
eval_after_fit: true

trainer:
  accelerator: gpu
  devices: 1
  precision: 16-mixed
  max_epochs: 200
  logger:
    class_path: lightning.pytorch.loggers.TensorBoardLogger
    init_args:
      save_dir: logs/wideresnet28x10
      name: mixupmp
      default_hp_metric: false
  callbacks:
    - class_path: torch_uncertainty.callbacks.TUClsCheckpoint
    - class_path: lightning.pytorch.callbacks.LearningRateMonitor
      init_args:
        logging_interval: step
    - class_path: lightning.pytorch.callbacks.EarlyStopping
      init_args:
        monitor: val/cls/Acc
        patience: 1000
        check_finite: true

routine:
  model:
    class_path: torch_uncertainty.models.classification.wideresnet28x10
    init_args:
      in_channels: 3
      num_classes: 10
      style: cifar
      dropout_rate: 0.0

  num_classes: 10

  # Use MixupMP-specific loss with default parameters
  loss: torch_uncertainty.losses.MixupMPLoss

  mixup_params:
    mixtype: "mixupmp"
    mixup_alpha: 2

data:
  root: ./data
  batch_size: 128
  num_workers: 4

optimizer:
  class_path: torch.optim.SGD
  init_args:
    lr: 0.1
    momentum: 0.9
    weight_decay: 5e-4
    nesterov: true

lr_scheduler:
  class_path: torch.optim.lr_scheduler.MultiStepLR
  init_args:
    milestones:
      - 60
      - 120
      - 160
    gamma: 0.2
"""

# %%
# 4. How to Ensemble MixupMP Models
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Ensembling MixupMP models is conceptually the same as ensembling
# any other model: train N different runs (with different seeds / settings),
# then point their ckpt_paths to the Deep Ensembles config.
#
# The only difference is that each individual model is a MixupMP-trained
# checkpoint instead of standard training or mixup training.
#
# Use the exact same deep_ensembles format as shown above (Section 2),
# but give it the paths to your MixupMP checkpoints.
#
# 5. References
# ~~~~~~~~~~~~~
#
# For more information on Mixup Ensembles, we refer to the following resources:
#
# - Combining ensembles and data augmentation can harm your calibration
#   `ICLR 2021 <https://arxiv.org/pdf/2010.09875>`_
#   (Yeming Wen, Ghassen Jerfel, Rafael Muller, Michael W. Dusenberry,
#   Jasper Snoek, Balaji Lakshminarayanan, and Dustin Tran)
#
# - Uncertainty quantification and deep ensembles
#   `NeurIPS 2021 <https://arxiv.org/pdf/2007.08792>`_
#   (Rahul Rahaman & Alexandre H. Thiery)
#
# For more information on MixupMP, we refer to the following resource:
#
# - Posterior Uncertainty Quantification in Neural Networks using Data Augmentation
#   `AISTATS 2024 <https://arxiv.org/abs/2403.12729>`_
#   (Luhuan Wu & Sinead Williamson)
