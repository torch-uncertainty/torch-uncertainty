# ruff: noqa: E402, D212, D415, T201
"""
DEUP: Direct Epistemic Uncertainty Prediction with TorchUncertainty
====================================================================

DEUP estimates the *epistemic* component of uncertainty by training a lightweight
error-predictor ``g`` on out-of-fold generalization errors collected from a held-out
calibration set (Algorithm 2 in Lahlou et al. 2023). Once fitted, ``g(x)`` returns
a non-negative score: higher means "the base model is more likely to be wrong here."

This tutorial has two parts:

1. **Synthetic walkthrough** - illustrates the DEUP API on random tabular data.
2. **CIFAR-10 + ClassificationRoutine** - integrates DEUP with a pretrained ResNet-18
   for OOD detection against SVHN, the standard CIFAR-10 OOD benchmark.

How DEUP works:
~~~~~~~~~~~~~~~

1. Run the (already-trained) base model on a held-out calibration set and compute
   per-sample errors (cross-entropy for classification, squared error for regression).
2. Perform K-fold cross-validation *on those calibration errors* to train K lightweight
   error-predictor MLPs.  The out-of-fold predictions become the targets for the final
   predictor, acting as generalization-error proxies.
3. Train the final error predictor ``g`` on all calibration features to predict those
   OOF targets.  At inference, ``g(x) ≥ 0`` is the epistemic uncertainty estimate.

Reference:
    Lahlou et al. (2023). *DEUP: Direct Epistemic Uncertainty Prediction.*
    TMLR. https://openreview.net/forum?id=eGLdVRvvfQ
"""

# %%
# 1. Imports
# ~~~~~~~~~~

import os

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from torch_uncertainty.post_processing import DEUP

# %%
# 2. Synthetic classification task
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We create a small random dataset to illustrate the DEUP API without any
# expensive training.  In practice the base model should be *pre-trained*;
# here we use random weights just to show the interface.

torch.manual_seed(0)
n_cal, in_dim, n_classes = 100, 8, 5

x_cal = torch.randn(n_cal, in_dim)
y_cal = torch.randint(0, n_classes, (n_cal,))
cal_loader = DataLoader(TensorDataset(x_cal, y_cal), batch_size=32)

model = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, n_classes))
# In a real use-case, load a pre-trained model here.

# %%
# 3. Fit DEUP on the calibration split
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ``DEUP.fit`` collects per-sample cross-entropy errors from the base model on the
# calibration loader, runs K-fold cross-validation to build OOF error estimates, and
# trains the final error predictor ``g`` on those estimates.

deup = DEUP(
    task="classification",
    model=model,
    num_folds=5,
    hidden_dim=32,
    max_epochs=30,
    device="cpu",
)
deup.fit(cal_loader)

# %%
# 4. Epistemic uncertainty at inference
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ``deup(x)`` takes raw inputs, runs them through the base model to extract features,
# and returns a non-negative epistemic score per sample.
# Higher scores signal that the base model is more likely to be unreliable on that input.

x_test = torch.randn(10, in_dim)
uncertainty = deup(x_test)
print("Epistemic scores g(x):", uncertainty)

probs = deup.predict_proba(x_test)
print("Base-model likelihood shape:", probs.shape)

# %%
# 5. Apply DEUP on CIFAR-10 with the ClassificationRoutine
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In this section we integrate DEUP with TorchUncertainty's
# :class:`~torch_uncertainty.routines.ClassificationRoutine` to evaluate OOD
# detection performance on CIFAR-10 versus SVHN.
#
# The routine fits the error predictor **automatically** on the validation
# split at the start of ``trainer.test()``, so no manual ``fit()`` call is
# needed here.

import torch
from huggingface_hub import hf_hub_download

from torch_uncertainty import TUTrainer
from torch_uncertainty.datamodules import CIFAR10DataModule
from torch_uncertainty.models.classification.resnet import resnet
from torch_uncertainty.post_processing import DEUP
from torch_uncertainty.routines import ClassificationRoutine

# %%
# 6. Load a pretrained ResNet-18 from Hugging Face
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We use a CIFAR-style ResNet-18 (3x3 first convolution, no max-pooling) from
# TorchUncertainty's HuggingFace hub. The CIFAR-style variant preserves more
# spatial information on small 32x32 images than the standard ImageNet variant.

cifar_model = resnet(in_channels=3, num_classes=10, arch=18, style="cifar", conv_bias=False)

ckpt_path = hf_hub_download(repo_id="torch-uncertainty/resnet18_c10", filename="resnet18_c10.ckpt")
weights = torch.load(ckpt_path, map_location="cpu", weights_only=True)
cifar_model.load_state_dict(weights)
cifar_model = cifar_model.cuda().eval()

# %%
# 7. DataModule and Trainer
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# :class:`~torch_uncertainty.datamodules.CIFAR10DataModule` with ``eval_ood=True``
# automatically provides the SVHN out-of-distribution test loader alongside the
# standard CIFAR-10 test loader.
#
# The key argument here is ``postprocess_set="val"``: it tells the routine to fit
# the DEUP error predictor on the *validation* split rather than the test set,
# avoiding any data leakage. We reserve 10% of the training set as validation via
# ``val_split=0.1``. The routine automatically builds this split before fitting
# DEUP, so no manual ``setup`` call is needed.

datamodule = CIFAR10DataModule(
    root=os.environ.get("TU_DATA_DIR", "data"),
    batch_size=256,
    num_workers=4,
    eval_ood=True,
    val_split=0.1,
    postprocess_set="val",
)

trainer = TUTrainer(
    accelerator="gpu",
    devices=1,
    max_epochs=1,
    enable_progress_bar=True,
)

# %%
# 8. ClassificationRoutine with DEUP post-processing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Passing ``post_processing=deup_c10`` and ``ood_criterion="deup"`` wires DEUP
# into the routine:
#
# - At the start of ``trainer.test()``, the routine calls ``deup_c10.fit()`` on the
#   validation dataloader (``postprocess_set="val"``).
# - During the test loop, the DEUP epistemic score is used as the OOD detection
#   criterion: higher score ⟹ more likely OOD.
#
# No loss or optimizer is needed since we are only running evaluation.

deup_c10 = DEUP(
    task="classification",
    hidden_dim=64,
    max_epochs=50,
    device="cuda",
)

routine = ClassificationRoutine(
    model=cifar_model,
    num_classes=10,
    loss=None,
    eval_ood=True,
    post_processing=deup_c10,
    ood_criterion="deup",
)

# %%
# 9. Evaluate OOD detection with DEUP
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ``trainer.test()`` runs the following sequence automatically:
#
# 1. Fit the DEUP error predictor on the CIFAR-10 validation split.
# 2. Compute in-distribution classification metrics on the CIFAR-10 test set.
# 3. Compute OOD detection metrics (AUROC, AUPR, FPR95) using SVHN as OOD data.
#
# OOD detection results are reported under the ``ood/`` prefix.

results_deup = trainer.test(routine, datamodule=datamodule)

# %%
# 10. Compare with the Maximum Softmax Probability baseline
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We can swap the OOD criterion to compare DEUP against the Maximum Softmax
# Probability (MSP) baseline — no re-training or re-fitting required.
# Only the OOD detection scores change; in-distribution metrics remain identical.

from torch_uncertainty.ood_criteria import MaxSoftmaxCriterion

routine.ood_criterion = MaxSoftmaxCriterion()

results_msp = trainer.test(routine, datamodule=datamodule)

# %%
# The ``ood/`` rows in both tables allow a direct comparison between the DEUP
# epistemic score and the MSP confidence score as OOD detectors on a well-trained
# ResNet-18.  DEUP is expected to complement confidence-based criteria by focusing
# on the epistemic component of the model's uncertainty.
#
# References
# ----------
#
# - **DEUP:** Lahlou, S., Jain, M., Nekoei, H., Butoi, V. I., Bertin, P.,
#   Rector-Brooks, J., ... & Bengio, Y. (2023). DEUP: Direct Epistemic Uncertainty
#   Prediction. TMLR. `openreview <https://openreview.net/forum?id=eGLdVRvvfQ>`_.
# - **ResNet:** He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning
#   for Image Recognition. CVPR 2016.
# - **MSP baseline:** Hendrycks, D., & Gimpel, K. (2017). A Baseline for Detecting
#   Misclassified and Out-of-Distribution Examples in Neural Networks. ICLR 2017.
