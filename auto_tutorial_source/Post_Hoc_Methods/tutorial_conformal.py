# ruff: noqa: D212, D415, T201
"""
Conformal Prediction on CIFAR-10 with TorchUncertainty
======================================================

We evaluate the model's performance both before and after applying different conformal predictors (THR, APS, RAPS), and visualize how conformal prediction estimates the prediction sets.

We use the pretrained ResNet models we provide on Hugging Face.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import hf_hub_download

from torch_uncertainty import TUTrainer
from torch_uncertainty.datamodules import CIFAR10DataModule
from torch_uncertainty.models.classification.resnet import resnet
from torch_uncertainty.post_processing import ConformalClsAPS, ConformalClsRAPS, ConformalClsTHR
from torch_uncertainty.routines import ClassificationRoutine

# %%
# 1. Load pretrained model from Hugging Face repository
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We use a ResNet18 model trained on CIFAR-10, provided by the TorchUncertainty team

ckpt_path = hf_hub_download(repo_id="torch-uncertainty/resnet18_c10", filename="resnet18_c10.ckpt")
model = resnet(in_channels=3, num_classes=10, arch=18, conv_bias=False, style="cifar")
ckpt = torch.load(ckpt_path, weights_only=True)
model.load_state_dict(ckpt)
model = model.cuda().eval()

# %%
# 2. Load CIFAR-10 Dataset & Define Dataloaders
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We set eval_ood to True to evaluate the performance of Conformal scores for detecting out-of-distribution
# samples. In this case, since we use a model trained on the full training set, we use the test set to as calibration
# set for the Conformal methods and for its evaluation. This is not a proper way to evaluate the coverage.

BATCH_SIZE = 128

datamodule = CIFAR10DataModule(
    root="./data",
    batch_size=BATCH_SIZE,
    num_workers=8,
    eval_ood=True,
    postprocess_set="test",
)
datamodule.prepare_data()
datamodule.setup()


# %%
# 3. Define the Lightning Trainer
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

trainer = TUTrainer(accelerator="gpu", devices=1, max_epochs=5, enable_progress_bar=False)


# %%
# 4. Function to Visualize the Prediction Sets
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def visualize_prediction_sets(inputs, labels, confidence_scores, classes, num_examples=5) -> None:
    _, axs = plt.subplots(2, num_examples, figsize=(15, 5))
    for i in range(num_examples):
        ax = axs[0, i]
        img = np.clip(
            inputs[i].permute(1, 2, 0).cpu().numpy() * datamodule.std + datamodule.mean, 0, 1
        )
        ax.imshow(img)
        ax.set_title(f"True: {classes[labels[i]]}")
        ax.axis("off")
        ax = axs[1, i]
        for j in range(len(classes)):
            ax.barh(classes[j], confidence_scores[i, j], color="blue")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Confidence Score")
    plt.tight_layout()
    plt.show()


# %%
# 5. Estimate Prediction Sets with ConformalClsTHR
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Using alpha=0.01, we aim for a 1% error rate.

print("[Phase 2]: ConformalClsTHR calibration")
conformal_model = ConformalClsTHR(alpha=0.01, device="cuda")

routine_thr = ClassificationRoutine(
    num_classes=10,
    model=model,
    loss=None,  # No loss needed for evaluation
    eval_ood=True,
    post_processing=conformal_model,
    ood_criterion="post_processing",
)
perf_thr = trainer.test(routine_thr, datamodule=datamodule)

# %%
# 6. Visualization of ConformalClsTHR prediction sets
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

inputs, labels = next(iter(datamodule.test_dataloader()[0]))

conformal_model.cuda()
confidence_scores = conformal_model.conformal(inputs.cuda())

classes = datamodule.test.classes

visualize_prediction_sets(inputs, labels, confidence_scores[:5].cpu(), classes)

# %%
# 7. Estimate Prediction Sets with ConformalClsAPS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print("[Phase 3]: ConformalClsAPS calibration")
conformal_model = ConformalClsAPS(alpha=0.01, device="cuda", enable_ts=True)

routine_aps = ClassificationRoutine(
    num_classes=10,
    model=model,
    loss=None,  # No loss needed for evaluation
    eval_ood=True,
    post_processing=conformal_model,
    ood_criterion="post_processing",
)
perf_aps = trainer.test(routine_aps, datamodule=datamodule)
conformal_model.cuda()
confidence_scores = conformal_model.conformal(inputs.cuda())
visualize_prediction_sets(inputs, labels, confidence_scores[:5].cpu(), classes)

# %%
# 8. Estimate Prediction Sets with ConformalClsRAPS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print("[Phase 4]: ConformalClsRAPS calibration")
conformal_model = ConformalClsRAPS(
    alpha=0.01, regularization_rank=3, penalty=0.002, model=model, device="cuda", enable_ts=True
)

routine_raps = ClassificationRoutine(
    num_classes=10,
    model=model,
    loss=None,  # No loss needed for evaluation
    eval_ood=True,
    post_processing=conformal_model,
    ood_criterion="post_processing",
)
perf_raps = trainer.test(routine_raps, datamodule=datamodule)
conformal_model.cuda()
confidence_scores = conformal_model.conformal(inputs.cuda())
visualize_prediction_sets(inputs, labels, confidence_scores[:5].cpu(), classes)

# %%
# Summary
# -------
#
# In this tutorial, we explored how to apply conformal prediction to a pretrained ResNet on CIFAR-10.
# We evaluated three methods: Thresholding (THR), Adaptive Prediction Sets (APS), and Regularized APS (RAPS).
# For each, we calibrated on a validation set, evaluated OOD performance, and visualized prediction sets.
# You can explore further by adjusting `alpha`, changing the model, or testing on other datasets.
