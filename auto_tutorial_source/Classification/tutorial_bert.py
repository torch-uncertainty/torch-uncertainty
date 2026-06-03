"""Benchamrk bert with torch-uncertainty on SST2
===============================================

This tutorial is about using torch-uncertainty to benchmark a bert model on the sst2 dataset with various robustness metricis
and apply easily a postprocess step (MC dropout) on top either of the single model or deep ensemble.

Dataset
-------

In this tutorial we will use sst2 dataset available directly through torch uncertainty a long with various far/near ood datasets
also handled automatically by torch-uncertainty.


1. Define and load the single bert model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""

# %%
from collections import OrderedDict

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification


def load_tu_ckpt_into_hf(
    backbone, repo_id: str, filename: str, strict: bool = True, map_location="cpu"
):
    ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)

    sd = torch.load(ckpt_path, map_location=map_location)
    sd = sd.get("state_dict", sd)

    def with_prefix(prefix):
        return OrderedDict((k[len(prefix) :], v) for k, v in sd.items() if k.startswith(prefix))

    for pref in ("model.backbone.", "model.", "backbone."):
        sub = with_prefix(pref)
        if sub:
            return backbone.load_state_dict(sub, strict=strict)

    return backbone.load_state_dict(sd, strict=strict)


class HFClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int = 2, local_files_only: bool = False):
        super().__init__()
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, local_files_only=local_files_only
        )

    def forward(self, *args, **kwargs):
        inputs = args[0] if (len(args) == 1 and isinstance(args[0], dict)) else kwargs
        return self.backbone(**inputs).logits


net1 = HFClassifier("bert-base-uncased", num_labels=2)

load_tu_ckpt_into_hf(
    net1.backbone,
    repo_id="torch-uncertainty/bert-sst2",
    filename="model1.ckpt",
)


# %%
# 2. Benchmark the single model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We define first the sst2 datamodule then run the classification routine as follows.

from torch_uncertainty import TUTrainer
from torch_uncertainty.datamodules import Sst2DataModule
from torch_uncertainty.routines import ClassificationRoutine

dm = Sst2DataModule(
    batch_size=64,
    eval_ood=True,
)

trainer = TUTrainer(accelerator="gpu", enable_progress_bar=True, devices=1)

routine = ClassificationRoutine(
    num_classes=2,
    model=net1,
    loss=nn.CrossEntropyLoss(),
    eval_ood=True,
)

res = trainer.test(routine, datamodule=dm)


# %%
# 3. Apply a postprocess step on top of the single model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Here we will be applying for example montecarlo dropout on top of the single model.
# but torch-uncertainty supports many other postprocess like temperature scaling,conformal... please refer to the documentation.

from torch_uncertainty.models import mc_dropout

mc_net = mc_dropout(
    model=net1,
    num_estimators=8,
    on_batch=False,
)

routine = ClassificationRoutine(
    num_classes=2,
    model=mc_net,
    loss=nn.CrossEntropyLoss(),
    eval_ood=True,
)

res = trainer.test(routine, datamodule=dm)


# %%
# 4. Load and benchmark a deep ensemble of bert models
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let us load the remaining models of the deep ensemble and then benchmark them easily with torch unceratinty.

net2 = HFClassifier("bert-base-uncased", num_labels=2)

load_tu_ckpt_into_hf(
    net2.backbone,
    repo_id="torch-uncertainty/bert-sst2",
    filename="model2.ckpt",
)


net3 = HFClassifier("bert-base-uncased", num_labels=2)

load_tu_ckpt_into_hf(
    net3.backbone,
    repo_id="torch-uncertainty/bert-sst2",
    filename="model3.ckpt",
)


from torch_uncertainty.models import deep_ensembles

deep = deep_ensembles([net1, net2, net3])


routine = ClassificationRoutine(
    num_classes=2,
    model=deep,
    loss=nn.CrossEntropyLoss(),
    eval_ood=True,
)
res = trainer.test(routine, datamodule=dm)
