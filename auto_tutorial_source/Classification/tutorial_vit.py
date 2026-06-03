"""ViT baslines with torch-uncertainty on imagenet1k
===============================================

This tutorial is about using torch-uncertainty to benchmark a ViT model on imagenet1k with various robustness metricis
and apply easily a postprocess step on top either of the single model or deep ensemble.

Dataset
-------

In this tutorial we will use imagenet1k dataset available directly through torch uncertainty a long with various far/near ood datasets
also handled automatically by torch-uncertainty.


1. Load the a single vit model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""

# %%
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from torchvision.models import vit_b_16


def load_model_from_hf(repo_id: str, filename: str, device: str = "cpu", revision: str = "main"):
    ckpt_path = hf_hub_download(
        repo_id=repo_id, filename=filename, repo_type="model", revision=revision
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)

    new_state = {}
    for k, v in state.items():
        name = k.removeprefix("model.")
        new_state[name] = v

    renamed = {}
    for k, v in new_state.items():
        if k == "heads.weight":
            renamed["heads.head.weight"] = v
        elif k == "heads.bias":
            renamed["heads.head.bias"] = v
        else:
            renamed[k] = v

    model = vit_b_16(weights=None, num_classes=1000, image_size=224)
    model.load_state_dict(renamed, strict=True)

    model.eval().to(device)
    return model


model1 = load_model_from_hf(
    repo_id="torch-uncertainty/vit-b-16-im1k",
    filename="model1.ckpt",
    device="cpu",
)


# %%
# 2. Benchmark the single model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We define first the imagnet1k datamodule then run the classification routine as follows.

from torch_uncertainty import TUTrainer
from torch_uncertainty.datamodules import ImageNetDataModule
from torch_uncertainty.routines import ClassificationRoutine

path = "./data"

dm = ImageNetDataModule(
    root=path,
    batch_size=512,
    num_workers=4,
    pin_memory=True,
    interpolation="bicubic",
    eval_ood=True,
)

trainer = TUTrainer(accelerator="gpu", enable_progress_bar=True, devices=1)

routine = ClassificationRoutine(
    num_classes=1000,
    model=model1,
    loss=nn.CrossEntropyLoss(),
    eval_ood=True,
)
res = trainer.test(routine, datamodule=dm)


# %%
# 3. Apply a postprocess step on top of the single model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Here we will be applying for example a temperature scaling postprocess step
# on top of the single model but torch-uncertainty supports many other postprocess please refer to the documentation.

from torch_uncertainty.post_processing import TemperatureScaler

dm.setup("fit")

scaler1 = TemperatureScaler(model=model1, device="cuda")
scaler1.cuda()
scaler1.fit(dataloader=dm.postprocess_dataloader())
print(scaler1.temperature[0])


routine = ClassificationRoutine(
    num_classes=1000,
    model=scaler1,
    loss=nn.CrossEntropyLoss(),
    eval_ood=True,
)
res = trainer.test(routine, datamodule=dm)


# %%
# 4. Load and benchmark a deep ensemble of ViT models
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let us load the remaining models of the deep ensemble and then benchmark them easily with torch unceratinty.

model2 = load_model_from_hf(
    repo_id="torch-uncertainty/vit-b-16-im1k",
    filename="model2.ckpt",
    device="cpu",
)

model3 = load_model_from_hf(
    repo_id="torch-uncertainty/vit-b-16-im1k",
    filename="model3.ckpt",
    device="cpu",
)


from torch_uncertainty.models import deep_ensembles

deep = deep_ensembles([model1, model2, model3])


routine = ClassificationRoutine(
    num_classes=1000,
    model=deep,
    loss=nn.CrossEntropyLoss(),
    eval_ood=True,
)
res = trainer.test(routine, datamodule=dm)

# %%
# Next, let us also apply a temperature scaling postprocess step on top of the deep ensemble.

dm.setup("fit")

scaler2 = TemperatureScaler(model=deep, device="cuda")
scaler2.cuda()
scaler2.fit(dataloader=dm.postprocess_dataloader())
print(scaler2.temperature[0])

routine = ClassificationRoutine(
    num_classes=1000,
    model=scaler2,
    loss=nn.CrossEntropyLoss(),
    eval_ood=True,
)
res = trainer.test(routine, datamodule=dm)

# %%
# 5. Load and benchmark packed ensemble of ViT model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let us load the packed ensemble vit and benchmark it with torch unceratinty.

from torch_uncertainty.models.classification.vit import PackedVit


def load_packedvit_from_hf(
    repo_id: str,
    filename: str,
    device: str = "cpu",
    revision: str = "main",
):
    ckpt_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model",
        revision=revision,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)

    clean_state = {
        k.replace("model.", "").replace("routine.model.", ""): v for k, v in state.items()
    }

    model = PackedVit(
        image_size=224,
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=774,
        mlp_dim=3072,
        num_classes=1000,
        num_estimators=3,
        alpha=2,
    )

    model.load_state_dict(clean_state, strict=True)
    model.eval().to(device)

    return model


packed = load_packedvit_from_hf(
    repo_id="torch-uncertainty/vit-b-16-im1k",
    filename="packed.ckpt",
    device="cpu",
)

routine = ClassificationRoutine(
    num_classes=1000,
    model=packed,
    loss=nn.CrossEntropyLoss(),
    eval_ood=True,
)
res = trainer.test(routine, datamodule=dm)


# %%
# Next, let us also apply a temperature scaling postprocess step on top of the packedvit.

dm.setup("fit")

scaler3 = TemperatureScaler(model=packed, device="cuda")
scaler3.cuda()
scaler3.fit(dataloader=dm.postprocess_dataloader())
print(scaler3.temperature[0])

routine = ClassificationRoutine(
    num_classes=1000,
    model=scaler3,
    loss=nn.CrossEntropyLoss(),
    eval_ood=True,
)
res = trainer.test(routine, datamodule=dm)
