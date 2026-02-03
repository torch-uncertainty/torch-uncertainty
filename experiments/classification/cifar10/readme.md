# CIFAR10 - Benchmark

This folder contains the code to train models on the CIFAR10 dataset. The task is to classify images into $10$ classes.

## ResNet-backbone models

`torch-uncertainty` leverages [LightningCLI](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.cli.LightningCLI.html#lightning.pytorch.cli.LightningCLI) the configurable command line tool for pytorch-lightning. To ease the train of models, we provide a set of predefined configurations for the CIFAR10 dataset (corresponding to the experiments reported in [Packed-Ensembles for Efficient Uncertainty Estimation](https://arxiv.org/abs/2210.09184)). The configurations are located in the `configs` folder.

*Examples:*

* Training a standard ResNet18 model as in [Packed-Ensembles for Efficient Uncertainty Estimation](https://arxiv.org/abs/2210.09184):

```bash
python main.py fit --config configs/resnet18/standard.yaml
```

* Training Packed-Ensembles ResNet50 model as in [Packed-Ensembles for Efficient Uncertainty Estimation](https://arxiv.org/abs/2210.09184):

```bash
python main.py fit --config configs/resnet50/packed.yaml
```

## Available configurations:

### ResNet

|                  | ResNet18 | ResNet34 | ResNet50 | ResNet101 | ResNet152 |
| ---------------- | -------- | -------- | -------- | --------- | --------- |
| Standard         | ✅        | ✅        | ✅        | ✅         | ✅         |
| Packed-Ensembles | ✅        | ✅        | ✅        | ✅         | ✅         |
| BatchEnsemble    | ✅        | ✅        | ✅        | ✅         | ✅         |
| Masked-Ensembles | ✅        | ✅        | ✅        | ✅         | ✅         |
| MIMO             | ✅        | ✅        | ✅        | ✅         | ✅         |
| MC Dropout       | ✅        | ✅        | ✅        | ✅         | ✅         |


### WideResNet

|                  | WideResNet28-10 |
| ---------------- | --------------- |
| Standard         | ✅               |
| Packed-Ensembles | ✅               |
| BatchEnsemble    | ✅               |
| Masked-Ensembles | ✅               |
| MIMO             | ✅               |
| MC Dropout       | ✅               |
