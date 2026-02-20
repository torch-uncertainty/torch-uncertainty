<div align="center">

![TorchUncertaintyLogo](https://github.com/torch-uncertainty/torch-uncertainty/blob/main/docs/source/_static/images/torch_uncertainty.png)

[![pypi](https://img.shields.io/pypi/v/torch_uncertainty.svg)](https://pypi.python.org/pypi/torch_uncertainty)
[![tests](https://github.com/torch-uncertainty/torch-uncertainty/actions/workflows/run-tests.yml/badge.svg?branch=main&event=push)](https://github.com/torch-uncertainty/torch-uncertainty/actions/workflows/run-tests.yml)
[![Docs](https://github.com/torch-uncertainty/torch-uncertainty/actions/workflows/build-docs.yml/badge.svg)](https://torch-uncertainty.github.io/)
[![PRWelcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/torch-uncertainty/torch-uncertainty/pulls)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code Coverage](https://codecov.io/github/torch-uncertainty/torch-uncertainty/coverage.svg?branch=master)](https://codecov.io/gh/torch-uncertainty/torch-uncertainty)
[![Downloads](https://static.pepy.tech/badge/torch-uncertainty)](https://pepy.tech/project/torch-uncertainty)
</div>

_TorchUncertainty_ is a package designed to help leverage [uncertainty quantification techniques](https://github.com/ENSTA-U2IS-AI/awesome-uncertainty-deeplearning) to make deep neural networks more reliable. It aims at being collaborative and including as many methods as possible, so reach out to add yours!

:construction: _TorchUncertainty_ is in early development :construction: - expect changes, but reach out and contribute if you are interested in the project! **Please raise an issue if you have any bugs or difficulties and join the [discord server](https://discord.gg/HMCawt5MJu).**

:books: Our webpage and documentation is available here: [torch-uncertainty.github.io](https://torch-uncertainty.github.io). :books:

TorchUncertainty contains the _official implementations_ of multiple papers from _major machine-learning and computer vision conferences_ and was featured in tutorials at **[WACV](https://wacv2024.thecvf.com/) 2024**, **[HAICON](https://haicon24.de/) 2024** and **[ECCV](https://eccv.ecva.net/) 2024**.

Torch-Uncertainty is published at [NeurIPS D&B 2025](https://neurips.cc/virtual/2025/loc/san-diego/poster/121463). Please consider citing the paper if the framework is helpful for your research.

---

This package provides a multi-level API, including:

- easy-to-use :zap: lightning **uncertainty-aware** training & evaluation routines for **4 tasks**: classification, probabilistic and pointwise regression, and segmentation.
- fully automated evaluation of the performance of models with proper scores, selective classification, out-of-distribution detection and distribution shift performance metrics!
- ready-to-train baselines on research datasets, such as ImageNet and CIFAR
- **layers**, **models**, **metrics**, & **losses** available for your networks
- scikit-learn style post-processing methods such as Temperature Scaling.
- transformations and augmentations, including corruptions resulting in additional "corrupted datasets" available on [HuggingFace](https://huggingface.co/torch-uncertainty)

Have a look at the [Reference page](https://torch-uncertainty.github.io/references.html) or the [API reference](https://torch-uncertainty.github.io/api.html) for a more exhaustive list of the implemented methods, datasets, metrics, etc.

## :gear: Installation

TorchUncertainty requires Python 3.10 or greater. Install the desired PyTorch version in your environment.
Then, install the package from PyPI:

```sh
pip install torch-uncertainty
```

The installation procedure for contributors is different: have a look at the [contribution page](https://torch-uncertainty.github.io/contributing.html).

### :whale: Docker image for contributors

For contributors running experiments on cloud GPU instances, we provide a pre-built Docker image that includes all necessary dependencies and configurations and the Dockerfile for building your custom Docker images.
This allows you to quickly launch an experiment-ready container with minimal setup. Please refer to [DOCKER.md](docker/DOCKER.md) for further details.

## :racehorse: Quickstart

We make a quickstart available at [torch-uncertainty.github.io/quickstart](https://torch-uncertainty.github.io/quickstart.html).

## :books: Implemented methods

TorchUncertainty currently supports **classification**, **probabilistic** and pointwise **regression**, **segmentation** and **pixelwise regression** (such as monocular depth estimation).

We also provide the following methods:

### Uncertainty quantification models

To date, the following deep learning uncertainty quantification modes have been implemented. **Click** :inbox_tray: **on the methods for tutorials**:

- [Deep Ensembles](https://torch-uncertainty.github.io/auto_tutorials/Classification/tutorial_from_de_to_pe.html), BatchEnsemble, Masksembles, & MIMO
- [MC-Dropout](https://torch-uncertainty.github.io/auto_tutorials/Bayesian_Methods/tutorial_mc_dropout.html)
- [Packed-Ensembles](https://torch-uncertainty.github.io/auto_tutorials/Classification/tutorial_from_de_to_pe.html) (see [Blog post](https://medium.com/@adrien.lafage/make-your-neural-networks-more-reliable-with-packed-ensembles-7ad0b737a873))
- Mixup Ensembles and MixupMP
- [Variational Bayesian Neural Networks](https://torch-uncertainty.github.io/auto_tutorials/Bayesian_Methods/tutorial_bayesian.html)
- Checkpoint Ensembles & Snapshot Ensembles
- Stochastic Weight Averaging & Stochastic Weight Averaging Gaussian
- [Deep Evidential Classification](https://torch-uncertainty.github.io/auto_tutorials/Classification/tutorial_evidential_classification.html) & [Regression](https://torch-uncertainty.github.io/auto_tutorials/Regression/tutorial_der_cubic.html)
- Regression with Beta Gaussian NLL Loss
- Test-time adaptation with Zero

### Augmentation methods

The following data augmentation methods have been implemented:

- Mixup, MixupIO, RegMixup, WarpingMixup
- Modernized corruptions to evaluate model performance under distribution shift

### Post-processing methods

To date, the following post-processing methods have been implemented:

- [Temperature](https://torch-uncertainty.github.io/auto_tutorials/Post_Hoc_Methods/tutorial_scaler.html), Vector, & Matrix scaling
- [Conformal Predictions](https://torch-uncertainty.github.io/auto_tutorials/Post_Hoc_Methods/tutorial_conformal.html) with APS and RAPS
- [Monte Carlo Batch Normalization](https://torch-uncertainty.github.io/auto_tutorials/Bayesian_Methods/tutorial_mc_batch_norm.html)
- Laplace approximation through the [Laplace library](https://github.com/aleximmer/Laplace)

### Official Implementations

It includes the official codes of the following papers:

- _Packed-Ensembles for Efficient Uncertainty Estimation_ - [ICLR 2023](https://arxiv.org/abs/2210.09184) - [Tutorial](https://torch-uncertainty.github.io/auto_tutorials/Classification/tutorial_pe_cifar10.html)
- _LP-BNN: Encoding the latent posterior of Bayesian Neural Networks for uncertainty quantification_ - [IEEE TPAMI 2023](https://arxiv.org/abs/2012.02818)
- _MUAD: Multiple Uncertainties for Autonomous Driving, a benchmark for multiple uncertainty types and tasks_ - [BMVC 2022](https://arxiv.org/abs/2203.01437)

## Tutorials

Check out all our tutorials at [torch-uncertainty.github.io/auto_tutorials](https://torch-uncertainty.github.io/auto_tutorials/index.html).

## Experiments

We provide some configuration files in the `experiments` folder. You may find more examples in other repositories related to the Torch-Uncertainty organization, such as the [Packed-Ensembles reproducibility repo.](https://github.com/torch-uncertainty/packed-ensembles-ICLR-experiments) or the [Torch-Uncertainty reproducibility repo](https://github.com/torch-uncertainty/torch-uncertainty-neurIPS-experiments).

## :telescope: Projects using TorchUncertainty

The following projects use TorchUncertainty:

- _Towards Understanding and Quantifying Uncertainty for Text-to-Image Generation_ - [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Franchi_Towards_Understanding_and_Quantifying_Uncertainty_for_Text-to-Image_Generation_CVPR_2025_paper.pdf)
- _Towards Understanding Why Label Smoothing Degrades Selective Classification and How to Fix It_ - [ICLR 2025](https://arxiv.org/abs/2403.14715)
- _A Symmetry-Aware Exploration of Bayesian Neural Network Posteriors_ - [ICLR 2024](https://arxiv.org/abs/2310.08287)

**If you are using TorchUncertainty in your project, please let us know, and we will add your project to this list!**

## Citation

If you use this software, please cite its corresponding paper:

```bibtex
@inproceedings{lafage2025torch_uncertainty,
    title={Torch-Uncertainty: A Deep Learning Framework for Uncertainty Quantification},
    author={Lafage, Adrien and Laurent, Olivier and Gabetni, Firas and Franchi, Gianni},
    booktitle={NeurIPS Datasets and Benchmarks Track},
    year={2025}
}
```
