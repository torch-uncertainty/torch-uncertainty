.. TorchUncertainty documentation master file, created by
   sphinx-quickstart on Wed Feb  1 18:07:01 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Torch Uncertainty
============================

.. role:: bash(code)
   :language: bash

Welcome to the documentation of TorchUncertainty.

This website contains the documentation for
`installing <https://torch-uncertainty.github.io/installation.html>`_
and `contributing <https://torch-uncertainty.github.io/>`_ to TorchUncertainty,
details on the `API <https://torch-uncertainty.github.io/api.html>`_, and a
`comprehensive list of the references <https://torch-uncertainty.github.io/references.html>`_  of
the models and metrics implemented.

-----

Installation
^^^^^^^^^^^^

Make sure you have Python 3.10 or later installed, as well as PyTorch (CPU or GPU).

.. code-block:: bash

   pip install torch-uncertainty

To install TorchUncertainty with contribution in mind, check the
`contribution page <https://torch-uncertainty.github.io/contributing.html>`_.

-----

Official Implementations
^^^^^^^^^^^^^^^^^^^^^^^^

TorchUncertainty also houses multiple official implementations of papers from major conferences & journals.

**A Symmetry-Aware Exploration of Bayesian Neural Network Posteriors**

* Authors: *Olivier Laurent, Emanuel Aldea, and Gianni Franchi*
* Paper: `ICLR 2024 <https://arxiv.org/abs/2310.08287>`_.

**Encoding the latent posterior of Bayesian Neural Networks for uncertainty quantification**

* Authors: *Gianni Franchi, Andrei Bursuc, Emanuel Aldea, Severine Dubuisson, and Isabelle Bloch*
* Paper: `IEEE TPAMI <https://arxiv.org/abs/2012.02818>`_.

**Packed-Ensembles for Efficient Uncertainty Estimation**

* Authors: *Olivier Laurent, Adrien Lafage, Enzo Tartaglione, Geoffrey Daniel, Jean-Marc Martinez, Andrei Bursuc, and Gianni Franchi*
* Paper: `ICLR 2023 <https://arxiv.org/abs/2210.09184>`_.

**MUAD: Multiple Uncertainties for Autonomous Driving, a benchmark for multiple uncertainty types and tasks**

* Authors: *Gianni Franchi, Xuanlong Yu, Andrei Bursuc, Angel Tena, Rémi Kazmierczak, Séverine Dubuisson, Emanuel Aldea, David Filliat*
* Paper: `BMVC 2022 <https://arxiv.org/abs/2203.01437>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   auto_tutorials/index
   evaluation
   cli_guide
   mlflow_guide
   api
   contributing
   references

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
