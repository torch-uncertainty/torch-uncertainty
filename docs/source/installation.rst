Installation
============

.. role:: bash(code)
    :language: bash


You can install the package either from PyPI or from source. Choose the latter if you
want to access the files included the `experiments <https://github.com/torch-uncertainty/torch-uncertainty/tree/main/experiments>`_
folder or if you want to contribute to the project.


From PyPI
---------

Check that you have Python 3.10 (or later) and  PyTorch (cpu or gpu) installed on your system. Then, install
the package via pip:

.. parsed-literal::

    pip install torch-uncertainty

To update the package, run:

.. parsed-literal::

    pip install -U torch-uncertainty

Options
-------

You can install the package with the following options:

* dev: includes all the dependencies for the development of the package
    including ruff, the pre-commits hooks, and sphinx for the documentation.
* experiment: includes all the dependencies to make use of the `experiments` folder including
    tensorboard, huggingface-hub, and safetensors.
* image: includes all the dependencies for the image processing module
    including opencv, kornia, h5py, and torch-uncertainty-assets
* distribution: include scipy
* timeseries: includes tslearn
* others: with curvlinops-for-pytorch, glest, laplace-torch and scikit-learn
* all: includes all the aforementioned dependencies

Example:

.. parsed-literal::

    pip install torch-uncertainty[image]

From source
-----------

To install the project from source, you can use pip directly.

Again, with PyTorch already installed, clone the repository with:

.. parsed-literal::

    git clone https://github.com/torch-uncertainty/torch-uncertainty.git
    cd torch-uncertainty

To install the package and all its dependencies, we recommend to use `uv <https://docs.astral.sh/uv/getting-started/installation/>`_:

.. parsed-literal::

    # If no NVIDIA gpu
    uv sync --extra cpu
    
    # or
    uv sync --extra gpu
