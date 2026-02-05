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

From source
-----------

To install the project from source, you can use pip directly.

Again, with PyTorch already installed, clone the repository with:

.. parsed-literal::

    git clone https://github.com/torch-uncertainty/torch-uncertainty.git
    cd torch-uncertainty

Create a new conda environment and activate it:

.. parsed-literal::

    conda create -n uncertainty python=3.12
    conda activate uncertainty

Install the package using pip in editable mode:

.. parsed-literal::

    pip install -e .

If PyTorch is not installed, the latest version will be installed automatically.

Options
-------

You can install the package with the following options:

* dev: includes all the dependencies for the development of the package
    including ruff and the pre-commits hooks.
* docs: includes all the dependencies for the documentation of the package
    based on sphinx
* image: includes all the dependencies for the image processing module
    including opencv and scikit-image
* tabular: includes pandas
* timeseries: includes tslearn
* others: with glest, laplace-torch and scikit-learn
* all: includes all the aforementioned dependencies

For example, to install the package with the dependencies for the development
and the documentation, run the following command. It is a mandatory step if you
want to contribute to the project.

.. parsed-literal::

    pip install -e .[dev,docs]

To install the package with all the dependencies, run:

.. parsed-literal::

    pip install -e .[all]
