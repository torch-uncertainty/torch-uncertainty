Contributing
============

.. role:: bash(code)
    :language: bash

.. role:: cmd(code)
    :language: bash

TorchUncertainty is in early development stage. We are looking for
contributors to help us build a comprehensive library for uncertainty
quantification in PyTorch.

We are particularly open to any comment that you would have on this project.
Specifically, we are open to changing these guidelines as the project evolves.

The scope of TorchUncertainty
-----------------------------

TorchUncertainty can host any method - if possible linked to a paper - and
roughly contained in the following fields:

* Uncertainty quantification in general, including Bayesian deep learning, Monte Carlo dropout, ensemble methods, etc.
* Out-of-distribution detection methods
* Applications (e.g. object detection, segmentation, etc.)

Common guidelines
-----------------

Clean development install of TorchUncertainty
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are interested in contributing to torch_uncertainty, we first advise you
to follow the following steps to reproduce a clean development environment
ensuring continuous integration does not break.

1. Install ``uv`` following the steps `here <https://docs.astral.sh/uv/getting-started/installation/>`_
2. Clone the repository
3. Install torch-uncertainty with the dev packages 
   :cmd:`uv sync --extra gpu` for GPU-based systems
   :cmd:`uv sync --extra cpu` if no GPUs are available
4. Install pre-commit hooks with :cmd:`uv run pre-commit install`

Note: failure to include the extra flag will result in the GPU version of PyTorch without CUDA and might cause issues.

Build the documentation locally
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To build the documentation on Linux, navigate to ``./docs`` and build the documentation with:

.. parsed-literal::

    make html

Optionally, specify ``html-noplot`` instead of ``html`` to avoid running the tutorials.

Guidelines
^^^^^^^^^^

**Commits**

We use ruff for code formatting, linting, and imports (as a drop-in
replacement for black, isort, and flake8). The pre-commit hooks will ensure
that your code is properly formatted and linted before committing.

Please ensure that the tests are passing on your machine before pushing on a
PR. This will avoid multiplying the number featureless commits. To do this,
run, at the root of the folder:

.. parsed-literal::

    uv run pytest tests

Try to include an emoji at the start of each commit message following the suggestions
from `this page <https://gist.github.com/parmentf/035de27d6ed1dce0b36a>`_.

**Pull requests**

To make your changes, create a branch on a personal fork and create a PR when your contribution
is mostly finished or if you need help.

Check that your PR complies with the following conditions:

* The name of your branch is not ``main`` nor ``dev`` (see issue #58)
* Your PR does not reduce the code coverage
* Your code is documented: the function signatures are typed, and the main functions have clear docstrings
* Your code is mostly original, and the parts coming from licensed sources are explicitly stated as such
* If you implement a method, please add a reference to the corresponding paper in the 
  `references page <https://torch-uncertainty.github.io/references.html>`_.
* Also, remember to add TorchUncertainty to the list of libraries implementing this reference
  on `PapersWithCode <https://paperswithcode.com>`_.

If you need help to implement a method, increase the coverage, or solve ruff-raised errors,
create the PR with the ``need-help`` flag and explain your problems in the comments. A maintainer
will do their best to help you.

Datasets & Datamodules
^^^^^^^^^^^^^^^^^^^^^^

We intend to include datamodules for the most popular datasets only.

Post-processing methods
^^^^^^^^^^^^^^^^^^^^^^^

For now, we intend to follow scikit-learn style API for post-processing
methods (except that we use a validation dataset instead of a numpy array).
You may get inspiration from the already implemented
`temperature-scaling <https://github.com/torch-uncertainty/torch-uncertainty/blob/dev/torch_uncertainty/post_processing/calibration/temperature_scaler.py>`_.


License
-------

If you feel that the current license is an obstacle to your contribution, let
us know, and we may reconsider. However, the models’ weights hosted on Hugging
Face are likely to remain Apache 2.0.
